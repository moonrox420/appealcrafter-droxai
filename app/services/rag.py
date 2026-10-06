"""RAG pipeline: ingestion → semantic chunking → hybrid retrieval → citation context.

Implements the full RAG pipeline:
1. Approved ingestion (only approved documents are chunked and embedded)
2. Semantic chunking with overlap
3. Hybrid retrieval (semantic pgvector + keyword search)
4. Citation-bearing context for constrained generation
5. Permanent template fallback when retrieval is insufficient
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import DocumentChunk, KnowledgeDocument

logger = logging.getLogger(__name__)

CHUNK_SIZE_WORDS: int = 200
CHUNK_OVERLAP_WORDS: int = 50


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved knowledge chunk with citation metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    content: str
    similarity_score: float
    source_url: str | None


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a hybrid retrieval query."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""


class EmbeddingProvider:
    """Generate embeddings using sentence-transformers with lazy loading."""

    _model = None

    def __init__(self) -> None:
        self.settings = get_settings()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text strings."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.settings.llm.embedding_model_name
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; using fallback hashing embeddings"
                )
                return [self._fallback_embedding(text) for text in texts]
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [embedding.tolist() for embedding in embeddings]

    def _fallback_embedding(self, text: str) -> list[float]:
        """Generate a deterministic hashing-based fallback embedding vector."""
        vector: list[float] = []
        for index in range(self.settings.llm.embedding_dimensions):
            digest = hashlib.sha256(f"{text}:{index}".encode()).hexdigest()
            value = int(digest[:8], 16) / 0xFFFFFFFF
            vector.append(value * 2.0 - 1.0)
        return vector

    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        return self.embed_texts([text])[0]


class SemanticChunker:
    """Chunk knowledge documents into overlapping semantic segments."""

    def chunk_document(self, content: str) -> list[str]:
        """Split document content into overlapping word-based chunks."""
        words = content.split()
        if len(words) <= CHUNK_SIZE_WORDS:
            return [content]

        chunks: list[str] = []
        start_index = 0
        while start_index < len(words):
            end_index = min(start_index + CHUNK_SIZE_WORDS, len(words))
            chunk_text = " ".join(words[start_index:end_index])
            chunks.append(chunk_text)
            if end_index >= len(words):
                break
            start_index = end_index - CHUNK_OVERLAP_WORDS
        return chunks


class RagPipeline:
    """Coordinate ingestion, embedding, retrieval, and citation building."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session
        self.settings = get_settings()
        self.embedding_provider = EmbeddingProvider()
        self.chunker = SemanticChunker()

    def ingest_document(self, document_id: str) -> int:
        """Chunk and embed an approved knowledge document."""
        document = self.db_session.get(KnowledgeDocument, document_id)
        if document is None:
            raise ValueError(f"Knowledge document {document_id} not found.")
        if not document.is_approved:
            raise ValueError("Only approved documents can be ingested into RAG.")

        chunks = self.chunker.chunk_document(document.content)
        embeddings = self.embedding_provider.embed_texts(chunks)

        for existing_chunk in document.chunks:
            self.db_session.delete(existing_chunk)
        self.db_session.flush()

        for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk_text,
                embedding_vector=embedding,
            )
            self.db_session.add(chunk)

        self.db_session.commit()
        logger.info(
            "Knowledge document ingested",
            extra={"document_id": document.id, "chunk_count": len(chunks)},
        )
        return len(chunks)

    def retrieve(self, query: str, limit: int = 5) -> RetrievalResult:
        """Perform hybrid retrieval combining semantic and keyword search."""
        query_embedding = self.embedding_provider.embed_single(query)
        query_terms = [term.lower() for term in query.split() if len(term) > 3]

        semantic_results = self.db_session.scalars(
            select(DocumentChunk)
            .join(KnowledgeDocument, DocumentChunk.document_id == KnowledgeDocument.id)
            .where(
                KnowledgeDocument.is_approved.is_(True),
                DocumentChunk.embedding_vector.isnot(None),
            )
            .order_by(DocumentChunk.embedding_vector.cosine_distance(query_embedding))
            .limit(limit * 2)
        ).all()

        keyword_results = self.db_session.scalars(
            select(DocumentChunk)
            .join(KnowledgeDocument, DocumentChunk.document_id == KnowledgeDocument.id)
            .where(
                KnowledgeDocument.is_approved.is_(True),
                or_(
                    *(DocumentChunk.content.ilike(f"%{term}%") for term in query_terms)
                ),
            )
            .limit(limit * 2)
        ).all()

        merged_chunks: dict[str, RetrievedChunk] = {}
        for semantic_chunk in semantic_results:
            distance = (
                semantic_chunk.embedding_vector.cosine_distance(query_embedding)
                if semantic_chunk.embedding_vector
                else 1.0
            )
            similarity_score = max(0.0, 1.0 - float(distance))
            merged_chunks[semantic_chunk.id] = RetrievedChunk(
                chunk_id=semantic_chunk.id,
                document_id=semantic_chunk.document_id,
                document_title=(
                    semantic_chunk.document.title
                    if semantic_chunk.document
                    else semantic_chunk.document_id
                ),
                content=semantic_chunk.content,
                similarity_score=similarity_score,
                source_url=(
                    semantic_chunk.document.source_url
                    if semantic_chunk.document
                    else None
                ),
            )

        for keyword_chunk in keyword_results:
            if keyword_chunk.id in merged_chunks:
                existing = merged_chunks[keyword_chunk.id]
                combined_score = existing.similarity_score + 0.15
                merged_chunks[keyword_chunk.id] = RetrievedChunk(
                    chunk_id=existing.chunk_id,
                    document_id=existing.document_id,
                    document_title=existing.document_title,
                    content=existing.content,
                    similarity_score=combined_score,
                    source_url=existing.source_url,
                )
            else:
                merged_chunks[keyword_chunk.id] = RetrievedChunk(
                    chunk_id=keyword_chunk.id,
                    document_id=keyword_chunk.document_id,
                    document_title=(
                        keyword_chunk.document.title
                        if keyword_chunk.document
                        else keyword_chunk.document_id
                    ),
                    content=keyword_chunk.content,
                    similarity_score=0.5,
                    source_url=(
                        keyword_chunk.document.source_url
                        if keyword_chunk.document
                        else None
                    ),
                )

        ranked_chunks = sorted(
            merged_chunks.values(),
            key=lambda retrieved_chunk: retrieved_chunk.similarity_score,
            reverse=True,
        )[:limit]
        return RetrievalResult(chunks=ranked_chunks, query=query)

    def build_citation_context(self, retrieval_result: RetrievalResult) -> str:
        """Build a citation-bearing context string for constrained generation."""
        citation_sections: list[str] = []
        for chunk in retrieval_result.chunks:
            citation = f"[Source: {chunk.document_title} | Similarity: {chunk.similarity_score:.3f}]"
            if chunk.source_url:
                citation = f"[Source: {chunk.document_title} ({chunk.source_url}) | Similarity: {chunk.similarity_score:.3f}]"
            citation_sections.append(f"{citation}\n{chunk.content}")
        return "\n\n".join(citation_sections)


def get_rag_pipeline(db_session: Session) -> RagPipeline:
    """Return a configured RAG pipeline instance."""
    return RagPipeline(db_session)
