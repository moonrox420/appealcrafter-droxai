"""A/B experiment service with sticky assignment and statistical analysis.

Implements configurable A/B and multivariate experiments with sticky
assignment, statistical tests on conversion data, full lifecycle
persistence, and promotion of winning variants with audit trail.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from statistics import NormalDist

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.metrics import EXPERIMENT_EVENTS
from app.models.entities import (
    Appeal,
    Delivery,
    DeliveryStatus,
    Experiment,
    ExperimentAssignment,
    ExperimentStatus,
    ExperimentVariant,
    Template,
)

logger = logging.getLogger(__name__)

DEFAULT_SIGNIFICANCE_LEVEL: float = 0.05
MIN_SAMPLE_SIZE_PER_VARIANT: int = 50


class ExperimentService:
    """Manage experiment lifecycle, assignment, and promotion."""

    def __init__(self, db_session: Session) -> None:
        self.db_session = db_session

    def assign_variant(
        self, experiment: Experiment, subject_key: str
    ) -> ExperimentVariant:
        """Return a sticky-assigned variant for a subject."""
        existing_assignment = (
            self.db_session.execute(
                select(ExperimentAssignment).where(
                    ExperimentAssignment.experiment_id == experiment.id,
                    ExperimentAssignment.subject_key == subject_key,
                )
            )
            .scalars()
            .first()
        )

        if existing_assignment is not None:
            return self.db_session.get(
                ExperimentVariant, existing_assignment.variant_id
            )

        variants = list(
            self.db_session.execute(
                select(ExperimentVariant).where(
                    ExperimentVariant.experiment_id == experiment.id
                )
            )
            .scalars()
            .all()
        )
        if not variants:
            raise ValueError(f"Experiment {experiment.id} has no variants.")

        total_weight = sum(variant.weight for variant in variants)
        hash_digest = hashlib.sha256(
            f"{experiment.id}:{subject_key}".encode()
        ).hexdigest()
        hash_value = int(hash_digest[:16], 16) / 0xFFFFFFFFFFFFFFFF
        threshold = hash_value * total_weight

        cumulative_weight = 0.0
        selected_variant = variants[0]
        for variant in variants:
            cumulative_weight += variant.weight
            if threshold <= cumulative_weight:
                selected_variant = variant
                break

        assignment = ExperimentAssignment(
            experiment_id=experiment.id,
            variant_id=selected_variant.id,
            subject_key=subject_key,
        )
        self.db_session.add(assignment)
        self.db_session.commit()
        self.db_session.refresh(assignment)
        EXPERIMENT_EVENTS.labels(event_type="assignment").inc()
        return selected_variant

    def compute_statistical_results(self, experiment: Experiment) -> dict:
        """Compute conversion statistics with confidence intervals."""
        variants = list(
            self.db_session.execute(
                select(ExperimentVariant).where(
                    ExperimentVariant.experiment_id == experiment.id
                )
            )
            .scalars()
            .all()
        )

        variant_stats: list[dict] = []
        for variant in variants:
            total_deliveries = self.db_session.execute(
                select(func.count(Delivery.id))
                .join(Appeal, Delivery.appeal_id == Appeal.id)
                .where(
                    Appeal.experiment_variant_id == variant.id,
                    Delivery.status.in_(
                        [
                            DeliveryStatus.SENT,
                            DeliveryStatus.DELIVERED,
                            DeliveryStatus.OPENED,
                            DeliveryStatus.CLICKED,
                            DeliveryStatus.CONVERTED,
                        ]
                    ),
                )
            ).scalar_one()

            converted_deliveries = self.db_session.execute(
                select(func.count(Delivery.id))
                .join(Appeal, Delivery.appeal_id == Appeal.id)
                .where(
                    Appeal.experiment_variant_id == variant.id,
                    Delivery.status == DeliveryStatus.CONVERTED,
                )
            ).scalar_one()

            conversion_rate = (
                (converted_deliveries / total_deliveries)
                if total_deliveries > 0
                else 0.0
            )
            standard_error = (
                (conversion_rate * (1.0 - conversion_rate) / total_deliveries) ** 0.5
                if total_deliveries > 0
                else 0.0
            )
            confidence_interval = (
                round(conversion_rate - 1.96 * standard_error, 4),
                round(conversion_rate + 1.96 * standard_error, 4),
            )
            variant_stats.append(
                {
                    "variant_id": variant.id,
                    "variant_name": variant.name,
                    "total_deliveries": total_deliveries,
                    "converted_deliveries": converted_deliveries,
                    "conversion_rate": round(conversion_rate, 4),
                    "confidence_interval": confidence_interval,
                }
            )

        control_stats = next(
            (stats for stats in variant_stats if stats["variant_name"] == "control"),
            None,
        )
        results: dict = {"variants": variant_stats}

        if len(variant_stats) >= 2 and control_stats is not None:
            winners: list[str] = []
            p_values: dict[str, float] = {}
            for stats in variant_stats:
                if stats["variant_id"] == control_stats["variant_id"]:
                    continue
                p_value = self._z_test_p_value(
                    control_rate=control_stats["conversion_rate"],
                    control_n=control_stats["total_deliveries"],
                    variant_rate=stats["conversion_rate"],
                    variant_n=stats["total_deliveries"],
                )
                p_values[stats["variant_name"]] = p_value
                if p_value < DEFAULT_SIGNIFICANCE_LEVEL:
                    winners.append(stats["variant_name"])
                if stats["total_deliveries"] < MIN_SAMPLE_SIZE_PER_VARIANT:
                    logger.info(
                        "Experiment variant below minimum sample size",
                        extra={
                            "variant_id": stats["variant_id"],
                            "sample_size": stats["total_deliveries"],
                        },
                    )
            results["p_values"] = p_values
            results["significant_winners"] = winners

        experiment.confidence_level = round(1.0 - DEFAULT_SIGNIFICANCE_LEVEL, 4)
        experiment.p_value = (
            min(results.get("p_values", {}).values())
            if results.get("p_values")
            else None
        )
        self.db_session.commit()
        return results

    def _z_test_p_value(
        self,
        control_rate: float,
        control_n: int,
        variant_rate: float,
        variant_n: int,
    ) -> float:
        """Compute a two-tailed z-test p-value for two proportions."""
        if control_n == 0 or variant_n == 0:
            return 1.0
        pooled_proportion = (
            (control_rate * control_n) + (variant_rate * variant_n)
        ) / (control_n + variant_n)
        standard_error = (
            pooled_proportion
            * (1.0 - pooled_proportion)
            * (1.0 / control_n + 1.0 / variant_n)
        ) ** 0.5
        if standard_error == 0.0:
            return 1.0
        z_score = (variant_rate - control_rate) / standard_error
        return round(2.0 * (1.0 - NormalDist().cdf(abs(z_score))), 4)

    def promote_winning_variant(
        self, experiment: Experiment, variant_id: str
    ) -> Experiment:
        """Promote a winning variant with an audit trail."""
        variant = self.db_session.get(ExperimentVariant, variant_id)
        if variant is None:
            raise ValueError(f"Experiment variant {variant_id} not found.")

        if variant.template_id is not None:
            template = self.db_session.get(Template, variant.template_id)
            if template is not None:
                template.is_active = True
                template.current_version_id = variant.template_id

        experiment.status = ExperimentStatus.PROMOTED
        experiment.promoted_variant_id = variant_id
        experiment.completed_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(experiment)
        EXPERIMENT_EVENTS.labels(event_type="promotion").inc()
        logger.info(
            "Experiment variant promoted",
            extra={"experiment_id": experiment.id, "variant_id": variant_id},
        )
        return experiment

    def start_experiment(self, experiment: Experiment) -> Experiment:
        """Mark an experiment as running."""
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(experiment)
        EXPERIMENT_EVENTS.labels(event_type="start").inc()
        return experiment

    def complete_experiment(self, experiment: Experiment) -> Experiment:
        """Mark an experiment as completed with statistical evaluation."""
        self.compute_statistical_results(experiment)
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        self.db_session.commit()
        self.db_session.refresh(experiment)
        EXPERIMENT_EVENTS.labels(event_type="complete").inc()
        return experiment


def get_experiment_service(db_session: Session) -> ExperimentService:
    """Return a configured experiment service instance."""
    return ExperimentService(db_session)
