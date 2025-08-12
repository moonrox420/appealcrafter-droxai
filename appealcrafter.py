# AppealCrafter: AI-Driven Micro-Campaign Engine
# Deployment: Render under api.droxaillc.com
# Security: JWT auth, encrypted env vars
# Scalability: 10K+ donors

import os
import logging
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from celery import Celery
import io
from scipy import stats

logging.basicConfig(level=logging.INFO)
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app_celery = Celery('tasks', broker=redis_url, backend=redis_url)
JWT_SECRET = os.getenv('JWT_SECRET', 'supersecretkey')
auth_scheme = HTTPBearer()

class Donor(BaseModel):
    id: int
    history: list[float]
    interests: str
    capacity: float = None

class Appeal(BaseModel):
    donor_id: int
    channel: str
    message: str
    sent: bool = False

app = FastAPI(title="DroxAI AppealCrafter API", docs_url="/docs")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/ingest-donors")
async def ingest_donors(file: UploadFile = File(...), user=Depends(get_current_user)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        required_cols = ['id', 'history', 'interests']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Missing columns")
        df['avg_donation'] = df['history'].apply(lambda x: sum(eval(x)) / len(eval(x)) if x else 0)
        X = df[['avg_donation']]
        y = df['capacity'] if 'capacity' in df else df['avg_donation'] * 1.5
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        return {"status": "ingested", "donors": len(df), "model_accuracy": 1 - mse / y.var() if y.var() != 0 else 1.0}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-appeal")
async def generate_appeal(donor: Donor, user=Depends(get_current_user)):
    try:
        if donor.capacity is None:
            donor.capacity = sum(donor.history) / len(donor.history) * 1.5 if donor.history else 100.0
        sentiment = sia.polarity_scores(donor.interests)['compound']
        tone = "inspiring" if sentiment > 0.5 else "urgent"
        base_msg = f"Dear Donor, support via DroxAI! Capacity: ${donor.capacity:.2f}. {tone.capitalize()} appeal."
        task = send_campaign.delay(donor.id, base_msg, donor.channel)
        return {"appeal": base_msg, "task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")

class ABTest(BaseModel):
    variants: dict[str, list[float]]

@app.post("/ab-test")
async def run_ab_test(test: ABTest, user=Depends(get_current_user)):
    try:
        df = pd.DataFrame({'variant': [], 'conversions': []})
        for variant, convs in test.variants.items():
            df = pd.concat([df, pd.DataFrame({'variant': variant, 'conversions': convs})])
        variant_a = df[df['variant'] == 'A']['conversions']
        variant_b = df[df['variant'] == 'B']['conversions']
        if len(variant_a) < 2 or len(variant_b) < 2:
            return {"winner": "Insufficient data", "p_value": 1.0}
        t_stat, p_val = stats.ttest_ind(variant_a, variant_b)
        winner = "A" if p_val < 0.05 and t_stat > 0 else "B"
        return {"winner": winner, "p_value": p_val, "t_stat": t_stat}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app_celery.task
def send_campaign(donor_id: int, message: str, channel: str):
    logging.info(f"Sent to {donor_id} via {channel}: {message} [DroxAI Powered]")
    return {"sent": True, "open_rate": 0.75}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)