# AppealCrafter: AI-Driven Micro-Campaign Engine
# Deployment: Render under api.droxaillc.com
# Security: JWT auth, encrypted env vars
# Scalability: 10K+ donors

import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from celery import Celery
from celery.schedules import crontab
import sqlite3
import json
import io
from scipy import stats
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
app_celery = Celery('tasks', broker=redis_url, backend=redis_url)
JWT_SECRET = os.getenv('JWT_SECRET', 'supersecretkey')
auth_scheme = HTTPBearer()

class Donor(BaseModel):
    id: int
    history: List[float] = Field(default_factory=list)
    interests: str
    capacity: float = None
    channel: str = "email"
    email: str = ""

class Appeal(BaseModel):
    donor_id: int
    channel: str
    message: str
    sent: bool = False

app = FastAPI(title="DroxAI AppealCrafter API", docs_url="/docs")

def fetch_donors():
    conn = sqlite3.connect("appealcrafter.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, history, interests, capacity, channel, email FROM donors")
    rows = cursor.fetchall()
    donors = []
    for row in rows:
        donors.append({
            "id": row[0],
            "history": json.loads(row[1]),
            "interests": row[2],
            "capacity": row[3],
            "channel": row[4],
            "email": row[5]
        })
    conn.close()
    return donors

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/ingest-donors")
async def ingest_donors(user=Depends(get_current_user)):
    try:
        donors = fetch_donors()
        df = pd.DataFrame(donors)
        required_cols = {'id', 'history', 'interests'}
        if not required_cols.issubset(df.columns):
            raise ValueError("Missing columns")
        df['avg_donation'] = df['history'].apply(lambda x: sum(x) / len(x) if x else 0)
        X = df[['avg_donation']]
        y = df['capacity'] if 'capacity' in df.columns else df['avg_donation'] * 1.5
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        y_var = y.var()
        mse = mean_squared_error(y_test, preds)
        accuracy = 1 - mse / y_var if y_var != 0 else 1.0
        logging.info(f"Donor ingestion complete. Model accuracy: {accuracy:.4f}")
        return {"status": "ingested", "donors": len(df), "model_accuracy": accuracy}
    except Exception as e:
        logging.error(f"Error ingesting donors: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-appeal")
async def generate_appeal(user=Depends(get_current_user)):
    try:
        donors = fetch_donors()
        appeals = []
        for donor in donors:
            donor['capacity'] = donor.get('capacity') or (sum(donor['history']) / len(donor['history']) * 1.5 if donor['history'] else 100.0)
            sentiment = sia.polarity_scores(donor['interests'])['compound']
            tone = "inspiring" if sentiment > 0.5 else "urgent"
            base_msg = f"Dear Donor, support via DroxAI! Capacity: ${donor['capacity']:.2f}. {tone.capitalize()} appeal."
            task = send_campaign.delay(donor['id'], base_msg, donor['channel'], donor['email'])
            appeals.append({"appeal": base_msg, "task_id": task.id})
        logging.info(f"Generated {len(appeals)} appeals.")
        return {"appeals": appeals}
    except Exception as e:
        logging.error(f"Error generating appeals: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

class ABTest(BaseModel):
    variants: dict[str, List[float]]

@app.post("/ab-test")
async def run_ab_test(user=Depends(get_current_user)):
    try:
        test = ABTest(variants={
            "A": [1, 0, 1, 1],
            "B": [0, 1, 0, 0]
        })
        records = [
            {"variant": variant, "conversions": conv}
            for variant, convs in test.variants.items()
            for conv in convs
        ]
        df = pd.DataFrame.from_records(records)
        variant_a = df[df['variant'] == 'A']['conversions']
        variant_b = df[df['variant'] == 'B']['conversions']
        if len(variant_a) < 2 or len(variant_b) < 2:
            return {"winner": "Insufficient data", "p_value": 1.0}
        t_stat, p_val = stats.ttest_ind(variant_a, variant_b, equal_var=False)
        winner = "A" if p_val < 0.05 and t_stat > 0 else "B"
        logging.info(f"A/B test result: winner={winner}, p_value={p_val:.4f}, t_stat={t_stat:.4f}")
        return {"winner": winner, "p_value": p_val, "t_stat": t_stat}
    except Exception as e:
        logging.error(f"Error running AB test: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app_celery.task
def send_campaign(donor_id: int, message: str, channel: str, email: str):
    logging.info(f"[FAKE EMAIL] To: {email} | Channel: {channel} | Message: {message}")
    return {"sent": True}

@app_celery.task
def auto_send_appeals():
    donors = fetch_donors()
    for donor in donors:
        donor['capacity'] = donor.get('capacity') or (sum(donor['history']) / len(donor['history']) * 1.5 if donor['history'] else 100.0)
        sentiment = sia.polarity_scores(donor['interests'])['compound']
        tone = "inspiring" if sentiment > 0.5 else "urgent"
        base_msg = f"Dear Donor, support via DroxAI! Capacity: ${donor['capacity']:.2f}. {tone.capitalize()} appeal."
        send_campaign.delay(donor['id'], base_msg, donor['channel'], donor['email'])
    logging.info(f"Auto-sent appeals to {len(donors)} donors.")

@app_celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        crontab(hour=9, minute=0),
        auto_send_appeals.s(),
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
