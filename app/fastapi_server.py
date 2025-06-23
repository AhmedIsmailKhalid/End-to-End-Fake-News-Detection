from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "model" / "model.pkl"
VECTORIZER_PATH = BASE_DIR.parent / "model" / "vectorizer.pkl"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# FastAPI app
app = FastAPI(
    title="Fake News Detector API",
    version="1.0",
    description="API for predicting whether a news article is fake or real"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    X = vectorizer.transform([request.text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][pred]

    return PredictResponse(
        prediction="Fake" if pred == 1 else "Real",
        confidence=round(proba, 4)
    )
