import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import jensenshannon
import joblib
from datetime import datetime

# Paths
SCRAPED_PATH = Path("data/scraped_real.csv")
TRAIN_PATH = Path("data/combined_dataset.csv")
VECTORIZER_PATH = Path("model/vectorizer.pkl")
LOG_PATH = Path("logs/monitoring_log.json")

def compute_js_divergence(vec1, vec2):
    # Add smoothing to avoid 0s
    vec1 += 1e-9
    vec2 += 1e-9
    return jensenshannon(vec1, vec2)

def monitor_drift():
    if not SCRAPED_PATH.exists() or not TRAIN_PATH.exists():
        return

    df_scraped = pd.read_csv(SCRAPED_PATH)
    df_train = pd.read_csv(TRAIN_PATH)

    # Sample same size for fair comparison
    scraped_texts = df_scraped['text'].dropna().sample(n=100, replace=True, random_state=42)
    train_texts = df_train['text'].dropna().sample(n=100, replace=True, random_state=42)

    vectorizer = joblib.load(VECTORIZER_PATH)

    scraped_vec = vectorizer.transform(scraped_texts).mean(axis=0).A1
    train_vec = vectorizer.transform(train_texts).mean(axis=0).A1

    drift_score = compute_js_divergence(train_vec, scraped_vec)

    # Log
    record = {
        "timestamp": datetime.now().isoformat(),
        "drift_score": round(float(drift_score), 5)
    }

    if LOG_PATH.exists():
        logs = json.loads(LOG_PATH.read_text())
    else:
        logs = []

    logs.append(record)
    LOG_PATH.write_text(json.dumps(logs, indent=2))

    print(f"✅ Drift score computed: {drift_score:.5f}")
    return drift_score
