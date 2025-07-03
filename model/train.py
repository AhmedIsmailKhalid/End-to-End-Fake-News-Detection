import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import json
import datetime
import hashlib

# # Paths
# BASE_DIR = Path(__file__).resolve().parent
# DATA_PATH = BASE_DIR.parent / "data" / "combined_dataset.csv"
# MODEL_PATH = BASE_DIR / "model.pkl"
# VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
# METADATA_PATH = BASE_DIR / "metadata.json"

# Base dir and data location inside /tmp
BASE_DIR = Path("/tmp")
DATA_PATH = BASE_DIR / "data" / "combined_dataset.csv"

# Model artifacts also in /tmp (or you can keep these in /app/model if you want to persist them in the container)
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
# METADATA_PATH = BASE_DIR / "metadata.json"
METADATA_PATH = Path("/tmp/metadata.json")


def hash_file(filepath):
    content = Path(filepath).read_bytes()
    return hashlib.md5(content).hexdigest()

def main():
    # Load dataset
    # print('Dataset Loaded')
    df = pd.read_csv(DATA_PATH)
    X = df['text']
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # print('Train/Test Splits Created')
    # print('Starting Model Training')

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # print('Model Training Completed')
     #print('Model Evaluation Starting!')

    # Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    # Save model + vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # print('Model Evaluation Done')
    # print('Model Saved!')

    # Save metadata
    metadata = {
        "model_version": f"v1.0",
        "data_version": hash_file(DATA_PATH),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_accuracy": round(acc, 4),
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Model trained and saved.")
    print(f"📊 Test Accuracy: {acc:.4f}")
    print(f"📝 Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    main()
