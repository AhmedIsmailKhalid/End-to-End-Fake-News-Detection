import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json
import hashlib
import datetime
import shutil

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
LOGS_DIR = BASE_DIR.parent / "logs"

COMBINED = DATA_DIR / "combined_dataset.csv"
SCRAPED = DATA_DIR / "scraped_real.csv"
GENERATED = DATA_DIR / "generated_fake.csv"

PROD_MODEL = BASE_DIR / "model.pkl"
PROD_VECTORIZER = BASE_DIR / "vectorizer.pkl"

CANDIDATE_MODEL = BASE_DIR / "model_candidate.pkl"
CANDIDATE_VECTORIZER = BASE_DIR / "vectorizer_candidate.pkl"

METADATA_PATH = BASE_DIR / "metadata.json"

def hash_file(path: Path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def load_new_data():
    dfs = [pd.read_csv(COMBINED)]
    if SCRAPED.exists():
        dfs.append(pd.read_csv(SCRAPED))
    if GENERATED.exists():
        dfs.append(pd.read_csv(GENERATED))
    df = pd.concat(dfs, ignore_index=True)
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.strip() != ""]
    return df

def train_model(df):
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_vec))

    return model, vec, acc, len(X_train), len(X_test)

def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return None

def bump_version(version: str) -> str:
    major, minor = map(int, version.replace("v", "").split("."))
    return f"v{major}.{minor+1}"

def main():
    print("🔄 Retraining candidate model...")
    df = load_new_data()
    model, vec, acc, train_size, test_size = train_model(df)

    print(f"📊 Candidate Accuracy: {acc:.4f}")
    joblib.dump(model, CANDIDATE_MODEL)
    joblib.dump(vec, CANDIDATE_VECTORIZER)

    metadata = load_metadata()
    prod_acc = metadata["test_accuracy"] if metadata else 0
    model_version = bump_version(metadata["model_version"]) if metadata else "v1.0"

    if acc > prod_acc:
        print("✅ Candidate outperforms production. Promoting model...")
        shutil.copy(CANDIDATE_MODEL, PROD_MODEL)
        shutil.copy(CANDIDATE_VECTORIZER, PROD_VECTORIZER)
        metadata = {
            "model_version": model_version,
            "data_version": hash_file(COMBINED),
            "train_size": train_size,
            "test_size": test_size,
            "test_accuracy": round(acc, 4),
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"🟢 Model promoted. Version: {model_version}")
    else:
        print("🟡 Candidate did not outperform production. Keeping existing model.")

if __name__ == "__main__":
    main()