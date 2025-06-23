# app/streamlit_app.py

import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import time
import subprocess
import sys
from pathlib import Path

# Add root to sys.path for imports if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

# ---- Constants ----
# API_URL = "http://127.0.0.1:8000/predict"
API_URL = "http://localhost:8000/predict""
CUSTOM_DATA_PATH = Path(__file__).parent.parent / "data" / "custom_upload.csv"
METADATA_PATH = Path(__file__).parent.parent / "model" / "metadata.json"
ACTIVITY_LOG_PATH = Path(__file__).parent.parent / "logs" / "activity_log.json"
DRIFT_LOG_PATH = Path(__file__).parent.parent / "logs" / "monitoring_log.json"

# ---- Streamlit UI ----
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Enter a news article's headline or content to predict if it's **Fake** or **Real**.")

# ---- Prediction Form ----
with st.form(key="predict_form"):
    user_input = st.text_area("News Text", height=150)
    submit = st.form_submit_button("🧠 Predict")

if submit:
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        try:
            response = requests.post(API_URL, json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                pred = result["prediction"]
                prob = result["confidence"]
                st.success(f"🧾 Prediction: **{pred}**")
                st.info(f"📈 Confidence: {prob * 100:.2f}%")
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Failed to connect to FastAPI: {e}")

# ---- Upload + Train ----
st.header("📤 Train with Your Own CSV")

with st.expander("Upload CSV to Retrain Model (columns: `text`, `label`)"):
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df_custom = pd.read_csv(uploaded_file)
            if "text" not in df_custom.columns or "label" not in df_custom.columns:
                st.error("CSV must contain 'text' and 'label' columns.")
            else:
                st.success("✅ File looks good. Starting training...")

                # Save CSV
                df_custom.to_csv(CUSTOM_DATA_PATH, index=False)

                # Progress bar animation
                progress_bar = st.progress(0)
                status_text = st.empty()
                for percent in range(0, 101, 10):
                    progress_bar.progress(percent)
                    status_text.text(f"Training Progress: {percent}%")
                    time.sleep(0.2)

                # Trigger training subprocess
                result = subprocess.run(
                    [sys.executable, "model/train.py", "--data_path", str(CUSTOM_DATA_PATH), "--output_path", "model/custom_model.pt"],
                    capture_output=True, text=True
                )

                if result.returncode == 0:
                    acc = float(result.stdout.strip())
                    new_version = "custom_" + time.strftime("%H%M%S")
                    metadata = {
                        "model_version": new_version,
                        "test_accuracy": round(acc, 4),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                    with open(METADATA_PATH, "w") as f:
                        json.dump(metadata, f, indent=2)
                    status_text.text("🎉 Training complete!")
                    st.success(f"New model trained with accuracy: {acc:.4f}")
                else:
                    st.error("Training failed.")
                    st.text(result.stderr)
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ---- Sidebar Info ----
st.sidebar.header("📊 Model Info")
if METADATA_PATH.exists():
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    st.sidebar.markdown(f"**Version**: `{meta['model_version']}`")
    st.sidebar.markdown(f"**Accuracy**: `{meta['test_accuracy']}`")
    st.sidebar.markdown(f"**Updated**: `{meta['timestamp'].split('T')[0]}`")
else:
    st.sidebar.warning("No metadata found.")

# ---- Activity Log ----
st.sidebar.header("📜 Activity Log")
if ACTIVITY_LOG_PATH.exists():
    with open(ACTIVITY_LOG_PATH) as f:
        activity_log = json.load(f)
    for entry in reversed(activity_log[-5:]):
        st.sidebar.text(f"{entry['timestamp']} - {entry['event']}")
else:
    st.sidebar.info("No recent logs found.")

# ---- Drift Chart ----
st.sidebar.header("📉 Drift Monitoring")
if DRIFT_LOG_PATH.exists():
    drift_df = pd.read_json(DRIFT_LOG_PATH)
    drift_df["timestamp"] = pd.to_datetime(drift_df["timestamp"])
    drift_df["status"] = drift_df["drift_detected"].map({True: "Drift", False: "Stable"})

    chart = alt.Chart(drift_df).mark_line(point=True).encode(
        x="timestamp:T",
        y=alt.Y("test_accuracy:Q", title="Test Accuracy"),
        color="status:N",
        tooltip=["timestamp", "test_accuracy", "status"]
    ).properties(title="Model Performance & Drift", height=250)

    st.sidebar.altair_chart(chart, use_container_width=True)
else:
    st.sidebar.info("No drift data available.")
