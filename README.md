---
title: Fake News Detection MLOs Web App
emoji: 📈
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
short_description: An end to end web app that allows to check for fake news
license: mit
---
# 📰 Fake News Detector

A full-stack AI-powered fake news detection system built using Python, FastAPI, Streamlit, and traditional ML. This project demonstrates end-to-end skills in AI, machine learning, MLOps, and cloud deployment. It includes real-time inference, automatic retraining based on new data, drift detection, live monitoring, and user interactivity—all hosted for free on HuggingFace Spaces.

🔗 Live App: https://huggingface.co/spaces/Ahmedik95316/Fake-News-Detection-MLOs-Web-App

---

## 🚀 Overview: What This Project Demonstrates

This system showcases:

* ✅ Real-world problem framing and supervised learning pipeline
* ✅ Logistic Regression + TF-IDF for binary text classification (Fake vs Real news)
* ✅ FastAPI as a backend service for live model inference
* ✅ Streamlit frontend with confidence feedback and visualization
* ✅ Hourly scraping of real articles from news websites
* ✅ Generation of fake news headlines using prompt-style templates
* ✅ Automated retraining when new data is added
* ✅ Promotion strategy for candidate model vs production
* ✅ Jensen-Shannon divergence for detecting data drift
* ✅ JSON-based metadata and model versioning
* ✅ Activity and monitoring logs for training, drift, and promotion
* ✅ Custom CSV upload and live training inside the UI
* ✅ Entirely deployed and hosted on Render.com — no setup required for end users

This project proves your ability to bridge machine learning, DevOps, and user experience design—all critical MLOps competencies.

---

## 🧠 What the System Does

This project automatically:

1. Scrapes real news articles every hour from **Reuters**, **BBC**, and **NPR**
2. Generates fake news headlines using programmatic templates
3. Appends new data to the dataset
4. Triggers retraining if data is added
5. Compares model accuracy to existing model
6. Promotes candidate model if it performs better
7. Logs drift scores and training events
8. Allows users to manually upload datasets and monitor training
9. Predicts Fake or Real for any given text through the Streamlit interface

---

## 📂 Directory Breakdown

### `/app/`

* `fastapi_server.py` – FastAPI backend serving the `/predict` endpoint. Used by Streamlit to perform live model inference.
* `streamlit_app.py` – The main UI for users. Handles input, prediction, training visualization, metadata, drift monitoring, and upload support.

### `/data/`

* `prepare_datasets.py` – Merges Kaggle and LIAR datasets, standardizes formats, and outputs a unified training file.
* `scrape_real_news.py` – Scrapes the latest articles from Reuters, BBC, and NPR using newspaper3k.
* `generate_fake_news.py` – Uses predefined templates to generate believable fake headlines and articles.
* `combined_dataset.csv` – The master dataset combining real and fake news.
* `scraped_real.csv` – Output from the scraper.
* `generated_fake.csv` – Output from the fake news generator.

### `/model/`

* `train.py` – Trains the ML model (Logistic Regression + TF-IDF) on the dataset provided.
* `retrain.py` – Trains a candidate model on newly added data, compares it to the production model, and promotes it if accuracy improves.
* `model.pkl` – Current production model.
* `vectorizer.pkl` – TF-IDF encoder used with the production model.
* `model_candidate.pkl` – Temporarily trained candidate model.
* `vectorizer_candidate.pkl` – TF-IDF encoder for candidate model.
* `metadata.json` – Tracks model version, training accuracy, and timestamp of last promotion.

### `/monitor/`

* `monitor_drift.py` – Calculates Jensen-Shannon divergence between real-time data and training data to identify distributional shift.

### `/scheduler/`

* `schedule_tasks.py` – Central scheduler that triggers scraping, fake generation, retraining, drift monitoring, and logging every hour.

### `/logs/`

* `activity_log.json` – Timestamped logs of scraping, generation, and retraining activities.
* `monitoring_log.json` – Drift scores and evaluation logs of candidate vs production model.

### Root Files

* `requirements.txt` – All project dependencies
* `render.yaml` – Configuration file used to deploy both backend and frontend on Render.com
* `README.md` – The file you're reading

---

## 💡 Automation Logic

Every hour (or every minute in test mode), the system:

1. Scrapes 15 new real news articles
2. Generates 20 new fake news articles
3. Appends them to the existing dataset
4. Triggers retraining of a candidate model
5. Compares it to the existing production model
6. If better, promotes the candidate to production
7. Logs drift score using Jensen-Shannon divergence
8. Updates visual logs and accuracy charts in the UI

---

🌐 Live Deployment: Hugging Face Spaces
This project is fully deployed on Hugging Face Spaces using a Dockerized setup that includes both the Streamlit UI and FastAPI backend in a single container.

🔗 Launch the App on Hugging Face Spaces
The app runs entirely within a Hugging Face-hosted Docker container.

Both the FastAPI inference server and Streamlit web interface are packaged together, ensuring fast internal communication.

The API_URL in streamlit_app.py is set to http://localhost:8000/predict to support intra-container requests.

The container uses Python 3.11.6, aligned with the local development environment for consistency and reproducibility.

⚙️ Deployment Infrastructure
The deployment uses a custom Dockerfile tailored to match the exact development environment.

All dependencies are pinned to specific versions in the requirements.txt file to avoid incompatibilities.

The container runs both services concurrently using a process supervisor (if needed), ensuring a single deployment handles the complete user workflow.

---

## 🎯 Skills Demonstrated

* AI/ML: Logistic Regression, TF-IDF, binary text classification
* MLOps: scheduled retraining, model promotion, version tracking
* Drift Detection: Jensen-Shannon divergence implementation
* Cloud DevOps: deploying two services via `render.yaml`
* UI/UX: live model prediction, upload, progress bars, logging
* Data Engineering: merging datasets, web scraping, labeling

---

## 🧠 Credits

* [LIAR Dataset (Politifact)](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
* [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
* [newspaper3k](https://github.com/codelucas/newspaper)
* FastAPI, Streamlit, scikit-learn, Render

---

## 📜 License

MIT