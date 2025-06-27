import schedule
import time
from pathlib import Path
from data.scrape_real_news import scrape_articles
from data.generate_fake_news import generate_fake_news
from monitor.monitor_drift import monitor_drift
from model.retrain import train_model
import json
from datetime import datetime

LOG_PATH = Path("logs/activity_log.json")

def log_event(event: str):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "event": event
    }
    if LOG_PATH.exists():
        logs = json.loads(LOG_PATH.read_text())
    else:
        logs = []

    logs.append(log_entry)
    LOG_PATH.write_text(json.dumps(logs, indent=2))

def run_scraper_and_generator():
    print("⏳ Running scraping and generation tasks...")
    scrape_real_articles()
    generate_fake_articles()
    log_event("New data scraped and uploaded, triggering retraining now")

    print("🔁 Retraining pipeline started...")
    retrain_if_needed()

    print("🔎 Monitoring for data drift...")
    drift_score = monitor_drift()
    log_event(f"Drift Score: {drift_score:.5f}")

    print("✅ All tasks completed and logged.\n")

# Initial run
run_scraper_and_generator()

# Schedule hourly
schedule.every().hour.do(run_scraper_and_generator)

print("📅 Scheduler started. Running tasks every hour.\n")

while True:
    schedule.run_pending()
    time.sleep(60)
