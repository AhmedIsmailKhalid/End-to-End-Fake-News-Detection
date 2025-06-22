# new data/generate_fake_news.py
# generates fake news with 50% probability to introduce drift


import random
import pandas as pd
from pathlib import Path
from datetime import datetime

# Output file path
OUTPUT_PATH = Path(__file__).parent / "generated_fake.csv"
NUM_SAMPLES = 20
DRIFT_PROBABILITY = 0.5  # 50% chance to introduce drift

# Standard (non-drifted) style fake news
NORMAL_TOPICS = [
    "Government", "Politics", "Elections", "Healthcare", "Education", "Foreign Policy"
]

# Drifted (new or unusual) topics or style
DRIFT_TOPICS = [
    "Aliens", "Quantum Energy Scams", "Flat Earth Revival", "Mind Control Tech",
    "Crypto Time Machines", "Interdimensional Portals"
]

NORMAL_TEMPLATE = "BREAKING: {topic} scandal unfolds, shocking the nation."
DRIFT_TEMPLATE = "🚨 ALERT: Shocking evidence of {topic} revealed by anonymous sources!"

def generate_article(topic_list, template):
    topic = random.choice(topic_list)
    return template.format(topic=topic)

def main():
    articles = []

    for _ in range(NUM_SAMPLES):
        if random.random() < DRIFT_PROBABILITY:
            # Drifted fake news
            text = generate_article(DRIFT_TOPICS, DRIFT_TEMPLATE)
            drifted = True
        else:
            # Normal fake news
            text = generate_article(NORMAL_TOPICS, NORMAL_TEMPLATE)
            drifted = False

        articles.append({
            "text": text,
            "label": 1,  # 1 = Fake
            "drifted": drifted,
            "generated_at": datetime.utcnow().isoformat()
        })

    df = pd.DataFrame(articles)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {NUM_SAMPLES} fake articles with drift injected into {df['drifted'].sum()} of them.")
    print(f"📄 Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
