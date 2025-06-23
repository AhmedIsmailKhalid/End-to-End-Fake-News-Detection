import pandas as pd
import random
from pathlib import Path
import datetime

# Save location
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "generated_fake.csv"

# Simple templates (can later be replaced with GPT calls)
SEED_TITLES = [
    "BREAKING: {person} spotted meeting with {group} in {location}",
    "SHOCKING: {event} blamed on secret {conspiracy}",
    "{celebrity} caught using {product} to communicate with aliens",
    "Scientists confirm link between {topic1} and {topic2}",
    "You won’t believe what happened when {person} tried to {action}"
]

PERSONS = ["Elon Musk", "Taylor Swift", "Joe Biden", "Mark Zuckerberg"]
GROUPS = ["the Illuminati", "CIA operatives", "Area 51 agents"]
LOCATIONS = ["Nevada desert", "secret DC facility", "Mars base"]
EVENTS = ["solar eclipse", "stock market crash", "bird migration"]
CONSPIRACIES = ["government cover-up", "climate manipulation", "AI mind control"]
CELEBRITIES = ["Kanye West", "Oprah", "Tom Hanks"]
PRODUCTS = ["microwave ovens", "WiFi routers", "Apple Watches"]
TOPICS = ["flat earth", "5G radiation", "cryptocurrency"]
ACTIONS = ["hack the system", "uncover the truth", "expose the elite"]

def generate_one():
    template = random.choice(SEED_TITLES)
    return template.format(
        person=random.choice(PERSONS),
        group=random.choice(GROUPS),
        location=random.choice(LOCATIONS),
        event=random.choice(EVENTS),
        conspiracy=random.choice(CONSPIRACIES),
        celebrity=random.choice(CELEBRITIES),
        product=random.choice(PRODUCTS),
        topic1=random.choice(TOPICS),
        topic2=random.choice(TOPICS),
        action=random.choice(ACTIONS)
    )

def generate_fake_news(n=20):
    rows = []
    for _ in range(n):
        text = generate_one()
        rows.append({
            "text": text,
            "label": 1,
            "source": "synthetic_gpt",
            "timestamp": datetime.datetime.now().isoformat()
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {n} fake articles and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_fake_news(20)
