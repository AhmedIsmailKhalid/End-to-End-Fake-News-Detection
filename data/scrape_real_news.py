from newspaper import Article, build
from pathlib import Path
import pandas as pd
import datetime

# Output path
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "scraped_real.csv"

# News sources
NEWS_SITES = [
    ("Reuters", "https://www.reuters.com/"),
    ("BBC", "https://www.bbc.com/news"),
    ("NPR", "https://www.npr.org/")
]

MAX_ARTICLES = 15

def scrape_articles():
    all_articles = []
    total_scraped = 0

    for source_name, url in NEWS_SITES:
        print(f"📡 Scraping from {source_name}...")
        paper = build(url, memoize_articles=False)

        for article in paper.articles:
            if total_scraped >= MAX_ARTICLES:
                break

            try:
                article.download()
                article.parse()

                if len(article.text.strip()) < 100:
                    continue  # Skip very short ones

                text = article.title + ". " + article.text
                all_articles.append({
                    "text": text,
                    "label": 0,
                    "source": source_name,
                    "timestamp": datetime.datetime.now().isoformat()
                })

                total_scraped += 1

            except Exception:
                continue  # Skip failed downloads

        if total_scraped >= MAX_ARTICLES:
            break  # Stop scraping once target reached

    if all_articles:
        df = pd.DataFrame(all_articles)

        if OUTPUT_PATH.exists():
            df_existing = pd.read_csv(OUTPUT_PATH)
            df = pd.concat([df_existing, df], ignore_index=True)

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Scraped and saved {len(all_articles)} new articles.")
    else:
        print("⚠️ No articles scraped.")

if __name__ == "__main__":
    scrape_articles()
