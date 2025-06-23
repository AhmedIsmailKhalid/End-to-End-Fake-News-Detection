import pandas as pd
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Kaggle dataset paths
KAGGLE_FAKE = BASE_DIR / "kaggle" / "Fake.csv"
KAGGLE_REAL = BASE_DIR / "kaggle" / "True.csv"

# LIAR dataset paths
LIAR_PATHS = [
    BASE_DIR / "liar" / "train.tsv",
    BASE_DIR / "liar" / "test.tsv",
    BASE_DIR / "liar" / "valid.tsv"
]

# Output path
OUTPUT_PATH = BASE_DIR / "combined_dataset.csv"

def load_kaggle():
    df_fake = pd.read_csv(KAGGLE_FAKE)
    df_real = pd.read_csv(KAGGLE_REAL)

    df_fake['label'] = 1
    df_real['label'] = 0

    df_fake['text'] = df_fake['title'].fillna('') + ". " + df_fake['text'].fillna('')
    df_real['text'] = df_real['title'].fillna('') + ". " + df_real['text'].fillna('')

    return pd.concat([df_fake[['text', 'label']], df_real[['text', 'label']]], ignore_index=True)

def load_liar():
    liar_dfs = []
    for path in LIAR_PATHS:
        df = pd.read_csv(path, sep='\t', header=None, quoting=3, on_bad_lines='skip')
        df.columns = ['id', 'label_text', 'statement', 'subject', 'speaker', 'job', 
                      'state', 'party', 'barely_true', 'false', 'half_true', 'mostly_true', 
                      'pants_on_fire', 'context']
        df['label'] = df['label_text'].apply(lambda x: 1 if x in ['false', 'pants-fire'] else 0)
        df['text'] = df['statement']
        liar_dfs.append(df[['text', 'label']])
    return pd.concat(liar_dfs, ignore_index=True)

def main():
    print("Loading Kaggle dataset...")
    df_kaggle = load_kaggle()

    print("Loading LIAR dataset...")
    df_liar = load_liar()

    print("Combining both datasets...")
    full_df = pd.concat([df_kaggle, df_liar], ignore_index=True)

    full_df.dropna(subset=['text'], inplace=True)
    full_df = full_df[full_df['text'].str.strip() != ""]

    print(f"Final dataset size: {len(full_df)} samples")
    full_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Combined dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
