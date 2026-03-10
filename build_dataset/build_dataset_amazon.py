"""
Build Amazon Movies & TV dataset.

Raw data should be placed at:
    data/amazon_mt_2023/Movies_and_TV.jsonl.gz
    data/amazon_mt_2023/meta_Movies_and_TV.jsonl.gz

Download from https://amazon-reviews-2023.github.io/ :
  - 5-core interactions : Movies_and_TV.jsonl.gz
  - Item metadata       : meta_Movies_and_TV.jsonl.gz

Run from the project root:
    python build_dataset/build_dataset_amazon.py

Output:
    datasets/MoviesAndTV/train_data_30k.csv
    datasets/MoviesAndTV/valid_data_30k.csv
    datasets/MoviesAndTV/user_dict_30k.pickle
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.datasets import EntityDictionary

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/amazon_mt_2023")
OUTPUT_DIR = Path("datasets/MoviesAndTV")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = RAW_DIR / "Movies_and_TV.jsonl.gz"
META_PATH = RAW_DIR / "meta_Movies_and_TV.jsonl.gz"


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_jsonl_gz(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def train_valid_split(df: pd.DataFrame, val_ratio: float = 0.2):
    """Split each user's interactions chronologically (last val_ratio → valid)."""
    train_parts, valid_parts = [], []
    for _, user_df in df.groupby('user'):
        user_df = user_df.sort_values('timestamp')
        split = int(len(user_df) * (1 - val_ratio))
        train_parts.append(user_df.iloc[:split])
        valid_parts.append(user_df.iloc[split:])
    return pd.concat(train_parts), pd.concat(valid_parts)


# ── Load raw data ────────────────────────────────────────────────────────────
print("Loading interaction data...")
df_main = load_jsonl_gz(DATA_PATH)
print("Loading metadata...")
df_meta = load_jsonl_gz(META_PATH)
print(f"  interactions: {len(df_main):,}  |  metadata: {len(df_meta):,}")

# ── Attach item titles ───────────────────────────────────────────────────────
title_map = df_meta.set_index('parent_asin')['title'].to_dict()

df = df_main[['user_id', 'parent_asin', 'rating', 'timestamp']].copy()
df['ItemTitle'] = df['parent_asin'].map(title_map)
df = df.dropna(subset=['ItemTitle'])
df = df[df['ItemTitle'].str.strip() != '']
print(f"After attaching titles: {len(df):,} interactions")

# ── Filter: items with < 10 interactions ────────────────────────────────────
item_counts = df['parent_asin'].value_counts()
df = df[df['parent_asin'].isin(item_counts[item_counts >= 10].index)]

# ── Filter: users with < 20 interactions ────────────────────────────────────
user_counts = df['user_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 20].index)]
print(f"After activity filtering: {len(df):,} interactions, {df['user_id'].nunique():,} users")

# ── Cap each user at 85 most-recent interactions ─────────────────────────────
df = (
    df.sort_values(['user_id', 'timestamp'], ascending=[True, False])
      .groupby('user_id', as_index=False)
      .head(85)
      .reset_index(drop=True)
)
df = df.rename(columns={'user_id': 'user', 'parent_asin': 'ItemID'})

# ── Train / validation split ─────────────────────────────────────────────────
train_df, valid_df = train_valid_split(df)
print(f"Train: {len(train_df):,}  |  Valid: {len(valid_df):,}")

# ── Shuffle ──────────────────────────────────────────────────────────────────
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Build and save user dictionary ──────────────────────────────────────────
np.random.seed(42)
user_dict = EntityDictionary()
for uid in np.random.permutation(df['user'].unique()):
    user_dict.add_entity(uid)

user_dict_path = OUTPUT_DIR / "user_dict_30k.pickle"
user_dict.save(str(user_dict_path))
print(f"User dict saved → {user_dict_path}  ({len(user_dict):,} users)")

# ── Rename and save ──────────────────────────────────────────────────────────
train_df = train_df.rename(columns={'user': 'UserID'})
valid_df = valid_df.rename(columns={'user': 'UserID'})

train_path = OUTPUT_DIR / "train_data_30k.csv"
valid_path = OUTPUT_DIR / "valid_data_30k.csv"
train_df.to_csv(train_path, sep='\t', index=False)
valid_df.to_csv(valid_path, sep='\t', index=False)
print(f"Saved  {train_path}  ({len(train_df):,} rows)")
print(f"Saved  {valid_path}  ({len(valid_df):,} rows)")