"""
Build Yelp dataset.

Raw data should be placed at:
    data/yelp/yelp_academic_dataset_business.json
    data/yelp/yelp_academic_dataset_review.json

Download from https://www.yelp.com/dataset :
  yelp_dataset.tar → extract the two JSON files into data/yelp/

Run from the project root:
    python build_dataset/build_dataset_yelp.py

Output:
    datasets/Yelp/train_reviews_32k.csv
    datasets/Yelp/valid_reviews_32k.csv
    datasets/Yelp/user_dict.pickle
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from pathlib import Path
from utils.datasets import EntityDictionary

SEED = 42

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DIR    = Path("data/yelp")
OUTPUT_DIR = Path("datasets/Yelp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, 'r') as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return pd.DataFrame(rows)


def build_description(row) -> str:
    cats = ', '.join(c.strip() for c in str(row['categories']).split(',')[:10])
    desc = (f"{row['name']}, which is a business with {row['stars']} stars rating "
            f"based on {row['review_count']} reviews.")
    if cats:
        desc += f" It is categorized as {cats}."
    return desc


def train_valid_split(df: pd.DataFrame, val_ratio: float = 0.2):
    """Split each user's reviews chronologically (last val_ratio → valid)."""
    train_parts, valid_parts = [], []
    for _, user_df in df.groupby('UserID'):
        user_df = user_df.sort_values('date')
        n_valid = max(1, int(len(user_df) * val_ratio))
        train_parts.append(user_df.iloc[:-n_valid])
        valid_parts.append(user_df.iloc[-n_valid:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(valid_parts).reset_index(drop=True)


# ── Load raw data ────────────────────────────────────────────────────────────
print("Loading business data...")
business = load_jsonl(RAW_DIR / "yelp_academic_dataset_business.json")
print("Loading review data...")
review = load_jsonl(RAW_DIR / "yelp_academic_dataset_review.json")
print(f"  businesses: {len(business):,}  |  reviews: {len(review):,}")

# ── Clean businesses ─────────────────────────────────────────────────────────
# Keep entries with non-empty name, positive review_count, non-empty categories
mask = (
    business['name'].notna() & (business['name'].str.strip() != '') &
    business['review_count'].notna() & (business['review_count'] > 0) &
    business['categories'].notna() & (business['categories'].str.strip() != '')
)
business_clean = business[mask].reset_index(drop=True)
print(f"After business cleaning: {len(business_clean):,} businesses")

# ── Build description map ────────────────────────────────────────────────────
business_info = {
    row['business_id']: {
        'name': row['name'],
        'description': build_description(row),
    }
    for _, row in business_clean.iterrows()
}

# ── Filter reviews ───────────────────────────────────────────────────────────
valid_bids = set(business_info)
reviews_filtered = review[review['business_id'].isin(valid_bids)].copy()

# Keep users with >= 25 reviews
user_counts = reviews_filtered['user_id'].value_counts()
active_users = user_counts[user_counts >= 25].index
reviews_filtered = reviews_filtered[reviews_filtered['user_id'].isin(active_users)]
print(f"After activity filtering: {len(reviews_filtered):,} reviews, "
      f"{reviews_filtered['user_id'].nunique():,} users")

# ── Cap each user at 75 most-recent reviews ──────────────────────────────────
reviews_filtered['date'] = pd.to_datetime(reviews_filtered['date'])
df = (
    reviews_filtered
    .sort_values(['user_id', 'date'], ascending=[True, False])
    .groupby('user_id', as_index=False)
    .head(75)
    .sort_values(['user_id', 'date'], ascending=[True, True])
    .reset_index(drop=True)
)

# ── Build combined dataset with business descriptions ────────────────────────
records = []
for _, row in df.iterrows():
    bid = row['business_id']
    if bid in business_info:
        records.append({
            'UserID':               row['user_id'],
            'business_id':          bid,
            'business_description': business_info[bid]['description'],
            'business_name':       business_info[bid]['name'],
            'rating':               row['stars'],
            'date':                 row['date'],
        })
dataset = pd.DataFrame(records)
print(f"Combined dataset: {len(dataset):,} entries, {dataset['UserID'].nunique():,} users")

# ── Train / validation split ─────────────────────────────────────────────────
train_reviews, valid_reviews = train_valid_split(dataset)
print(f"Train: {len(train_reviews):,}  |  Valid: {len(valid_reviews):,}")

# ── Build and save user dictionary ──────────────────────────────────────────
np.random.seed(SEED)
user_dict = EntityDictionary()
for uid in np.random.permutation(dataset['UserID'].unique()):
    user_dict.add_entity(uid)

user_dict_path = OUTPUT_DIR / "user_dict.pickle"
user_dict.save(str(user_dict_path))
print(f"User dict saved → {user_dict_path}  ({len(user_dict):,} users)")

# ── Shuffle and save ─────────────────────────────────────────────────────────
train_out = train_reviews.sample(frac=1, random_state=SEED).reset_index(drop=True)
valid_out  = valid_reviews.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_path = OUTPUT_DIR / "train_reviews_32k.csv"
valid_path = OUTPUT_DIR / "valid_reviews_32k.csv"
train_out.to_csv(train_path, sep='\t', index=False)
valid_out.to_csv(valid_path, sep='\t', index=False)
print(f"Saved  {train_path}  ({len(train_out):,} rows)")
print(f"Saved  {valid_path}  ({len(valid_out):,} rows)")
