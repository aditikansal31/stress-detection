"""Prepare the Dreaddit dataset for the stress-model training pipeline.

This script intentionally keeps the data pipeline transparent and reproducible:
- load the Hugging Face Dreaddit dataset
- clean the text field
- keep only the relevant columns needed for stress classification
- save the processed dataset as Parquet for later model training and Spark ingestion
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CACHE = PROJECT_ROOT / 'data' / 'cache' / 'huggingface' / 'datasets'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    text = str(value or '').replace('\r', ' ').replace('\n', ' ')
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_frame(split_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[['text', 'label']].copy()
    subset.columns = ['text', 'label']
    subset['text'] = subset['text'].map(clean_text)
    subset = subset.dropna(subset=['text', 'label'])
    subset['label'] = subset['label'].astype(int)
    subset['split'] = split_name
    return subset.reset_index(drop=True)


def main() -> None:
    ds = load_dataset('andreagasparini/dreaddit', cache_dir=str(DATA_CACHE))

    train_df = build_frame('train', ds['train'].to_pandas())
    test_df = build_frame('test', ds['test'].to_pandas())

    train_path = PROCESSED_DIR / 'dreaddit_train.parquet'
    test_path = PROCESSED_DIR / 'dreaddit_test.parquet'
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    manifest = {
        'dataset': 'andreagasparini/dreaddit',
        'train_rows': int(len(train_df)),
        'test_rows': int(len(test_df)),
        'train_path': str(train_path),
        'test_path': str(test_path),
        'label_counts_train': train_df['label'].value_counts().sort_index().to_dict(),
        'label_counts_test': test_df['label'].value_counts().sort_index().to_dict(),
    }
    manifest_path = PROCESSED_DIR / 'dreaddit_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print(f'Train rows: {len(train_df)} -> {train_path}')
    print(f'Test rows: {len(test_df)} -> {test_path}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
