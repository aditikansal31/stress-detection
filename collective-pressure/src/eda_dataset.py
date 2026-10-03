"""Generate a project-level EDA summary for all available dataset sources.

This module reads the local datasets in ../datasets and project-level GoEmotions files,
then writes a machine-readable JSON and a markdown summary that the later phases can use
for dataset-role decisions and preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BDA_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = BDA_ROOT / 'datasets'
REPORT_DIR = PROJECT_ROOT / 'reports'
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def summarize_dataframe(df: pd.DataFrame, source: str, table: str) -> dict[str, Any]:
    summary = {
        'source': source,
        'table': table,
        'rows': int(len(df)),
        'columns': list(df.columns),
        'column_count': int(df.shape[1]),
        'missing_values': {str(col): int(df[col].isna().sum()) for col in df.columns},
        'duplicate_rows': int(df.duplicated().sum()),
    }

    if 'text' in df.columns:
        text_col = df['text'].fillna('').astype(str)
        text_lengths = text_col.str.len()
        summary['text_length_summary'] = {
            'min': int(text_lengths.min()) if len(text_lengths) else 0,
            'p25': safe_float(text_lengths.quantile(0.25)),
            'median': safe_float(text_lengths.median()),
            'p75': safe_float(text_lengths.quantile(0.75)),
            'p90': safe_float(text_lengths.quantile(0.90)),
            'max': int(text_lengths.max()) if len(text_lengths) else 0,
            'mean': safe_float(text_lengths.mean()),
        }

    for col in df.columns:
        series = df[col]
        if series.nunique(dropna=True) <= 20:
            summary[f'category_counts:{col}'] = series.value_counts(dropna=False).head(20).astype(int).to_dict()

    if any(col in df.columns for col in ['label', 'stress_type']):
        label_col = 'label' if 'label' in df.columns else 'stress_type'
        summary['label_distribution'] = df[label_col].value_counts(dropna=False).astype(int).to_dict()

    return summary


def audit_mendeley() -> list[dict[str, Any]]:
    path = DATASET_ROOT / 'Stress Indicators Dataset for Mental Health Classi' / 'Stress Indicators Dataset for Mental Health Classi' / 'Stress Indicators Dataset for Mental Health Classification.csv'
    df = pd.read_csv(path)
    return [summarize_dataframe(df, str(path.relative_to(BDA_ROOT)), 'StressIndicators_Mendeley')]


def audit_goemotions() -> list[dict[str, Any]]:
    files = [
        DATASET_ROOT / 'archive' / 'data' / 'train.tsv',
        DATASET_ROOT / 'archive' / 'data' / 'test.tsv',
        DATASET_ROOT / 'archive' / 'data' / 'dev.tsv',
        DATASET_ROOT / 'archive' / 'data' / 'full_dataset' / 'goemotions_1.csv',
        DATASET_ROOT / 'archive' / 'data' / 'full_dataset' / 'goemotions_2.csv',
        DATASET_ROOT / 'archive' / 'data' / 'full_dataset' / 'goemotions_3.csv',
    ]
    summaries = []
    for path in files:
        if path.suffix == '.tsv':
            df = pd.read_csv(path, sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
        else:
            df = pd.read_csv(path)
        summaries.append(summarize_dataframe(df, str(path.relative_to(BDA_ROOT)), path.name))
    return summaries


def audit_senticnet() -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((DATASET_ROOT / 'stress-detection-main' / 'stress-detection-main').glob('*.xlsx')):
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name)
            summaries.append(summarize_dataframe(df, str(path.relative_to(BDA_ROOT)), f'{path.stem}:{sheet_name}'))
    return summaries


def audit_dreaddit() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset('andreagasparini/dreaddit', cache_dir=str(PROJECT_ROOT / 'data' / 'cache' / 'huggingface' / 'datasets'))
    summaries = []
    for split_name, split_dataset in ds.items():
        df = split_dataset.to_pandas()
        summaries.append(summarize_dataframe(df, 'HuggingFace:andreagasparini/dreaddit', split_name))
    return summaries


def main() -> None:
    summaries = []
    summaries.extend(audit_mendeley())
    summaries.extend(audit_goemotions())
    summaries.extend(audit_senticnet())
    summaries.extend(audit_dreaddit())

    json_path = REPORT_DIR / 'eda_summary.json'
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = ['# Dataset EDA Summary', '']
    for summary in summaries:
        lines.append(f"## {summary['source']} / {summary['table']}")
        lines.append(f"- Rows: {summary['rows']}")
        lines.append(f"- Columns: {summary['column_count']}")
        lines.append(f"- Duplicate rows: {summary['duplicate_rows']}")
        if 'label_distribution' in summary:
            lines.append('- Label distribution: ' + ', '.join(f'{k}={v}' for k, v in summary['label_distribution'].items()))
        if 'text_length_summary' in summary:
            tls = summary['text_length_summary']
            lines.append('- Text length summary: ' + ', '.join(f'{k}={v}' for k, v in tls.items() if v is not None))
        lines.append('')
    md_path = REPORT_DIR / 'eda_summary.md'
    md_path.write_text('\n'.join(lines), encoding='utf-8')

    print(f'EDA summary written to {json_path}')
    print(f'Markdown report written to {md_path}')


if __name__ == '__main__':
    main()
