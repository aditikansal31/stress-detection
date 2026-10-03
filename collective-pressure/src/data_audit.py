"""Audit local project datasets and optionally download Dreaddit for inspection."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATASET_ROOT = WORKSPACE_ROOT / "datasets"
GOEMOTIONS_ROOT = WORKSPACE_ROOT / "goemotions"
HF_CACHE = PROJECT_ROOT / "data" / "cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE / "datasets"))

import pandas as pd

LOGGER = logging.getLogger("dataset_audit")
TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".parquet"}
TEXT_NAME_PARTS = ("text", "comment", "content", "body", "title", "post")
LABEL_NAME_PARTS = ("label", "target", "stress", "emotion")
GOEMOTIONS_SPLIT_NAMES = {"train.tsv", "dev.tsv", "test.tsv"}


def _is_text_column(name: str) -> bool:
    lowered = name.casefold()
    return any(part in lowered for part in TEXT_NAME_PARTS)


def _serializable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def summarize_frame(frame: pd.DataFrame, source: str, table: str) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, int]] = {}
    text_lengths: dict[str, dict[str, float | int]] = {}
    binary_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        name = str(column)
        non_null = series.dropna()
        unique_count = int(non_null.nunique())
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 3)
                if len(frame)
                else 0.0,
                "unique_non_null": unique_count,
            }
        )

        if _is_text_column(name):
            lengths = non_null.astype(str).str.len()
            if not lengths.empty:
                quantiles = lengths.quantile([0.5, 0.9, 0.99]).to_dict()
                text_lengths[name] = {
                    "empty_or_whitespace": int(non_null.astype(str).str.strip().eq("").sum()),
                    "mean_characters": round(float(lengths.mean()), 2),
                    "median_characters": round(float(quantiles[0.5]), 2),
                    "p90_characters": round(float(quantiles[0.9]), 2),
                    "p99_characters": round(float(quantiles[0.99]), 2),
                }

        if unique_count <= 2 and unique_count > 0:
            values = set(non_null.astype(str).unique())
            if values.issubset({"0", "1", "0.0", "1.0", "False", "True"}):
                binary_columns.append(name)

        if unique_count <= 30 and any(part in name.casefold() for part in LABEL_NAME_PARTS):
            counts = series.value_counts(dropna=False).head(30)
            distributions[name] = {
                str(_serializable(value)): int(count)
                for value, count in counts.items()
            }

    if len(binary_columns) >= 5:
        for name in binary_columns:
            counts = frame[name].value_counts(dropna=False)
            distributions[name] = {
                str(_serializable(value)): int(count) for value, count in counts.items()
            }

    safe_preview: dict[str, list[Any]] = {}
    for column in frame.columns:
        name = str(column)
        lowered = name.casefold()
        if _is_text_column(name) or any(token in lowered for token in ("author", "user", "_id", "id")):
            continue
        safe_preview[name] = [_serializable(value) for value in frame[name].head(3).tolist()]

    try:
        duplicate_rows = int(frame.duplicated().sum())
    except (TypeError, ValueError):
        duplicate_rows = None

    return {
        "source": source,
        "table": table,
        "rows": int(len(frame)),
        "columns_count": int(len(frame.columns)),
        "columns": columns,
        "duplicate_rows": duplicate_rows,
        "text_lengths": text_lengths,
        "label_distributions": distributions,
        "safe_non_text_preview": safe_preview,
    }


def _read_tabular(path: Path) -> dict[str, pd.DataFrame]:
    suffix = path.suffix.casefold()
    if suffix == ".xlsx":
        return pd.read_excel(path, sheet_name=None, engine="openpyxl")
    if suffix == ".parquet":
        return {path.stem: pd.read_parquet(path)}
    if suffix == ".tsv":
        if path.name in GOEMOTIONS_SPLIT_NAMES and path.parent.name == "data":
            return {
                path.stem: pd.read_csv(
                    path,
                    sep="\t",
                    header=None,
                    names=["text", "emotion_ids", "comment_id"],
                    dtype={"text": "string", "emotion_ids": "string", "comment_id": "string"},
                )
            }
        return {path.stem: pd.read_csv(path, sep="\t", low_memory=False)}
    return {path.stem: pd.read_csv(path, low_memory=False)}


def _audit_local_files() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    roots = [root for root in (DATASET_ROOT, GOEMOTIONS_ROOT) if root.exists()]
    files = sorted(
        {
            path
            for root in roots
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in TABULAR_SUFFIXES
            and ":Zone" not in path.name
        }
    )
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in files:
        try:
            if path.suffix.casefold() == ".tsv" and path.name in GOEMOTIONS_SPLIT_NAMES:
                # The local GoEmotions README documents these splits as headerless text/labels/id TSVs.
                pass
            tables = _read_tabular(path)
            for table_name, frame in tables.items():
                records.append(
                    summarize_frame(
                        frame,
                        source=str(path.relative_to(WORKSPACE_ROOT)),
                        table=table_name,
                    )
                )
                LOGGER.info("Audited %s (%s rows)", path.relative_to(WORKSPACE_ROOT), len(frame))
        except Exception as error:  # Keep auditing other files, but report the failure explicitly.
            LOGGER.exception("Could not audit %s", path)
            errors.append({"source": str(path), "error": f"{type(error).__name__}: {error}"})

    return records, errors


def _audit_dreaddit() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "andreagasparini/dreaddit",
            cache_dir=str(HF_CACHE / "datasets"),
        )
        records = []
        for split_name, split_dataset in dataset.items():
            frame = split_dataset.to_pandas()
            records.append(summarize_frame(frame, "Hugging Face: andreagasparini/dreaddit", split_name))
            LOGGER.info("Audited Dreaddit %s (%s rows)", split_name, len(frame))
        return records, []
    except Exception as error:
        LOGGER.exception("Could not download or audit Dreaddit")
        return [], [{"source": "andreagasparini/dreaddit", "error": f"{type(error).__name__}: {error}"}]


def write_reports(records: list[dict[str, Any]], errors: list[dict[str, str]]) -> None:
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {"records": records, "errors": errors}
    (reports_dir / "dataset_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Dataset Audit",
        "",
        "This report describes file schemas and aggregate counts. It does not infer an individual's mental or clinical condition.",
        "",
        f"Files/tables successfully audited: {len(records)}",
        f"Audit errors: {len(errors)}",
        "",
        "| Source | Table | Rows | Columns | Duplicate rows |",
        "|---|---|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| `{record['source']}` | `{record['table']}` | {record['rows']} | "
            f"{record['columns_count']} | {record['duplicate_rows']} |"
        )
    for error in errors:
        lines.extend(["", f"- **Audit error:** `{error['source']}`: {error['error']}"])
    (reports_dir / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dreaddit",
        action="store_true",
        help="Download and audit andreagasparini/dreaddit from Hugging Face.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    records, errors = _audit_local_files()
    if args.download_dreaddit:
        remote_records, remote_errors = _audit_dreaddit()
        records.extend(remote_records)
        errors.extend(remote_errors)

    write_reports(records, errors)
    print(f"Audited {len(records)} tables; errors: {len(errors)}")
    print(f"Report: {PROJECT_ROOT / 'reports' / 'dataset_audit.md'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())