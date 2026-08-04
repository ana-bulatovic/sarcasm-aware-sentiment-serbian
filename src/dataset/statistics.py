"""Osnovne statistike dataseta (radi i pre i posle anotacije)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import resolve_path
from src.common.schema import SARCASM_VALUES, SENTIMENT_VALUES


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return text


def compute_dataset_statistics(
    config: dict[str, Any] | None = None,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Izračunaj brojeve po izvoru, sentimentu, sarkazmu i kombinacijama."""
    if csv_path is None:
        if config is None:
            raise ValueError("Potreban je config ili csv_path.")
        csv_path = resolve_path(config["paths"]["dataset_csv"])
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset nije pronađen: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")
    total = len(df)

    by_source = Counter(df["source"].astype(str).tolist()) if "source" in df.columns else {}

    sentiments = df["sentiment"].map(_norm_label) if "sentiment" in df.columns else pd.Series([""] * total)
    sarcasms = df["sarcasm"].map(_norm_label) if "sarcasm" in df.columns else pd.Series([""] * total)

    sentiment_counts = {
        label: int((sentiments == label).sum()) for label in SENTIMENT_VALUES
    }
    sentiment_counts["unlabeled"] = int((sentiments == "").sum())
    sentiment_counts["other"] = int(
        (~sentiments.isin(list(SENTIMENT_VALUES) + [""])).sum()
    )

    sarcasm_counts = {label: int((sarcasms == label).sum()) for label in SARCASM_VALUES}
    sarcasm_counts["unlabeled"] = int((sarcasms == "").sum())
    sarcasm_counts["other"] = int((~sarcasms.isin(list(SARCASM_VALUES) + [""])).sum())

    combinations: dict[str, int] = {}
    for s in SENTIMENT_VALUES:
        for c in SARCASM_VALUES:
            key = f"{s}/{c}"
            combinations[key] = int(((sentiments == s) & (sarcasms == c)).sum())
    combinations["unlabeled_pair"] = int(((sentiments == "") | (sarcasms == "")).sum())

    stats: dict[str, Any] = {
        "total_texts": total,
        "by_source": dict(sorted(by_source.items())),
        "sentiment": sentiment_counts,
        "sarcasm": sarcasm_counts,
        "sentiment_sarcasm_combinations": combinations,
        "annotation_progress": {
            "fully_labeled": int(((sentiments != "") & (sarcasms != "")).sum()),
            "missing_sentiment": int((sentiments == "").sum()),
            "missing_sarcasm": int((sarcasms == "").sum()),
        },
    }

    if config is not None:
        out = resolve_path(config["paths"]["stats_json"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[stats] Sačuvano → {out}")

    return stats


def print_statistics(stats: dict[str, Any]) -> None:
    print("\n=== Dataset statistike ===")
    print(f"Ukupno tekstova: {stats['total_texts']}")
    print("\nPo izvoru:")
    for source, count in stats["by_source"].items():
        print(f"  {source}: {count}")
    print("\nSentiment:")
    for k, v in stats["sentiment"].items():
        print(f"  {k}: {v}")
    print("\nSarkazam:")
    for k, v in stats["sarcasm"].items():
        print(f"  {k}: {v}")
    print("\nKombinacije sentiment/sarkazam:")
    for k, v in stats["sentiment_sarcasm_combinations"].items():
        print(f"  {k}: {v}")
    print("\nNapredak anotacije:")
    for k, v in stats["annotation_progress"].items():
        print(f"  {k}: {v}")
