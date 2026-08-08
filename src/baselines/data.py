"""Učitavanje i normalizacija CSV-a za baseline eksperimente."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.common.schema import SARCASM_VALUES, SENTIMENT_VALUES
from src.preprocessing.baseline import clean_text_from_config

TaskName = Literal["sentiment", "sarcasm"]

VALID_SENT = set(SENTIMENT_VALUES)
VALID_SARC = set(SARCASM_VALUES)

# Alias-i kolona (korisnički format ↔ interni)
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "source": ("source", "url"),
    "tip": ("tip", "topic"),
    "text": ("text",),
    "id": ("id",),
    "sentiment": ("sentiment",),
    "sarcasm": ("sarcasm",),
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Pronađi prvu postojeću kolonu među aliasima za kanonsko ime."""
    for name in _COLUMN_ALIASES[canonical]:
        if name in df.columns:
            return name
    return None


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Mapiraj url→source, topic→tip; zadrži kanonske kolone."""
    out = df.copy()
    rename: dict[str, str] = {}

    src_col = _resolve_column(out, "source")
    if src_col and src_col != "source":
        rename[src_col] = "source"
    tip_col = _resolve_column(out, "tip")
    if tip_col and tip_col != "tip":
        rename[tip_col] = "tip"

    if rename:
        out = out.rename(columns=rename)

    text_col = _resolve_column(out, "text")
    if text_col is None:
        raise ValueError("CSV mora imati kolonu 'text'.")
    if text_col != "text":
        out = out.rename(columns={text_col: "text"})

    for col in ("sentiment", "sarcasm"):
        if col not in out.columns:
            raise ValueError(f"CSV mora imati kolonu '{col}'.")

    if "id" not in out.columns:
        out["id"] = [f"row-{i:05d}" for i in range(1, len(out) + 1)]
    if "source" not in out.columns:
        out["source"] = ""
    if "tip" not in out.columns:
        out["tip"] = ""

    out["text"] = out["text"].astype(str)
    out["sentiment"] = out["sentiment"].astype(str).str.strip()
    out["sarcasm"] = out["sarcasm"].astype(str).str.strip()
    # float-like "1.0" / "-1.0" → "1" / "-1"
    for col in ("sentiment", "sarcasm"):
        out[col] = out[col].str.replace(r"\.0$", "", regex=True)

    return out


def filter_labeled(df: pd.DataFrame, task: TaskName) -> pd.DataFrame:
    """Zadrži redove sa validnim labelama za dati task (+ neprazan tekst)."""
    has_text = df["text"].astype(str).str.strip() != ""
    if task == "sentiment":
        valid = has_text & df["sentiment"].isin(VALID_SENT)
    else:
        valid = has_text & df["sarcasm"].isin(VALID_SARC)
    return df.loc[valid].copy().reset_index(drop=True)


def load_baseline_frame(
    csv_path: str | Path,
    *,
    task: TaskName,
    preprocess_cfg: dict[str, Any] | None = None,
    apply_preprocessing: bool = True,
) -> pd.DataFrame:
    """Učitaj CSV, filtriraj labele; opciono ``clean_text`` (samo za TF-IDF)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV nije pronađen: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    df = normalize_dataframe(df)
    df = filter_labeled(df, task=task)

    if apply_preprocessing:
        cfg = preprocess_cfg or {}
        df["text_raw"] = df["text"]
        df["text"] = df["text_raw"].map(lambda t: clean_text_from_config(t, cfg))
        # Prazan tekst posle čišćenja — odbaci
        df = df.loc[df["text"].str.strip() != ""].reset_index(drop=True)

    return df


def labels_for_task(df: pd.DataFrame, task: TaskName) -> list[str]:
    """Izvuci listu string labela za dati task (sentiment ili sarcasm)."""
    if task == "sentiment":
        return df["sentiment"].astype(str).tolist()
    return df["sarcasm"].astype(str).tolist()


def class_names_for_task(task: TaskName) -> list[str]:
    """Kanonski redosled imena klasa za task (iz sheme)."""
    if task == "sentiment":
        return list(SENTIMENT_VALUES)
    return list(SARCASM_VALUES)
