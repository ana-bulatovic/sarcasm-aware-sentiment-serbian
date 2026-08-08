"""Class weights i sample weights za balansiranje (iz train splita)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

from src.modeling.labels import SARCASM_LABEL2ID, SENTIMENT_LABEL2ID


def _label_ids(series: pd.Series, label2id: dict[str, int]) -> np.ndarray:
    """Mapiraj string labele iz Series u numpy niz ID-jeva."""
    return np.asarray(
        [label2id[str(x).strip()] for x in series.astype(str)],
        dtype=np.int64,
    )


def compute_class_weights_from_train(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Izračunaj balanced class weights za sentiment i sarcasm iz train.csv."""
    sent_ids = _label_ids(train_df["sentiment"], SENTIMENT_LABEL2ID)
    sarc_ids = _label_ids(train_df["sarcasm"], SARCASM_LABEL2ID)

    sent_classes = np.arange(len(SENTIMENT_LABEL2ID), dtype=np.int64)
    sarc_classes = np.arange(len(SARCASM_LABEL2ID), dtype=np.int64)

    # Ako neka klasa fali u train-u, stavi weight 1.0
    sent_present = np.unique(sent_ids)
    sarc_present = np.unique(sarc_ids)

    sent_w = np.ones(len(sent_classes), dtype=np.float64)
    if len(sent_present):
        bal = compute_class_weight(
            class_weight="balanced", classes=sent_present, y=sent_ids
        )
        for cls, w in zip(sent_present, bal):
            sent_w[int(cls)] = float(w)

    sarc_w = np.ones(len(sarc_classes), dtype=np.float64)
    if len(sarc_present):
        bal = compute_class_weight(
            class_weight="balanced", classes=sarc_present, y=sarc_ids
        )
        for cls, w in zip(sarc_present, bal):
            sarc_w[int(cls)] = float(w)

    # Brojači po klasi / kombinaciji (za log)
    sent_counts = {
        label: int((train_df["sentiment"].astype(str).str.strip() == label).sum())
        for label in SENTIMENT_LABEL2ID
    }
    sarc_counts = {
        label: int((train_df["sarcasm"].astype(str).str.strip() == label).sum())
        for label in SARCASM_LABEL2ID
    }
    combo_counts: dict[str, int] = {}
    for s in SENTIMENT_LABEL2ID:
        for c in SARCASM_LABEL2ID:
            key = f"{s}|{c}"
            combo_counts[key] = int(
                (
                    (train_df["sentiment"].astype(str).str.strip() == s)
                    & (train_df["sarcasm"].astype(str).str.strip() == c)
                ).sum()
            )

    info = {
        "sentiment_class_weights": {
            label: float(sent_w[i]) for label, i in SENTIMENT_LABEL2ID.items()
        },
        "sarcasm_class_weights": {
            label: float(sarc_w[i]) for label, i in SARCASM_LABEL2ID.items()
        },
        "sentiment_counts": sent_counts,
        "sarcasm_counts": sarc_counts,
        "combo_counts": combo_counts,
        "n_train": int(len(train_df)),
    }
    return {
        "sentiment_weights_tensor": torch.tensor(sent_w, dtype=torch.float32),
        "sarcasm_weights_tensor": torch.tensor(sarc_w, dtype=torch.float32),
        "info": info,
    }


def compute_combo_sample_weights(train_df: pd.DataFrame) -> torch.DoubleTensor:
    """Težina po primeru = 1 / frekvencija (sentiment|sarcasm) kombinacije.

    Retke kombinacije (npr. 1|1, 0|1, 0|0) dobijaju veću verovatnoću u sampleru.
    """
    sent = train_df["sentiment"].astype(str).str.strip()
    sarc = train_df["sarcasm"].astype(str).str.strip()
    combo = sent + "|" + sarc
    counts = combo.value_counts()
    weights = combo.map(lambda k: 1.0 / float(counts[k])).astype(float)
    # Normalizacija nije neophodna za WeightedRandomSampler, ali pomaže numerici
    arr = weights.to_numpy(dtype=np.float64)
    return torch.as_tensor(arr, dtype=torch.double)
