"""Metrike: accuracy, macro-F1, podskup sarcasm=yes."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
    target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=list(target_names) if target_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "per_class": report,
        "n": int(len(y_true_arr)),
    }


def subset_where(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    mask: Sequence[bool],
    labels: Sequence[int] | None = None,
    target_names: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    mask_arr = np.asarray(mask, dtype=bool)
    if not mask_arr.any():
        return None
    y_true_arr = np.asarray(y_true)[mask_arr]
    y_pred_arr = np.asarray(y_pred)[mask_arr]
    return classification_metrics(
        y_true_arr, y_pred_arr, labels=labels, target_names=target_names
    )


def pack_single_task_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    sarcasm_raw: Sequence[str],
    label_names: Sequence[str],
) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    overall = classification_metrics(
        y_true, y_pred, labels=labels, target_names=label_names
    )
    sarcasm_mask = [str(x).strip().lower() == "yes" for x in sarcasm_raw]
    on_sarcastic = subset_where(
        y_true, y_pred, sarcasm_mask, labels=labels, target_names=label_names
    )
    return {
        "overall": overall,
        "on_sarcasm_yes": on_sarcastic,
        "n_sarcasm_yes": int(sum(sarcasm_mask)),
    }


def pack_multitask_metrics(
    sent_true: Sequence[int],
    sent_pred: Sequence[int],
    sarc_true: Sequence[int],
    sarc_pred: Sequence[int],
    sarcasm_raw: Sequence[str],
    sentiment_names: Sequence[str],
    sarcasm_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "sentiment": pack_single_task_metrics(
            sent_true, sent_pred, sarcasm_raw, sentiment_names
        ),
        "sarcasm": pack_single_task_metrics(
            sarc_true, sarc_pred, sarcasm_raw, sarcasm_names
        ),
    }
