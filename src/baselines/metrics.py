"""Metrike za baseline klasifikatore."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_baseline_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str],
) -> dict[str, Any]:
    """Izračunaj accuracy, macro P/R/F1 i confusion matrix za baseline model.

    ``labels`` određuje redosled klasa u matrici (npr. sentiment ``1/0/-1``).
    """
    y_true_arr = np.asarray(y_true, dtype=str)
    y_pred_arr = np.asarray(y_pred, dtype=str)
    label_list = list(labels)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=label_list)
    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=label_list,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(
                y_true_arr, y_pred_arr, average="macro", labels=label_list, zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_arr, y_pred_arr, average="macro", labels=label_list, zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true_arr, y_pred_arr, average="macro", labels=label_list, zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                average="weighted",
                labels=label_list,
                zero_division=0,
            )
        ),
        "confusion_matrix": {
            "labels": label_list,
            "matrix": cm.tolist(),
        },
        "per_class": report,
        "n": int(len(y_true_arr)),
    }
