"""Zajednički flagovi za trening (baseline + BERTić).

Čita ``training.use_class_weights`` (prioritet), pa ``modeling.use_class_weights``.
Koristi se i u ``src.baselines`` i u ``src.modeling``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def resolve_use_class_weights(config: dict[str, Any], default: bool = True) -> bool:
    """Pročitaj ``training.use_class_weights`` (prioritet), zatim ``modeling``.

    Ako je ``false``, koristi se običan (neponderisan) loss / bez class_weight.
    """
    training = config.get("training")
    if isinstance(training, dict) and "use_class_weights" in training:
        return bool(training["use_class_weights"])
    modeling = config.get("modeling")
    if isinstance(modeling, dict) and "use_class_weights" in modeling:
        return bool(modeling["use_class_weights"])
    return default


def balanced_class_weight_map(
    y: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """sklearn ``balanced`` težine po labeli (string ključevi)."""
    y_arr = np.asarray([str(v) for v in y], dtype=str)
    if labels is None:
        classes = np.unique(y_arr)
    else:
        classes = np.asarray([str(x) for x in labels], dtype=str)
        # samo klase koje postoje u y (inače compute_class_weight puca)
        present = set(y_arr.tolist())
        classes = np.asarray([c for c in classes if c in present], dtype=str)
        if len(classes) == 0:
            classes = np.unique(y_arr)

    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_arr)
    return {str(c): float(w) for c, w in zip(classes, weights)}


def balanced_sample_weights(y: Sequence[str]) -> np.ndarray:
    """Po-primeru težine ekvivalentne class_weight='balanced'."""
    y_arr = np.asarray([str(v) for v in y], dtype=str)
    return compute_sample_weight(class_weight="balanced", y=y_arr)
