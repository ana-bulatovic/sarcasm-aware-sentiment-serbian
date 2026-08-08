"""Inferenca sa sačuvanim baseline ``model.joblib`` pipeline-ima."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
from sklearn.pipeline import Pipeline

from src.baselines.pipeline import CLASSIFIER_NAMES, ClassifierName
from src.common.config import resolve_path
from src.preprocessing.baseline import clean_text_from_config

TaskName = Literal["sentiment", "sarcasm"]


def resolve_baseline_model_path(
    output_dir: Path,
    task: TaskName,
    model: ClassifierName,
) -> Path:
    """Putanja do ``models/baselines/<task>/<model>/model.joblib``."""
    return output_dir / task / model / "model.joblib"


def load_baseline_pipeline(model_path: Path) -> Pipeline:
    """Učitaj sklearn Pipeline (TF-IDF + klasifikator)."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Nema sačuvanog baseline modela: {model_path}\n"
            "Prvo: python scripts/baselines/train_baselines.py --task all"
        )
    pipe = joblib.load(model_path)
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"Očekivan sklearn Pipeline, dobijeno: {type(pipe)}")
    return pipe


def _prepare_texts(
    texts: Sequence[str],
    *,
    preprocess_cfg: dict[str, Any] | None,
    apply_preprocessing: bool,
) -> list[str]:
    """Primeni baseline pretprocesiranje (isto kao pri treningu)."""
    out: list[str] = []
    for raw in texts:
        text = str(raw) if raw is not None else ""
        if apply_preprocessing:
            text = clean_text_from_config(text, preprocess_cfg or {})
        out.append(text)
    return out


def _proba_maps(
    pipe: Pipeline,
    texts: list[str],
) -> list[dict[str, float] | None]:
    """Verovatnoće po klasi ako model podržava ``predict_proba``; inače None.

    ``LinearSVC`` nema ``predict_proba`` — tada vraća listu ``None``.
    """
    try:
        proba = pipe.predict_proba(texts)
    except Exception:
        return [None] * len(texts)

    clf = pipe.named_steps.get("clf")
    class_labels = [str(c) for c in getattr(clf, "classes_", [])]
    if not class_labels and len(proba):
        class_labels = [str(j) for j in range(proba.shape[1])]

    maps: list[dict[str, float] | None] = []
    for row in proba:
        maps.append({class_labels[j]: float(row[j]) for j in range(len(class_labels))})
    return maps


def predict_baseline_texts(
    texts: Sequence[str],
    *,
    config: dict[str, Any],
    task: TaskName,
    model: ClassifierName = "linear_svm",
    model_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Predikcije baseline modela za listu tekstova (labela + opcione verovatnoće).

    Tekst prolazi kroz ``baseline_preprocessing`` (uključujući lowercase),
    isto kao pri treningu — **ne** koristi se za BERTić.
    """
    if not texts:
        return []

    bl_cfg = config.get("baselines", {})
    prep_cfg = config.get("baseline_preprocessing", {})
    apply_prep = bool(bl_cfg.get("apply_preprocessing", True))

    out_root = (
        output_dir
        if output_dir is not None
        else resolve_path(bl_cfg.get("output_dir", "models/baselines"))
    )
    path = model_path or resolve_baseline_model_path(out_root, task, model)
    pipe = load_baseline_pipeline(path)

    cleaned = _prepare_texts(
        texts, preprocess_cfg=prep_cfg, apply_preprocessing=apply_prep
    )
    preds = [str(p) for p in pipe.predict(cleaned)]
    prob_maps = _proba_maps(pipe, cleaned)

    key = "sentiment" if task == "sentiment" else "sarcasm"
    results: list[dict[str, Any]] = []
    for i, raw in enumerate(texts):
        row: dict[str, Any] = {
            "text": str(raw),
            "text_preprocessed": cleaned[i],
            "task": task,
            "model": model,
            key: preds[i],
        }
        if prob_maps[i] is not None:
            row[f"{key}_probs"] = prob_maps[i]
        results.append(row)
    return results


def list_available_baseline_models(
    output_dir: Path,
    tasks: Sequence[TaskName] = ("sentiment", "sarcasm"),
    models: Sequence[ClassifierName] | None = None,
) -> list[tuple[TaskName, ClassifierName, Path]]:
    """Vrati (task, model, path) za sve postojeće ``model.joblib`` fajlove."""
    model_names = list(models) if models else list(CLASSIFIER_NAMES)
    found: list[tuple[TaskName, ClassifierName, Path]] = []
    for task in tasks:
        for name in model_names:
            path = resolve_baseline_model_path(output_dir, task, name)
            if path.exists():
                found.append((task, name, path))
    return found
