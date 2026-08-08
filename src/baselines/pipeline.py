"""TF-IDF + klasifikatori za baseline eksperimente."""

from __future__ import annotations

import inspect
from typing import Any, Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ClassifierName = Literal["naive_bayes", "logistic_regression", "linear_svm"]

CLASSIFIER_NAMES: tuple[ClassifierName, ...] = (
    "naive_bayes",
    "logistic_regression",
    "linear_svm",
)


def build_tfidf(cfg: dict[str, Any] | None = None) -> TfidfVectorizer:
    """Napravi ``TfidfVectorizer`` iz ``baselines.tfidf`` podešavanja."""
    cfg = cfg or {}
    ngram = cfg.get("ngram_range", [1, 2])
    if isinstance(ngram, list):
        ngram_range = (int(ngram[0]), int(ngram[1]))
    else:
        ngram_range = (1, 2)

    return TfidfVectorizer(
        max_features=int(cfg.get("max_features", 20000)) or None,
        ngram_range=ngram_range,
        min_df=int(cfg.get("min_df", 2)),
        max_df=float(cfg.get("max_df", 0.95)),
        sublinear_tf=bool(cfg.get("sublinear_tf", True)),
        analyzer=str(cfg.get("analyzer", "word")),
    )


def build_classifier(name: ClassifierName, cfg: dict[str, Any] | None = None) -> Any:
    """Instanciraj klasifikator; NB nema ``class_weight`` (vidi runner ``sample_weight``)."""
    cfg = cfg or {}
    # None / "none" / False → bez class_weight (podrazumevano balanced gde postoji)
    raw_cw = cfg.get("class_weight", "balanced")
    if raw_cw in (None, False, "none", "None", ""):
        class_weight = None
    else:
        class_weight = raw_cw

    if name == "naive_bayes":
        # MultinomialNB nema class_weight; balansiranje ide preko sample_weight pri fit-u
        return MultinomialNB(alpha=float(cfg.get("alpha", 1.0)))
    if name == "logistic_regression":
        # multi_class je uklonjen u novijem sklearn (≥1.8); starije verzije i dalje prihvataju
        lr_kwargs: dict[str, Any] = {
            "C": float(cfg.get("C", 1.0)),
            "max_iter": int(cfg.get("max_iter", 2000)),
            "class_weight": class_weight,
            "solver": str(cfg.get("solver", "lbfgs")),
            "random_state": int(cfg.get("random_state", 42)),
        }
        lr_params = inspect.signature(LogisticRegression.__init__).parameters
        if "multi_class" in lr_params:
            lr_kwargs["multi_class"] = str(cfg.get("multi_class", "auto"))
        return LogisticRegression(**lr_kwargs)
    if name == "linear_svm":
        return LinearSVC(
            C=float(cfg.get("C", 1.0)),
            class_weight=class_weight,
            max_iter=int(cfg.get("max_iter", 5000)),
            dual=cfg.get("dual", "auto"),
            random_state=int(cfg.get("random_state", 42)),
        )
    raise ValueError(f"Nepoznat klasifikator: {name}")


def build_pipeline(
    name: ClassifierName,
    *,
    tfidf_cfg: dict[str, Any] | None = None,
    clf_cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """Sklearn ``Pipeline``: TF-IDF → klasifikator (``tfidf``, ``clf`` koraci)."""
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf(tfidf_cfg)),
            ("clf", build_classifier(name, clf_cfg)),
        ]
    )
