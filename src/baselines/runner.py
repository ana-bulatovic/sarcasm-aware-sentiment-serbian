"""Pokretanje baseline eksperimenata i čuvanje rezultata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.baselines.data import (
    TaskName,
    class_names_for_task,
    labels_for_task,
    load_baseline_frame,
)
from src.baselines.metrics import compute_baseline_metrics
from src.baselines.pipeline import CLASSIFIER_NAMES, ClassifierName, build_pipeline
from src.common.config import resolve_path
from src.common.training_flags import resolve_use_class_weights

TaskChoice = Literal["sentiment", "sarcasm", "all"]


def _json_ready(obj: Any) -> Any:
    """Rekurzivno pretvori numpy/skalar tipove u JSON-serializabilne vrednosti."""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    """Upisi rečnik kao UTF-8 JSON (sa indentom)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _split_frame(
    df: pd.DataFrame,
    task: TaskName,
    *,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratifikovani train/test split; fallback bez stratifikacije ako treba."""
    y = labels_for_task(df, task)
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    except ValueError:
        # Premalo primera po klasi za stratifikaciju
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _run_one(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task: TaskName,
    model_name: ClassifierName,
    tfidf_cfg: dict[str, Any],
    clf_cfg: dict[str, Any],
    out_dir: Path,
    save_model: bool,
    use_class_weights: bool,
) -> dict[str, Any]:
    """Treniraj jedan model; NB koristi ``sample_weight``, ostali ``class_weight``."""
    from src.common.training_flags import (
        balanced_class_weight_map,
        balanced_sample_weights,
    )

    labels = class_names_for_task(task)
    X_train = train_df["text"].astype(str).tolist()
    y_train = labels_for_task(train_df, task)
    X_test = test_df["text"].astype(str).tolist()
    y_test = labels_for_task(test_df, task)

    # class_weight='balanced' gde sklearn podržava; NB → sample_weight (nema class_weight)
    fit_params: dict[str, Any] = {}
    class_weights_used: dict[str, float] | None = None
    class_weight_param: str | None = None
    sample_weight_mode: str | None = None

    cfg = dict(clf_cfg)
    if use_class_weights:
        class_weights_used = balanced_class_weight_map(y_train, labels=labels)
        if model_name == "naive_bayes":
            fit_params["clf__sample_weight"] = balanced_sample_weights(y_train)
            sample_weight_mode = "balanced"
            cfg["class_weight"] = None
        else:
            cfg["class_weight"] = "balanced"
            class_weight_param = "balanced"
    else:
        cfg["class_weight"] = None

    pipe = build_pipeline(model_name, tfidf_cfg=tfidf_cfg, clf_cfg=cfg)
    pipe.fit(X_train, y_train, **fit_params)
    y_pred = pipe.predict(X_test).tolist()

    metrics = compute_baseline_metrics(y_test, y_pred, labels=labels)

    task_dir = out_dir / task / model_name
    task_dir.mkdir(parents=True, exist_ok=True)

    pred_df = test_df[["id", "text", "sentiment", "sarcasm"]].copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(task_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    result = {
        "task": task,
        "model": model_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "labels": labels,
        "metrics": metrics,
    }
    _save_json(task_dir / "metrics.json", result)

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
    cm_df.to_csv(task_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    run_meta = {
        "task": task,
        "model": model_name,
        "use_class_weights": use_class_weights,
        "class_weight_param": class_weight_param,
        "sample_weight": sample_weight_mode,
        "class_weights_used": class_weights_used,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "labels": labels,
    }
    _save_json(task_dir / "run_meta.json", run_meta)

    if save_model:
        joblib.dump(pipe, task_dir / "model.joblib")

    return result


def run_baseline_experiments(
    config: dict[str, Any],
    *,
    csv_path: str | Path | None = None,
    tasks: Sequence[TaskChoice] | TaskChoice = "all",
    models: Sequence[ClassifierName] | None = None,
    output_dir: str | Path | None = None,
    test_size: float | None = None,
    save_model: bool | None = None,
) -> dict[str, Any]:
    """Pokreni TF-IDF baseline-e za sentiment i/ili sarkazam; sačuvaj summary."""
    bl_cfg = config.get("baselines", {})
    prep_cfg = config.get("baseline_preprocessing", {})
    seed = int(config.get("random_seed", 42))
    use_class_weights = resolve_use_class_weights(config, default=True)

    if csv_path is None:
        csv_path = resolve_path(
            bl_cfg.get("csv")
            or config.get("paths", {}).get("dataset_csv", "data/processed/dataset/dataset.csv")
        )
    else:
        csv_path = Path(csv_path)

    out = (
        resolve_path(output_dir)
        if output_dir is not None
        else resolve_path(bl_cfg.get("output_dir", "models/baselines"))
    )
    out.mkdir(parents=True, exist_ok=True)

    test_frac = float(test_size if test_size is not None else bl_cfg.get("test_size", 0.2))
    do_save = bool(save_model if save_model is not None else bl_cfg.get("save_model", True))
    tfidf_cfg = dict(bl_cfg.get("tfidf", {}))
    classifiers_cfg = dict(bl_cfg.get("classifiers", {}))

    if isinstance(tasks, str):
        task_list: list[TaskName] = (
            ["sentiment", "sarcasm"] if tasks == "all" else [tasks]  # type: ignore[list-item]
        )
    else:
        task_list = []
        for t in tasks:
            if t == "all":
                task_list.extend(["sentiment", "sarcasm"])
            else:
                task_list.append(t)  # type: ignore[arg-type]
        # unique preserve order
        seen: set[str] = set()
        task_list = [t for t in task_list if not (t in seen or seen.add(t))]

    model_list: list[ClassifierName] = list(models) if models else list(CLASSIFIER_NAMES)

    summary: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(csv_path),
        "seed": seed,
        "test_size": test_frac,
        "use_class_weights": use_class_weights,
        "preprocessing": prep_cfg,
        "tfidf": tfidf_cfg,
        "tasks": {},
    }

    print(f"[baselines] CSV: {csv_path}")
    print(f"[baselines] izlaz: {out}")
    print(f"[baselines] use_class_weights={use_class_weights}")
    print(f"[baselines] taskovi: {task_list} | modeli: {model_list}")

    for task in task_list:
        df = load_baseline_frame(
            csv_path,
            task=task,
            preprocess_cfg=prep_cfg,
            apply_preprocessing=bool(bl_cfg.get("apply_preprocessing", True)),
        )
        if len(df) < 10:
            raise RuntimeError(
                f"Premalo validnih primera za task={task}: {len(df)} (CSV: {csv_path})"
            )

        train_df, test_df = _split_frame(df, task, test_size=test_frac, seed=seed)
        # Sačuvaj split za reprodukciju
        split_dir = out / task / "_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_dir / "train.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(split_dir / "test.csv", index=False, encoding="utf-8-sig")

        task_results: dict[str, Any] = {
            "n_labeled": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "models": {},
        }

        print(f"\n=== Task: {task} (train={len(train_df)}, test={len(test_df)}) ===")
        for model_name in model_list:
            clf_cfg = dict(classifiers_cfg.get(model_name, {}))
            clf_cfg.setdefault("random_state", seed)
            result = _run_one(
                train_df=train_df,
                test_df=test_df,
                task=task,
                model_name=model_name,
                tfidf_cfg=tfidf_cfg,
                clf_cfg=clf_cfg,
                out_dir=out,
                save_model=do_save,
                use_class_weights=use_class_weights,
            )
            m = result["metrics"]
            print(
                f"  {model_name:22s}  "
                f"acc={m['accuracy']:.4f}  "
                f"P={m['precision_macro']:.4f}  "
                f"R={m['recall_macro']:.4f}  "
                f"macroF1={m['macro_f1']:.4f}"
            )
            task_results["models"][model_name] = {
                "accuracy": m["accuracy"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "n_test": m["n"],
            }

        summary["tasks"][task] = task_results

    _save_json(out / "summary.json", summary)
    print(f"\n[baselines] summary → {out / 'summary.json'}")
    return summary
