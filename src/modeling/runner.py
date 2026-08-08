"""Orkestracija treninga jednog ili svih taskova."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from src.common.config import ensure_dir, resolve_path
from src.common.training_flags import resolve_use_class_weights
from src.modeling.balancing import (
    compute_class_weights_from_train,
    compute_combo_sample_weights,
)
from src.modeling.data import load_splits, make_loader
from src.modeling.models import MultiTaskModel, build_single_task_model
from src.modeling.train_loop import (
    evaluate_multitask,
    evaluate_single_task,
    train_one_task,
    _to_jsonable,
)


def set_seed(seed: int) -> None:
    """Fiksiranje seed-a za random / numpy / torch (i CUDA ako postoji)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    """Vrati ``torch.device``; podrazumevano CUDA ako je dostupna, inače CPU."""
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def modeling_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Popuni podrazumevana ``modeling`` podešavanja iz config-a."""
    m = dict(config.get("modeling") or {})
    m.setdefault("model_name", "classla/bcms-bertic")
    m.setdefault("max_length", 128)
    m.setdefault("batch_size", 16)
    m.setdefault("eval_batch_size", 32)
    m.setdefault("learning_rate", 2.0e-5)
    m.setdefault("weight_decay", 0.01)
    m.setdefault("num_epochs", 4)
    m.setdefault("warmup_ratio", 0.1)
    m.setdefault("max_grad_norm", 1.0)
    m.setdefault("output_dir", "models")
    m.setdefault("device", None)
    # training.use_class_weights ima prioritet (vidi resolve_use_class_weights)
    m["use_class_weights"] = resolve_use_class_weights(config, default=True)
    m.setdefault("use_weighted_sampler", True)
    mt = dict(m.get("multitask") or {})
    mt.setdefault("sentiment_loss_weight", 1.0)
    mt.setdefault("sarcasm_loss_weight", 1.0)
    mt.setdefault("dropout", 0.1)
    m["multitask"] = mt
    return m


def _print_metrics_brief(task: str, metrics: dict[str, Any]) -> None:
    """Kratak ispis accuracy / macro-F1 (i podskup sarcasm=yes)."""
    if task == "multitask":
        s = metrics["sentiment"]["overall"]
        c = metrics["sarcasm"]["overall"]
        print(
            f"  sentiment: acc={s['accuracy']:.3f} macroF1={s['macro_f1']:.3f} | "
            f"sarcasm: acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f}"
        )
        sy = metrics["sentiment"].get("on_sarcasm_yes")
        if sy:
            print(
                f"  sentiment @ sarcasm=yes: acc={sy['accuracy']:.3f} "
                f"macroF1={sy['macro_f1']:.3f} (n={sy['n']})"
            )
    else:
        o = metrics["overall"]
        print(f"  overall: acc={o['accuracy']:.3f} macroF1={o['macro_f1']:.3f}")
        sy = metrics.get("on_sarcasm_yes")
        if sy:
            print(
                f"  @ sarcasm=yes: acc={sy['accuracy']:.3f} "
                f"macroF1={sy['macro_f1']:.3f} (n={sy['n']})"
            )


def run_training(
    config: dict[str, Any],
    task: str,
    splits_dir: Path | None = None,
    output_dir: Path | None = None,
    model_name: str | None = None,
    num_epochs: int | None = None,
    batch_size: int | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune jednog taska (sentiment / sarcasm / multitask) i sačuvaj artefakte."""
    if task not in {"sentiment", "sarcasm", "multitask"}:
        raise ValueError(f"Nepoznat task: {task}")

    seed = int(config.get("random_seed", 42))
    set_seed(seed)
    mcfg = modeling_defaults(config)

    if model_name:
        mcfg["model_name"] = model_name
    if num_epochs is not None:
        mcfg["num_epochs"] = num_epochs
    if batch_size is not None:
        mcfg["batch_size"] = batch_size

    device = resolve_device(device_name or mcfg.get("device"))
    splits_path = splits_dir or resolve_path(
        config.get("paths", {}).get("splits_dir", "data/processed/splits")
    )
    out_root = output_dir or resolve_path(mcfg["output_dir"])
    task_out = ensure_dir(out_root / task)

    print(f"[train] task={task}")
    print(f"[train] model={mcfg['model_name']}")
    print(f"[train] device={device}")
    print(f"[train] splits={splits_path}")
    print(f"[train] output={task_out}")

    splits = load_splits(splits_path)
    for name, df in splits.items():
        print(f"[train] {name}: {len(df)} primera")

    balance = compute_class_weights_from_train(splits["train"])
    balance_info = balance["info"]
    print(f"[train] sentiment class weights: {balance_info['sentiment_class_weights']}")
    print(f"[train] sarcasm class weights: {balance_info['sarcasm_class_weights']}")
    print(f"[train] combo counts: {balance_info['combo_counts']}")

    use_cw = bool(mcfg.get("use_class_weights", True))
    use_sampler = bool(mcfg.get("use_weighted_sampler", True))
    print(f"[train] use_class_weights={use_cw} (weighted CrossEntropyLoss)")
    print(f"[train] use_weighted_sampler={use_sampler}")

    sample_weights = None
    if use_sampler:
        sample_weights = compute_combo_sample_weights(splits["train"])
        print("[train] WeightedRandomSampler: ON (train only, po sentiment|sarcasm kombinaciji)")
    else:
        print("[train] WeightedRandomSampler: OFF")

    tokenizer = AutoTokenizer.from_pretrained(mcfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = make_loader(
        splits["train"],
        tokenizer,
        task=task,
        batch_size=int(mcfg["batch_size"]),
        max_length=int(mcfg["max_length"]),
        shuffle=True,
        seed=seed,
        sample_weights=sample_weights,
    )
    val_loader = make_loader(
        splits["val"],
        tokenizer,
        task=task,
        batch_size=int(mcfg["eval_batch_size"]),
        max_length=int(mcfg["max_length"]),
        shuffle=False,
        seed=seed,
    )
    test_loader = make_loader(
        splits["test"],
        tokenizer,
        task=task,
        batch_size=int(mcfg["eval_batch_size"]),
        max_length=int(mcfg["max_length"]),
        shuffle=False,
        seed=seed,
    )

    sent_cw = balance["sentiment_weights_tensor"] if use_cw else None
    sarc_cw = balance["sarcasm_weights_tensor"] if use_cw else None

    if task == "multitask":
        mt = mcfg["multitask"]
        model = MultiTaskModel(
            mcfg["model_name"],
            sentiment_weight=float(mt["sentiment_loss_weight"]),
            sarcasm_weight=float(mt["sarcasm_loss_weight"]),
            dropout=float(mt["dropout"]),
            sentiment_class_weights=sent_cw,
            sarcasm_class_weights=sarc_cw,
        )
        task_class_weights = None
    else:
        model = build_single_task_model(mcfg["model_name"], task)
        task_class_weights = sent_cw if task == "sentiment" else sarc_cw

    # Sačuvaj tokenizer + meta uz checkpoint
    tokenizer.save_pretrained(task_out)

    if use_cw:
        class_weights_used = {
            "sentiment": balance_info["sentiment_class_weights"],
            "sarcasm": balance_info["sarcasm_class_weights"],
        }
        loss_note = "weighted_cross_entropy"
    else:
        # Običan CE — efektivne težine 1.0 (ne ulaze u loss kao weight=None za single-task)
        class_weights_used = {
            "sentiment": {k: 1.0 for k in balance_info["sentiment_class_weights"]},
            "sarcasm": {k: 1.0 for k in balance_info["sarcasm_class_weights"]},
        }
        loss_note = "cross_entropy"

    meta = {
        "task": task,
        "model_name": mcfg["model_name"],
        "max_length": mcfg["max_length"],
        "seed": seed,
        "splits_dir": str(splits_path),
        "use_class_weights": use_cw,
        "use_weighted_sampler": use_sampler,
        "loss": loss_note,
        "class_weights_used": class_weights_used,
        "class_balance": balance_info,
    }
    (task_out / "run_meta.json").write_text(
        json.dumps(_to_jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = train_one_task(
        task=task,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        output_dir=task_out,
        num_epochs=int(mcfg["num_epochs"]),
        learning_rate=float(mcfg["learning_rate"]),
        weight_decay=float(mcfg["weight_decay"]),
        warmup_ratio=float(mcfg["warmup_ratio"]),
        max_grad_norm=float(mcfg["max_grad_norm"]),
        seed=seed,
        class_weights=task_class_weights,
        balance_info=balance_info,
    )

    print(f"[train] TEST rezultati ({task}):")
    _print_metrics_brief(task, summary["test_metrics"])
    return summary


def run_all_tasks(
    config: dict[str, Any],
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Pokreni trening za sentiment, sarcasm i multitask; uporedi rezultate."""
    results = {}
    for task in ("sentiment", "sarcasm", "multitask"):
        results[task] = run_training(config, task=task, **kwargs)

    out_root = resolve_path(modeling_defaults(config)["output_dir"])
    if kwargs.get("output_dir") is not None:
        out_root = Path(kwargs["output_dir"])
    comparison = {
        task: {
            "best_epoch": res.get("best_epoch"),
            "val_score": res.get("best_val_selection_score"),
            "test_metrics": res.get("test_metrics"),
        }
        for task, res in results.items()
    }
    path = ensure_dir(out_root) / "comparison.json"
    path.write_text(json.dumps(_to_jsonable(comparison), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] poređenje sačuvano → {path}")
    return results


def load_model_for_eval(
    task_dir: Path,
    task: str,
    model_name: str,
    device: torch.device,
    multitask_cfg: dict[str, Any] | None = None,
) -> torch.nn.Module:
    """Učitaj best.pt checkpoint za evaluaciju / inferencu."""
    ckpt_path = task_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Nema checkpointa: {ckpt_path}")

    if task == "multitask":
        mt = multitask_cfg or {}
        model = MultiTaskModel(
            model_name,
            sentiment_weight=float(mt.get("sentiment_loss_weight", 1.0)),
            sarcasm_weight=float(mt.get("sarcasm_loss_weight", 1.0)),
            dropout=float(mt.get("dropout", 0.1)),
        )
    else:
        model = build_single_task_model(model_name, task)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def run_evaluation(
    config: dict[str, Any],
    task: str,
    split: str = "test",
    splits_dir: Path | None = None,
    model_dir: Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluiraj sačuvani model na datom splitu; upisi ``{split}_metrics.json``."""
    mcfg = modeling_defaults(config)
    device = resolve_device(device_name or mcfg.get("device"))
    splits_path = splits_dir or resolve_path(
        config.get("paths", {}).get("splits_dir", "data/processed/splits")
    )
    task_dir = model_dir or (resolve_path(mcfg["output_dir"]) / task)

    meta_path = task_dir / "run_meta.json"
    model_name = mcfg["model_name"]
    max_length = int(mcfg["max_length"])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = meta.get("model_name", model_name)
        max_length = int(meta.get("max_length", max_length))

    splits = load_splits(splits_path)
    if split not in splits:
        raise ValueError(f"Split mora biti train|val|test, dobijeno: {split}")

    tokenizer = AutoTokenizer.from_pretrained(task_dir if (task_dir / "tokenizer_config.json").exists() else model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = make_loader(
        splits[split],
        tokenizer,
        task=task,
        batch_size=int(mcfg["eval_batch_size"]),
        max_length=max_length,
        shuffle=False,
    )
    model = load_model_for_eval(
        task_dir, task, model_name, device, multitask_cfg=mcfg["multitask"]
    )

    if task == "multitask":
        metrics = evaluate_multitask(model, loader, device)
    else:
        metrics = evaluate_single_task(model, loader, device, task=task)

    out_path = task_dir / f"{split}_metrics.json"
    out_path.write_text(json.dumps(_to_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] {task} / {split}:")
    _print_metrics_brief(task, metrics)
    print(f"[eval] sačuvano → {out_path}")
    return metrics
