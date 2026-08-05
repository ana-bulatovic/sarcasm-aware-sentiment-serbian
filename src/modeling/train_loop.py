"""Trening / evaluacija petlja (CPU ili CUDA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.modeling.labels import (
    SARCASM_ID2LABEL,
    SENTIMENT_ID2LABEL,
)
from src.modeling.metrics import pack_multitask_metrics, pack_single_task_metrics


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    return str(obj)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@torch.no_grad()
def evaluate_single_task(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    sarcasm_raw: list[str] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if outputs.loss is not None:
            losses.append(float(outputs.loss.item()))
        preds = outputs.logits.argmax(dim=-1)
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(preds.tolist())
        sarcasm_raw.extend(batch["sarcasm_raw"])

    label_names = (
        [SENTIMENT_ID2LABEL[i] for i in range(len(SENTIMENT_ID2LABEL))]
        if task == "sentiment"
        else [SARCASM_ID2LABEL[i] for i in range(len(SARCASM_ID2LABEL))]
    )
    metrics = pack_single_task_metrics(y_true, y_pred, sarcasm_raw, label_names)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


@torch.no_grad()
def evaluate_multitask(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    sent_true: list[int] = []
    sent_pred: list[int] = []
    sarc_true: list[int] = []
    sarc_pred: list[int] = []
    sarcasm_raw: list[str] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            sentiment_labels=batch["sentiment_labels"],
            sarcasm_labels=batch["sarcasm_labels"],
        )
        if outputs["loss"] is not None:
            losses.append(float(outputs["loss"].item()))
        sent_pred.extend(outputs["sentiment_logits"].argmax(dim=-1).tolist())
        sarc_pred.extend(outputs["sarcasm_logits"].argmax(dim=-1).tolist())
        sent_true.extend(batch["sentiment_labels"].tolist())
        sarc_true.extend(batch["sarcasm_labels"].tolist())
        sarcasm_raw.extend(batch["sarcasm_raw"])

    metrics = pack_multitask_metrics(
        sent_true,
        sent_pred,
        sarc_true,
        sarc_pred,
        sarcasm_raw,
        [SENTIMENT_ID2LABEL[i] for i in range(len(SENTIMENT_ID2LABEL))],
        [SARCASM_ID2LABEL[i] for i in range(len(SARCASM_ID2LABEL))],
    )
    metrics["loss"] = float(sum(losses) / max(len(losses), 1))
    return metrics


def _selection_score(metrics: dict[str, Any], task: str) -> float:
    """Veći je bolji — koristi macro-F1 (za multitask: prosek sentiment+sarcasm)."""
    if task == "multitask":
        s = metrics["sentiment"]["overall"]["macro_f1"]
        c = metrics["sarcasm"]["overall"]["macro_f1"]
        return 0.5 * (s + c)
    return float(metrics["overall"]["macro_f1"])


def train_one_task(
    *,
    task: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    seed: int,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    log = log_fn or print
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, len(train_loader) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_score = float("-inf")
    best_path = output_dir / "best.pt"
    history: list[dict[str, Any]] = []

    evaluate = evaluate_multitask if task == "multitask" else evaluate_single_task

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"{task} epoch {epoch}/{num_epochs}", leave=False)
        for batch in pbar:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if task == "multitask":
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sentiment_labels=batch["sentiment_labels"],
                    sarcasm_labels=batch["sarcasm_labels"],
                )
                loss = outputs["loss"]
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{running_loss / n_batches:.4f}")

        if task == "multitask":
            val_metrics = evaluate(model, val_loader, device)
        else:
            val_metrics = evaluate(model, val_loader, device, task=task)

        score = _selection_score(val_metrics, task)
        train_loss = running_loss / max(n_batches, 1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("loss"),
            "val_selection_score": score,
            "val_metrics": val_metrics,
        }
        history.append(row)
        log(
            f"[{task}] epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_score={score:.4f} val_loss={val_metrics.get('loss', float('nan')):.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "task": task,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "seed": seed,
                },
                best_path,
            )
            log(f"[{task}] novi best checkpoint → {best_path}")

    # Učitaj best i evaluiraj test
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if task == "multitask":
        test_metrics = evaluate(model, test_loader, device)
        val_metrics = evaluate(model, val_loader, device)
    else:
        test_metrics = evaluate(model, test_loader, device, task=task)
        val_metrics = evaluate(model, val_loader, device, task=task)

    summary = {
        "task": task,
        "best_epoch": checkpoint.get("epoch"),
        "best_val_selection_score": best_score,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint": str(best_path),
    }
    _save_json(output_dir / "metrics.json", summary)
    _save_json(output_dir / "test_metrics.json", test_metrics)
    log(f"[{task}] test metrics sačuvane → {output_dir / 'test_metrics.json'}")
    return summary
