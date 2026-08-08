"""Inferenca nad jednim ili više tekstova."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.common.config import resolve_path
from src.modeling.labels import SARCASM_ID2LABEL, SENTIMENT_ID2LABEL
from src.modeling.runner import load_model_for_eval, modeling_defaults, resolve_device


@torch.no_grad()
def predict_texts(
    texts: list[str],
    *,
    config: dict[str, Any],
    task: str = "sentiment",
    model_dir: Path | None = None,
    device_name: str | None = None,
) -> list[dict[str, Any]]:
    """Predikcije za listu tekstova (labele + verovatnoće po klasama)."""
    if not texts:
        return []

    mcfg = modeling_defaults(config)
    device = resolve_device(device_name or mcfg.get("device"))
    task_dir = model_dir or (resolve_path(mcfg["output_dir"]) / task)

    meta_path = task_dir / "run_meta.json"
    model_name = mcfg["model_name"]
    max_length = int(mcfg["max_length"])
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = meta.get("model_name", model_name)
        max_length = int(meta.get("max_length", max_length))

    tok_src = task_dir if (task_dir / "tokenizer_config.json").exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_for_eval(
        task_dir, task, model_name, device, multitask_cfg=mcfg["multitask"]
    )

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    results: list[dict[str, Any]] = []
    if task == "multitask":
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        sent_prob = F.softmax(outputs["sentiment_logits"], dim=-1)
        sarc_prob = F.softmax(outputs["sarcasm_logits"], dim=-1)
        sent_pred = sent_prob.argmax(dim=-1).tolist()
        sarc_pred = sarc_prob.argmax(dim=-1).tolist()
        for i, text in enumerate(texts):
            results.append(
                {
                    "text": text,
                    "sentiment": SENTIMENT_ID2LABEL[sent_pred[i]],
                    "sentiment_probs": {
                        SENTIMENT_ID2LABEL[j]: float(sent_prob[i, j])
                        for j in range(sent_prob.size(1))
                    },
                    "sarcasm": SARCASM_ID2LABEL[sarc_pred[i]],
                    "sarcasm_probs": {
                        SARCASM_ID2LABEL[j]: float(sarc_prob[i, j])
                        for j in range(sarc_prob.size(1))
                    },
                }
            )
        return results

    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
    )
    probs = F.softmax(outputs.logits, dim=-1)
    preds = probs.argmax(dim=-1).tolist()
    id2label = SENTIMENT_ID2LABEL if task == "sentiment" else SARCASM_ID2LABEL
    key = "sentiment" if task == "sentiment" else "sarcasm"
    for i, text in enumerate(texts):
        results.append(
            {
                "text": text,
                key: id2label[preds[i]],
                f"{key}_probs": {
                    id2label[j]: float(probs[i, j]) for j in range(probs.size(1))
                },
            }
        )
    return results
