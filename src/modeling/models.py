"""Single-task i multitask modeli na Hugging Face encoderu."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from src.modeling.labels import (
    SARCASM_ID2LABEL,
    SARCASM_LABEL2ID,
    SENTIMENT_ID2LABEL,
    SENTIMENT_LABEL2ID,
)


def build_single_task_model(model_name: str, task: str) -> AutoModelForSequenceClassification:
    if task == "sentiment":
        label2id = SENTIMENT_LABEL2ID
        id2label = SENTIMENT_ID2LABEL
    elif task == "sarcasm":
        label2id = SARCASM_LABEL2ID
        id2label = SARCASM_ID2LABEL
    else:
        raise ValueError(f"Single-task očekuje sentiment|sarcasm, dobijeno: {task}")

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
    )


class MultiTaskModel(nn.Module):
    """Zajednički encoder + dve klasifikacione glave."""

    def __init__(
        self,
        model_name: str,
        sentiment_weight: float = 1.0,
        sarcasm_weight: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden, len(SENTIMENT_LABEL2ID))
        self.sarcasm_head = nn.Linear(hidden, len(SARCASM_LABEL2ID))
        self.sentiment_weight = sentiment_weight
        self.sarcasm_weight = sarcasm_weight
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_labels: torch.Tensor | None = None,
        sarcasm_labels: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        sentiment_logits = self.sentiment_head(cls)
        sarcasm_logits = self.sarcasm_head(cls)

        loss = None
        if sentiment_labels is not None and sarcasm_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss_s = ce(sentiment_logits, sentiment_labels)
            loss_c = ce(sarcasm_logits, sarcasm_labels)
            loss = self.sentiment_weight * loss_s + self.sarcasm_weight * loss_c

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,
            "sarcasm_logits": sarcasm_logits,
        }
