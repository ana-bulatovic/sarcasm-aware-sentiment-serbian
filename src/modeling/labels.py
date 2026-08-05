"""Mapiranje labela za sentiment i sarkazam."""

from __future__ import annotations

from src.common.schema import SARCASM_VALUES, SENTIMENT_VALUES

SENTIMENT_LABEL2ID = {label: i for i, label in enumerate(SENTIMENT_VALUES)}
SENTIMENT_ID2LABEL = {i: label for label, i in SENTIMENT_LABEL2ID.items()}

SARCASM_LABEL2ID = {label: i for i, label in enumerate(SARCASM_VALUES)}
SARCASM_ID2LABEL = {i: label for label, i in SARCASM_LABEL2ID.items()}
