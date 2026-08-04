"""Šema zapisa dataseta i dozvoljene vrednosti labela."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FINAL_COLUMNS = ["id", "source", "text", "sentiment", "sarcasm"]

SENTIMENT_VALUES = ("positive", "neutral", "negative")
SARCASM_VALUES = ("yes", "no")

# Dozvoljene vrednosti izvora (lako proširivo)
KNOWN_SOURCES = (
    "youtube",
    "tiktok",
    "reddit",
    "reviews",
)


@dataclass
class RawRecord:
    """Sirovi zapis pre preprocesiranja (bez PII)."""

    source: str
    text: str
    # Opcioni identifikator iz izvora (npr. comment hash), NE username
    source_item_id: str | None = None
    # Predpopunjene labele (npr. iz SentiComments.SR); prazno = rucna anotacija
    sentiment: str = ""
    sarcasm: str = ""
    # Dodatni ne-PII metapodaci (npr. video_id, original_label)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetRecord:
    """Finalni zapis za anotaciju."""

    id: str
    source: str
    text: str
    sentiment: str = ""
    sarcasm: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "sentiment": self.sentiment,
            "sarcasm": self.sarcasm,
        }
