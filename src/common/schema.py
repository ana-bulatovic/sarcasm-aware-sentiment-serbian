"""Šema zapisa dataseta i dozvoljene vrednosti labela."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FINAL_COLUMNS = ["id", "source", "text", "tip", "sentiment", "sarcasm"]

# Sentiment: 1 = positive, 0 = neutral, -1 = negative (čuvaju se kao string u CSV)
SENTIMENT_VALUES = ("1", "0", "-1")
# Sarkazam: 1 = da, 0 = ne
SARCASM_VALUES = ("1", "0")

# Dozvoljene vrednosti tipa sadržaja / domena
TIP_VALUES = ("filmovi",)

# Dozvoljene vrednosti izvora (lako proširivo)
KNOWN_SOURCES = (
    "youtube",
    "tiktok",
    "instagram",
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
    # Predpopunjene labele; prazno = rucna anotacija
    sentiment: str = ""
    sarcasm: str = ""
    tip: str = ""
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
    tip: str = ""
    sentiment: str = ""
    sarcasm: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "tip": self.tip,
            "sentiment": self.sentiment,
            "sarcasm": self.sarcasm,
        }
