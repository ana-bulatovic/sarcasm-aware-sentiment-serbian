"""Šema zapisa dataseta i dozvoljene vrednosti labela.

Kolone finalnog CSV-a: ``id``, ``source``, ``text``, ``tip``, ``sentiment``, ``sarcasm``.

Napomena o ``source``:
  - u raw/pipeline zapisu često stoji **pun URL** (npr. YouTube watch link);
  - u posebnim ``*_comments.csv`` fajlovima često stoji **alias** platforme
    (``youtube``, ``twitter``…).
  Za mapiranje na platformu koristi ``src.common.source_utils.platform_from_source``.

``tip`` = domen sadržaja (filmovi, politika…), **ne** tip fajla.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FINAL_COLUMNS = ["id", "source", "text", "tip", "sentiment", "sarcasm"]

# Sentiment: 1 = positive, 0 = neutral, -1 = negative (čuvaju se kao string u CSV)
SENTIMENT_VALUES = ("1", "0", "-1")
# Sarkazam: 1 = da, 0 = ne
SARCASM_VALUES = ("1", "0")

# Dozvoljene vrednosti tipa sadržaja / domena (subject)
TIP_VALUES = ("filmovi", "serije", "politika", "sport", "ostalo", "reddit")

# Dozvoljene vrednosti izvora (lako proširivo)
KNOWN_SOURCES = (
    "youtube",
    "twitter",
    "tiktok",
    "instagram",
    "reddit",
    "reviews",
)


@dataclass
class RawRecord:
    """Sirovi zapis pre preprocesiranja (bez PII).

    ``source`` može biti URL ili alias platforme — vidi module docstring.
    """

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
        """Pretvori dataclass u običan dict (za JSONL/CSV)."""
        return asdict(self)


@dataclass
class DatasetRecord:
    """Finalni zapis za anotaciju / trening (kolone = ``FINAL_COLUMNS``)."""

    id: str
    source: str
    text: str
    tip: str = ""
    sentiment: str = ""
    sarcasm: str = ""

    def to_dict(self) -> dict[str, str]:
        """Vrati red spreman za upis u annotation/dataset CSV."""
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "tip": self.tip,
            "sentiment": self.sentiment,
            "sarcasm": self.sarcasm,
        }
