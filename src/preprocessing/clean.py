"""Čišćenje teksta uz očuvanje interpunkcije i pisma (latinica/ćirilica)."""

from __future__ import annotations

import html
import re
from typing import Any

from bs4 import BeautifulSoup

# Emoji opsezi (osnovni + suplementarni); raw string zbog \U sekvenci
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)

_URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\[\]{}\"'()]+|"
    r"\b[a-z0-9.-]+\.(?:com|rs|org|net|edu|gov|info|me|io)\b(?:/[^\s]*)?",
    flags=re.IGNORECASE,
)

_MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")


def strip_html(text: str) -> str:
    if "<" not in text and "&" not in text:
        return text
    # Prvo dekoduj HTML entitete, zatim ukloni tagove
    unescaped = html.unescape(text)
    soup = BeautifulSoup(unescaped, "html.parser")
    return soup.get_text(separator=" ")


def remove_emojis(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def replace_urls(text: str, replacement: str = "[URL]") -> str:
    return _URL_RE.sub(replacement, text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Prelomi unutar teksta -> razmak (CSV red mora ostati u jednoj liniji)
    text = text.replace("\n", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_text(text: str, cfg: dict[str, Any] | None = None) -> str:
    """Primeni preprocesiranje prema konfiguraciji.

    Ne uklanja interpunkciju, stop reči ni sarkastične izraze.
    Ne radi transliteraciju latinica ↔ ćirilica.
    """
    cfg = cfg or {}
    if text is None:
        return ""

    out = str(text)

    if cfg.get("strip_html", True):
        out = strip_html(out)

    if cfg.get("remove_emojis", True):
        out = remove_emojis(out)

    url_token = cfg.get("replace_urls_with", "[URL]")
    if url_token is not None:
        out = replace_urls(out, replacement=str(url_token))

    # Eksplicitno: ne uklanjati interpunkciju / stopwords / transliteraciju
    if cfg.get("remove_punctuation", False):
        raise ValueError(
            "remove_punctuation=true nije dozvoljeno u ovom pipeline-u "
            "(interpunkcija je važna za sarkazam)."
        )
    if cfg.get("remove_stopwords", False):
        raise ValueError("remove_stopwords=true nije dozvoljeno u ovom pipeline-u.")
    if cfg.get("transliterate", False):
        raise ValueError(
            "transliterate=true nije dozvoljeno — latinica i ćirilica se čuvaju odvojeno."
        )

    if cfg.get("normalize_whitespace", True):
        out = normalize_whitespace(out)
    else:
        out = out.strip()

    return out
