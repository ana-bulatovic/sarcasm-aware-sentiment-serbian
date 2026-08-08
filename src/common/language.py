"""Heuristička detekcija srpskog / BCMS teksta.

Napomena: langdetect često meša sr/hr/bs. Zato podrazumevano dozvoljavamo
sve tri oznake. Latinica i ćirilica se ne konvertuju.
"""

from __future__ import annotations

import re
from typing import Any

# Osnovni ćirilični opseg (srpska/ruska ćirilica)
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
# Latinica sa srpskim dijakriticima
_SERBIAN_LATIN_MARKERS = re.compile(
    r"[čćšžđČĆŠŽĐ]",
)

try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGDETECT_AVAILABLE = False


def has_cyrillic(text: str) -> bool:
    """True ako tekst sadrži bar jedan ćirilični karakter."""
    return bool(_CYRILLIC_RE.search(text))


def has_serbian_latin_diacritics(text: str) -> bool:
    """True ako tekst sadrži srpske latinice dijakritike (č, ć, š, ž, đ…)."""
    return bool(_SERBIAN_LATIN_MARKERS.search(text))


def detect_language(text: str) -> tuple[str | None, float]:
    """Vrati (jezički_kod, pouzdanost) ili (None, 0.0)."""
    sample = (text or "").strip()
    if len(sample) < 10 or not _LANGDETECT_AVAILABLE:
        return None, 0.0
    try:
        langs = detect_langs(sample)
        if not langs:
            return None, 0.0
        best = langs[0]
        return best.lang, float(best.prob)
    except Exception:
        return None, 0.0


def is_likely_serbian(text: str, language_cfg: dict[str, Any]) -> bool:
    """True ako tekst prolazi jezički filter iz konfiguracije."""
    if not language_cfg.get("enabled", True):
        return True

    allowed = set(language_cfg.get("allowed_codes", ["sr", "hr", "bs"]))
    keep_unknown = bool(language_cfg.get("keep_unknown", True))
    min_conf = float(language_cfg.get("min_confidence", 0.5))

    code, conf = detect_language(text)
    if code is None:
        # Heuristika: ćirilica ili srpski dijakritici → verovatno OK
        if has_cyrillic(text) or has_serbian_latin_diacritics(text):
            return True
        return keep_unknown

    if code in allowed and conf >= min_conf:
        return True

    # Niska pouzdanost → unknown
    if conf < min_conf:
        if has_cyrillic(text) or has_serbian_latin_diacritics(text):
            return True
        return keep_unknown

    return False
