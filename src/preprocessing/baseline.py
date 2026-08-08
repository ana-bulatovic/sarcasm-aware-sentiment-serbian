"""Pretprocesiranje teksta za klasične ML baseline modele.

Ovaj modul NIJE namenjen BERTić / transformer pipeline-u.
Transformer modeli očekuju prirodan tekst (sa ćirilicom/latinicom kako jeste);
tokenizacija radi na raw (ili samo lagano očišćenom) tekstu iz dataseta.

Koristi se isključivo za baseline-e tipa TF-IDF + LR/SVM/NB.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.preprocessing.clean import remove_emojis as _strip_emojis

# Uklanjanje URL-ova (za baseline — potpuno, ne zamena tokenom)
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)[^\s<>\[\]{}\"'()]+|"
    r"\b[a-z0-9.-]+\.(?:com|rs|org|net|edu|gov|info|me|io)\b(?:/[^\s]*)?",
    flags=re.IGNORECASE,
)

_MULTI_SPACE_RE = re.compile(r"\s+")

# Srpska ćirilica → latinica (digrafi prvo)
_CYR_TO_LAT: list[tuple[str, str]] = [
    ("Љ", "Lj"),
    ("љ", "lj"),
    ("Њ", "Nj"),
    ("њ", "nj"),
    ("Џ", "Dž"),
    ("џ", "dž"),
    ("А", "A"),
    ("а", "a"),
    ("Б", "B"),
    ("б", "b"),
    ("В", "V"),
    ("в", "v"),
    ("Г", "G"),
    ("г", "g"),
    ("Д", "D"),
    ("д", "d"),
    ("Ђ", "Đ"),
    ("ђ", "đ"),
    ("Е", "E"),
    ("е", "e"),
    ("Ж", "Ž"),
    ("ж", "ž"),
    ("З", "Z"),
    ("з", "z"),
    ("И", "I"),
    ("и", "i"),
    ("Ј", "J"),
    ("ј", "j"),
    ("К", "K"),
    ("к", "k"),
    ("Л", "L"),
    ("л", "l"),
    ("М", "M"),
    ("м", "m"),
    ("Н", "N"),
    ("н", "n"),
    ("О", "O"),
    ("о", "o"),
    ("П", "P"),
    ("п", "p"),
    ("Р", "R"),
    ("р", "r"),
    ("С", "S"),
    ("с", "s"),
    ("Т", "T"),
    ("т", "t"),
    ("Ћ", "Ć"),
    ("ћ", "ć"),
    ("У", "U"),
    ("у", "u"),
    ("Ф", "F"),
    ("ф", "f"),
    ("Х", "H"),
    ("х", "h"),
    ("Ц", "C"),
    ("ц", "c"),
    ("Ч", "Č"),
    ("ч", "č"),
    ("Ш", "Š"),
    ("ш", "š"),
]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def normalize_script(text: str, *, to: str = "latin") -> str:
    """Normalizuj pismo: ćirilica → latinica (podrazumevano).

    Args:
        text: Ulazni tekst.
        to: Ciljno pismo. Trenutno podržano samo ``\"latin\"``.
    """
    if text is None:
        return ""
    if to != "latin":
        raise ValueError(f"Nepodržano ciljno pismo: {to!r} (dozvoljeno: 'latin').")

    out = str(text)
    for src, dst in _CYR_TO_LAT:
        out = out.replace(src, dst)
    return out


_SIMPLEMA_LANG_ALIASES = {
    "sr": "hbs",
    "hr": "hbs",
    "bs": "hbs",
    "cnr": "hbs",
    "sh": "hbs",
}


def lemmatize_text(text: str, *, lang: str = "hbs") -> str:
    """Lematizuj tokene (zahteva paket ``simplemma``).

    Interpunkcija se zadržava; lematizuju se samo tokeni sa slovima.
    Za srpski/BCMS koristi se kod ``hbs`` (simplemma nema poseban ``sr``);
    aliasi ``sr`` / ``hr`` / ``bs`` se automatski mapiraju na ``hbs``.
    """
    if text is None or not str(text).strip():
        return ""

    try:
        import simplemma
    except ImportError as exc:
        raise ImportError(
            "Lematizacija zahteva paket 'simplemma'. "
            "Instaliraj: pip install simplemma"
        ) from exc

    lang_code = _SIMPLEMA_LANG_ALIASES.get(lang.lower(), lang)

    parts: list[str] = []
    for token in _TOKEN_RE.findall(str(text)):
        if any(ch.isalpha() for ch in token):
            parts.append(simplemma.lemmatize(token, lang=lang_code))
        else:
            parts.append(token)
    return " ".join(parts)


def clean_text(
    text: str,
    *,
    remove_urls: bool = True,
    collapse_whitespace: bool = True,
    normalize_unicode: bool = True,
    remove_emojis: bool = False,
    cyrillic_to_latin: bool = False,
    lowercase: bool = False,
    lemmatize: bool = False,
    lemma_lang: str = "hbs",
) -> str:
    """Agresivnije čišćenje **samo** za TF-IDF baseline-e (ne za BERTić).

    Za dataset / transformere koristi ``clean.preprocess_text``.
    Redosled: NFC → URL → emoji → ćirilica→latinica → lowercase →
    lematizacija → beline (sve opciono).
    """
    if text is None:
        return ""

    out = str(text)

    if normalize_unicode:
        out = unicodedata.normalize("NFC", out)

    if remove_urls:
        out = _URL_RE.sub(" ", out)

    if remove_emojis:
        out = _strip_emojis(out)

    if cyrillic_to_latin:
        out = normalize_script(out, to="latin")

    if lowercase:
        out = out.lower()

    if lemmatize:
        out = lemmatize_text(out, lang=lemma_lang)

    if collapse_whitespace:
        out = _MULTI_SPACE_RE.sub(" ", out).strip()
    else:
        out = out.strip()

    return out


def clean_text_from_config(text: str, cfg: dict[str, Any] | None = None) -> str:
    """Primeni ``baseline_preprocessing`` iz config.yaml (samo TF-IDF baseline-i)."""
    cfg = cfg or {}
    return clean_text(
        text,
        remove_urls=bool(cfg.get("remove_urls", True)),
        collapse_whitespace=bool(cfg.get("collapse_whitespace", True)),
        normalize_unicode=bool(cfg.get("normalize_unicode", True)),
        remove_emojis=bool(cfg.get("remove_emojis", False)),
        cyrillic_to_latin=bool(cfg.get("cyrillic_to_latin", False)),
        lowercase=bool(cfg.get("lowercase", False)),
        lemmatize=bool(cfg.get("lemmatize", False)),
        lemma_lang=str(cfg.get("lemma_lang", "hbs")),
    )
