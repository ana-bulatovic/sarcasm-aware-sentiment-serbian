#!/usr/bin/env python3
"""CLI za baseline pretprocesiranje teksta (TF-IDF / klasični ML).

Razlikuje se od ``preprocess.py``: tamo je pipeline raw→interim
(``src.preprocessing.pipeline``); ovde se tekst čisti za baseline modele
preko ``src.preprocessing.baseline`` (URL, Unicode, opciono latinica/lema).

BERTić / transformer modeli NE koriste ovaj modul.

Primeri:
  python scripts/preprocessing/text_preprocessing.py --text "Види https://x.com тест 😀"
  python scripts/preprocessing/text_preprocessing.py --csv data/processed/dataset/dataset.csv \\
      --cyrillic-to-latin --lowercase --out data/processed/scratch/baseline_texts.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.preprocessing.baseline import (
    clean_text,
    clean_text_from_config,
    lemmatize_text,
    normalize_script,
)

__all__ = [
    "clean_text",
    "normalize_script",
    "lemmatize_text",
    "clean_text_from_config",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """Sastavi argparse za baseline čišćenje (jedan tekst ili CSV)."""
    p = argparse.ArgumentParser(
        description=(
            "Baseline pretprocesiranje teksta (NIJE za BERTić). "
            "Za klasične ML modele."
        )
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument(
        "--text",
        default=None,
        help="Jedan tekst za čišćenje (ispis na stdout).",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="Ulazni CSV sa kolonom teksta.",
    )
    p.add_argument(
        "--text-column",
        default="text",
        help="Ime kolone sa tekstom (podrazumevano: text).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Izlazni CSV (dodaje kolonu text_baseline).",
    )
    p.add_argument(
        "--use-config",
        action="store_true",
        help="Koristi baseline_preprocessing iz config.yaml umesto CLI flagova.",
    )
    # Opcije (podrazumevano kao u clean_text)
    p.add_argument("--remove-urls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--collapse-whitespace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--normalize-unicode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--remove-emojis",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument(
        "--cyrillic-to-latin",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--lemmatize", action=argparse.BooleanOptionalAction, default=False)
    return p


def main() -> None:
    """CLI ulazna tačka; poziva ``src.preprocessing.baseline`` (clean_text*).

    Čisti jedan ``--text`` ili CSV kolonu i upisuje ``text_baseline``.
    """
    configure_utf8_stdio()
    args = _build_parser().parse_args()
    config = load_config(args.config)

    if args.use_config:
        cfg = config.get("baseline_preprocessing", {})

        def _apply(t: str) -> str:
            """Primeni ``clean_text_from_config`` sa opcijama iz YAML-a."""
            return clean_text_from_config(t, cfg)

    else:

        def _apply(t: str) -> str:
            """Primeni ``clean_text`` sa CLI flagovima."""
            return clean_text(
                t,
                remove_urls=args.remove_urls,
                collapse_whitespace=args.collapse_whitespace,
                normalize_unicode=args.normalize_unicode,
                remove_emojis=args.remove_emojis,
                cyrillic_to_latin=args.cyrillic_to_latin,
                lowercase=args.lowercase,
                lemmatize=args.lemmatize,
            )

    if args.text is not None:
        print(_apply(args.text))
        return

    if args.csv is None:
        raise SystemExit("Navedi --text ili --csv.")

    import pandas as pd

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV nije pronađen: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if args.text_column not in df.columns:
        raise SystemExit(
            f"Kolona {args.text_column!r} ne postoji. Dostupno: {list(df.columns)}"
        )

    df["text_baseline"] = df[args.text_column].map(
        lambda x: _apply("" if pd.isna(x) else str(x))
    )

    out_path = Path(args.out) if args.out else csv_path.with_name(
        csv_path.stem + "_baseline.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[baseline-preprocess] {len(df)} redova → {out_path}")


if __name__ == "__main__":
    main()
