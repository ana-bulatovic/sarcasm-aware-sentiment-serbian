#!/usr/bin/env python3
"""CLI: grafikoni i tabele o datasetu za master rad."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config, resolve_path
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.plots import generate_dataset_figures
from src.dataset.statistics import compute_dataset_statistics, print_statistics


def main() -> None:
    """Napravi PNG slike i CSV tabele iz ``dataset.csv``."""
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Generiše grafikone i tabele o datasetu za izveštaj / master rad."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--csv",
        default=None,
        help="Putanja do dataset CSV-a (podrazumevano paths.dataset_csv).",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/dataset_statistike",
        help="Izlazni folder (figures/ + tabele/ + sazetak.json).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    csv_path = resolve_path(args.csv) if args.csv else resolve_path(config["paths"]["dataset_csv"])
    out_dir = resolve_path(args.out_dir)

    stats = compute_dataset_statistics(config=config, csv_path=csv_path)
    print_statistics(stats)

    summary = generate_dataset_figures(csv_path=csv_path, out_dir=out_dir)
    print(f"\n[plots] Ukupno: {summary['n_ukupno']}  anotirano: {summary['n_anotirano']}")
    print(f"[plots] Slike → {summary['figures_dir']}")
    print(f"[plots] Tabele → {summary['tables_dir']}")
    for name in summary["figures"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
