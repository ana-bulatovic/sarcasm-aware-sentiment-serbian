"""End-to-end orkestracija data-prep pipeline-a.

Redosled koraka:
  1. kolekcija sirovih tekstova (`src.collection`)
  2. čišćenje / dedup → interim (`src.preprocessing`)
  3. annotation / dataset CSV (`src.dataset.build`)
  4. statistike na konzolu
"""

from __future__ import annotations

from typing import Any

from src.collection.run_collection import run_collection
from src.dataset.build import build_annotation_dataset
from src.dataset.statistics import compute_dataset_statistics, print_statistics
from src.preprocessing.pipeline import run_preprocessing


def run_full_pipeline(config: dict[str, Any], sources: list[str] | None = None) -> None:
    """Pokreni ceo data-prep tok: raw → interim → annotation CSV + stats.

    Args:
        config: Učitana `config/config.yaml` mapa.
        sources: Opcioni podskup izvora (npr. ``["youtube"]``); ``None`` = svi
            omogućeni u konfiguraciji.
    """
    run_collection(config, sources=sources)
    run_preprocessing(config)
    build_annotation_dataset(config)
    stats = compute_dataset_statistics(config)
    print_statistics(stats)
