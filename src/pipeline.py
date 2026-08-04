"""End-to-end: kolekcija → preprocesiranje → annotation dataset."""

from __future__ import annotations

from typing import Any

from src.collection.run_collection import run_collection
from src.dataset.build import build_annotation_dataset
from src.dataset.statistics import compute_dataset_statistics, print_statistics
from src.preprocessing.pipeline import run_preprocessing


def run_full_pipeline(config: dict[str, Any], sources: list[str] | None = None) -> None:
    run_collection(config, sources=sources)
    run_preprocessing(config)
    build_annotation_dataset(config)
    stats = compute_dataset_statistics(config)
    print_statistics(stats)
