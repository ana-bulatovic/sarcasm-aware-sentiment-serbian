"""Izgradnja i dopuna finalnog annotation dataseta.

Moduli u ovom paketu:
- build: source CSV / cleaned.jsonl → dataset CSV (+ JSONL)
- statistics: brojevi po izvoru, sentimentu, sarkazmu → stats JSON
- append_common: zajedničko čišćenje/dedup i upis na annotation CSV
- append_youtube: YouTube → poseban youtube_comments.csv (source alias)
- append_twitter: X/Twitter → twitter_comments.csv (odvojeno od COLLECTOR_REGISTRY)
- append_tiktok / append_instagram / append_reddit: polu-ručni unos na annotation
"""

from src.dataset.append_youtube import append_new_youtube_videos, append_youtube_fetch
from src.dataset.build import build_annotation_dataset, build_dataset_from_sources
from src.dataset.statistics import compute_dataset_statistics

__all__ = [
    "append_new_youtube_videos",
    "append_youtube_fetch",
    "build_annotation_dataset",
    "build_dataset_from_sources",
    "compute_dataset_statistics",
]
