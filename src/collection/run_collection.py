"""Orkestracija prikupljanja iz svih omogućenih izvora.

Radi samo sa COLLECTOR_REGISTRY (senticomments_sr, youtube, reddit, reviews).
Twitter/X NIJE u registru — koristi twitter_fetch + append_twitter odvojeno.
"""

from __future__ import annotations

from typing import Any

from src.collection import COLLECTOR_REGISTRY
from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import save_jsonl
from src.common.schema import RawRecord


def _raw_quota(config: dict[str, Any]) -> tuple[int, dict[str, int]]:
    """Izračunaj koliko sirovih uzoraka ciljati (oversample + per-source)."""
    max_total = int(config["dataset"]["max_total_samples"])
    factor = float(config["dataset"].get("raw_oversample_factor", 1.5))
    raw_total_budget = int(max_total * factor)
    per_source = dict(config.get("per_source_limits", {}))

    # Oversample i po izvoru
    raw_per_source = {
        name: int(limit * factor) for name, limit in per_source.items()
    }
    return raw_total_budget, raw_per_source


def run_collection(
    config: dict[str, Any],
    sources: list[str] | None = None,
) -> list[RawRecord]:
    """Pokreni collect_and_save za omogućene izvore iz COLLECTOR_REGISTRY.

    sources: lista imena ili None → collection.enabled_sources.
    Piše po-izvor raw.jsonl i spojeni paths.raw_dir/merged_raw.jsonl.
    Twitter se ovde ne pokreće.
    """
    enabled = sources or config.get("collection", {}).get("enabled_sources", [])
    raw_total_budget, raw_per_source = _raw_quota(config)

    all_records: list[RawRecord] = []
    for name in enabled:
        if name not in COLLECTOR_REGISTRY:
            print(f"[collect] Nepoznat izvor '{name}' - preskacem.")
            continue
        limit = raw_per_source.get(name, raw_total_budget)
        remaining = raw_total_budget - len(all_records)
        if remaining <= 0:
            print(f"[collect] Dostignut raw budzet ({raw_total_budget}).")
            break
        target = min(limit, remaining)
        print(f"[collect] {name}: cilj <= {target} sirovih zapisa")
        collector = COLLECTOR_REGISTRY[name](config)
        batch = collector.collect_and_save(max_records=target)
        print(f"[collect] {name}: sacuvano {len(batch)} -> {collector.raw_path}")
        all_records.extend(batch)

    # Spojeni raw pregled (bez PII)
    merged_path = ensure_dir(resolve_path(config["paths"]["raw_dir"])) / "merged_raw.jsonl"
    save_jsonl([r.to_dict() for r in all_records], merged_path)
    print(f"[collect] Ukupno sirovih: {len(all_records)} -> {merged_path}")
    return all_records
