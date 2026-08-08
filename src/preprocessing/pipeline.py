"""Pipeline: raw → interim (očišćeno + filtrirano + deduplikovano)."""

from __future__ import annotations

from typing import Any

from src.collection.base import text_fingerprint
from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import load_jsonl, save_jsonl
from src.common.language import is_likely_serbian
from src.preprocessing.clean import preprocess_text
from src.preprocessing.deduplicate import deduplicate_records


def run_preprocessing(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Pipeline raw → interim: ``preprocess_text``, jezik, dužina, dedup.

    Koristi lagano čišćenje (dataset / BERTić), ne baseline ``clean_text``.
    Vraća listu zapisa i piše ``interim/cleaned.jsonl``.
    """
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    interim_dir = ensure_dir(resolve_path(config["paths"]["interim_dir"]))
    prep_cfg = config.get("preprocessing", {})
    lang_cfg = config.get("language", {})
    min_len = int(lang_cfg.get("min_text_length", 15))
    max_len = int(lang_cfg.get("max_text_length", 2000))

    raw_files = sorted(raw_dir.glob("*/raw.jsonl"))
    cleaned: list[dict[str, Any]] = []

    for path in raw_files:
        for rec in load_jsonl(path):
            text = preprocess_text(rec.get("text", ""), prep_cfg)
            if len(text) < min_len or len(text) > max_len:
                continue
            if not is_likely_serbian(text, lang_cfg):
                continue
            cleaned.append(
                {
                    "source": rec.get("source", path.parent.name),
                    "text": text,
                    "source_item_id": rec.get("source_item_id"),
                    "sentiment": rec.get("sentiment") or "",
                    "sarcasm": rec.get("sarcasm") or "",
                    "metadata": rec.get("metadata") or {},
                    "text_fp": text_fingerprint(text),
                }
            )

    if prep_cfg.get("deduplicate", True):
        before = len(cleaned)
        cleaned = deduplicate_records(cleaned, text_key="text")
        print(f"[preprocess] Deduplikacija: {before} → {len(cleaned)}")

    out_path = interim_dir / "cleaned.jsonl"
    save_jsonl(cleaned, out_path)
    print(f"[preprocess] Sačuvano {len(cleaned)} zapisa → {out_path}")
    return cleaned
