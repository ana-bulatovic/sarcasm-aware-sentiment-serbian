"""Kolektor za SentiComments.SR (CC BY-NC-SA 4.0).

Preuzima korpus sa zvaničnog GitHub repozitorijuma.
Originalne sentiment/sarkazam labele se mapiraju u positive/neutral/negative
i yes/no (vidi src.common.label_mapping).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from src.collection.base import BaseCollector
from src.common.config import ensure_dir, resolve_path
from src.common.label_mapping import map_senticomments_label
from src.common.schema import RawRecord

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/{repo}/master/{filename}"


class SentiCommentsSRCollector(BaseCollector):
    """Preuzima i parsira SentiComments.SR; source = 'senticomments_sr'."""

    source_name = "senticomments_sr"

    def __init__(self, config: dict[str, Any]):
        """Postavi i external_dir (data/external/senticomments_sr)."""
        super().__init__(config)
        external = resolve_path(config["paths"]["external_dir"]) / self.source_name
        self.external_dir = ensure_dir(external)

    def _download_file(self, filename: str) -> Path:
        """Vrati lokalni fajl; ako nedostaje, preuzmi sa GitHub raw (ako je dozvoljeno)."""
        dest = self.external_dir / filename
        if dest.exists():
            return dest

        if not self.source_cfg.get("download_if_missing", True):
            raise FileNotFoundError(
                f"Fajl {dest} ne postoji, a download_if_missing=false. "
                "Preuzmite SentiComments.SR ručno u data/external/senticomments_sr/."
            )

        repo = self.source_cfg.get("github_repo", "vukbatanovic/SentiComments.SR")
        url = GITHUB_RAW_BASE.format(repo=repo, filename=filename)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest

    def _parse_main_corpus(self, path: Path) -> list[RawRecord]:
        """Parsira TSV: sentiment_label \\t comment_id \\t comment_text → RawRecord."""
        use_labels = not bool(self.source_cfg.get("discard_original_labels", False))
        records: list[RawRecord] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                original_label = parts[0].strip()
                comment_id = parts[1]
                text = "\t".join(parts[2:]).strip()
                if not text:
                    continue

                sentiment, sarcasm = ("", "")
                if use_labels:
                    sentiment, sarcasm = map_senticomments_label(original_label)

                records.append(
                    RawRecord(
                        source=self.source_name,
                        text=text,
                        source_item_id=comment_id.strip() or None,
                        sentiment=sentiment,
                        sarcasm=sarcasm,
                        metadata={
                            "corpus_file": path.name,
                            "license": "CC BY-NC-SA 4.0",
                            "attribution": "Batanović et al., SentiComments.SR",
                            "original_label": original_label,
                        },
                    )
                )
        return records

    def collect(self, max_records: int) -> list[RawRecord]:
        """Preuzmi/parsiraš collection.senticomments_sr.files do max_records."""
        files = self.source_cfg.get("files", ["SentiComments.SR.orig.txt"])
        all_records: list[RawRecord] = []
        for filename in files:
            path = self._download_file(filename)
            all_records.extend(self._parse_main_corpus(path))
            if len(all_records) >= max_records:
                break
        return all_records[:max_records]
