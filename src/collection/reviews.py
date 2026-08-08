"""Generički učitavač javnih recenzija/komentara iz lokalnih fajlova.

Ne scrapuje web sajtove. Korisnik obezbeđuje podatke u skladu sa uslovima
korišćenja izvora (npr. ručno eksportovane javne recenzije).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.collection.base import BaseCollector
from src.common.config import resolve_path
from src.common.schema import RawRecord


class ReviewsCollector(BaseCollector):
    """Učitava lokalne recenzije iz CSV/TSV/JSON(L)/TXT; bez web scrapinga."""

    source_name = "reviews"

    def collect(self, max_records: int) -> list[RawRecord]:
        """Prođi input_paths i vrati do max_records RawRecord-a (source = source_label)."""
        input_paths = self.source_cfg.get("input_paths", ["data/external/reviews/"])
        text_col = self.source_cfg.get("text_column", "text")
        source_label = self.source_cfg.get("source_label", self.source_name)

        files: list[Path] = []
        for raw in input_paths:
            path = resolve_path(raw)
            if path.is_dir():
                for pattern in ("*.csv", "*.tsv", "*.jsonl", "*.json", "*.txt"):
                    files.extend(sorted(path.glob(pattern)))
            elif path.is_file():
                files.append(path)
            else:
                print(f"[reviews] Putanja ne postoji (preskačem): {path}")

        records: list[RawRecord] = []
        for file_path in files:
            if len(records) >= max_records:
                break
            try:
                batch = self._load_file(file_path, text_col, source_label)
            except Exception as exc:
                print(f"[reviews] Greška pri čitanju {file_path}: {exc}")
                continue
            for rec in batch:
                records.append(rec)
                if len(records) >= max_records:
                    break
        if not files:
            print(
                "[reviews] Nema ulaznih fajlova. Stavite CSV/JSONL/TXT u "
                "data/external/reviews/ (kolona 'text' ili jedan tekst po liniji)."
            )
        return records[:max_records]

    def _load_file(self, path: Path, text_col: str, source_label: str) -> list[RawRecord]:
        """Učitaj jedan fajl u RawRecord listu; text_col = ime tekstualne kolone."""
        suffix = path.suffix.lower()
        records: list[RawRecord] = []

        if suffix == ".txt":
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                text = line.strip()
                if text:
                    records.append(
                        RawRecord(
                            source=source_label,
                            text=text,
                            source_item_id=f"{path.stem}-{i}",
                            metadata={"file": path.name},
                        )
                    )
            return records

        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = str(obj.get(text_col) or obj.get("review") or obj.get("comment") or "").strip()
                    if text:
                        records.append(
                            RawRecord(
                                source=source_label,
                                text=text,
                                source_item_id=str(obj.get("id", f"{path.stem}-{i}")),
                                metadata={"file": path.name},
                            )
                        )
            return records

        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("data", [])
            for i, obj in enumerate(items):
                if not isinstance(obj, dict):
                    continue
                text = str(obj.get(text_col) or obj.get("review") or obj.get("comment") or "").strip()
                if text:
                    records.append(
                        RawRecord(
                            source=source_label,
                            text=text,
                            source_item_id=str(obj.get("id", f"{path.stem}-{i}")),
                            metadata={"file": path.name},
                        )
                    )
            return records

        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
            col = text_col if text_col in df.columns else None
            if col is None:
                for candidate in ("text", "review", "comment", "content", "body"):
                    if candidate in df.columns:
                        col = candidate
                        break
            if col is None:
                raise ValueError(f"Nema tekstualne kolone u {path}")
            for i, row in df.iterrows():
                text = str(row[col]).strip()
                if not text:
                    continue
                item_id = row["id"] if "id" in df.columns else f"{path.stem}-{i}"
                records.append(
                    RawRecord(
                        source=source_label,
                        text=text,
                        source_item_id=str(item_id),
                        metadata={"file": path.name},
                    )
                )
            return records

        raise ValueError(f"Nepodržan format: {path}")
