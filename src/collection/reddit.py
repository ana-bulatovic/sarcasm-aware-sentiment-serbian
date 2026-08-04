"""Učitavač Reddit podataka iz odobrenog ekspora.

Važno (ToS / politika Reddita):
Za akademsko istraživanje Reddit zahteva program Reddit for Researchers (RFR).
Korišćenje standardnog Data API-ja ili scrapinga za research krši njihove politike.

Ovaj modul:
- NE implementira scraping
- NE zaobilazi autentifikaciju / CAPTCHA / rate limite
- Učitava JSONL/CSV eksporte koje istraživač dobije formalnim, odobrenim putem
- Čuva samo tekst; ne očekuje niti čuva username / email / PII

Dokumentacija:
https://support.reddithelp.com/hc/en-us/articles/49381918834964-Reddit-for-Researchers-Program
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.collection.base import BaseCollector
from src.common.config import resolve_path
from src.common.schema import RawRecord

# Polja koja se NIKADA ne čuvaju ako se pojave u eksportu
_PII_KEYS = {
    "author",
    "username",
    "user",
    "user_name",
    "author_fullname",
    "author_fullname_ss",
    "email",
    "mail",
    "profile",
    "avatar",
}


class RedditExportCollector(BaseCollector):
    source_name = "reddit"

    def collect(self, max_records: int) -> list[RawRecord]:
        export_path = resolve_path(
            self.source_cfg.get("export_path", "data/external/reddit/export.jsonl")
        )
        if not export_path.exists():
            print(
                "[reddit] Eksport nije pronađen: "
                f"{export_path}\n"
                "  Za akademsko istraživanje prijavite se na Reddit for Researchers,\n"
                "  zatim sačuvajte odobreni eksport (samo tekstualna polja) na tu putanju.\n"
                "  Scraping / neovlašćeni API pristup nije podržan."
            )
            return []

        text_col = self.source_cfg.get("text_column", "text")
        id_col = self.source_cfg.get("id_column", "id")

        rows = self._load_rows(export_path)
        records: list[RawRecord] = []
        for row in rows:
            if len(records) >= max_records:
                break
            text = str(row.get(text_col) or row.get("body") or row.get("selftext") or "").strip()
            if not text:
                continue
            item_id = row.get(id_col)
            # Ne prosleđuj PII u metadata
            meta = {
                k: v
                for k, v in row.items()
                if k not in _PII_KEYS
                and k not in {text_col, "body", "selftext", id_col}
                and k.lower() not in _PII_KEYS
            }
            # Zadrži samo jednostavne ne-PII tipove
            safe_meta = {
                k: v
                for k, v in meta.items()
                if isinstance(v, (str, int, float, bool)) and k in {"subreddit", "created_utc", "score"}
            }
            records.append(
                RawRecord(
                    source=self.source_name,
                    text=text,
                    source_item_id=str(item_id) if item_id is not None else None,
                    metadata=safe_meta,
                )
            )
        return records[:max_records]

    def _load_rows(self, path: Path) -> list[dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "data" in data:
                return list(data["data"])
            raise ValueError(f"Nepoznat JSON oblik: {path}")
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, dtype=str).fillna("")
            return df.to_dict(orient="records")
        raise ValueError(f"Nepodržan format ekspora: {suffix}")
