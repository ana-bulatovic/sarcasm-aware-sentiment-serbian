"""Kolektor YouTube javnih komentara preko zvaničnog YouTube Data API v3.

Zahteva YOUTUBE_API_KEY. Ne scrapuje HTML, ne zaobilazi CAPTCHA/auth.
Čuva samo tekst komentara i ne-PII metapodatke (npr. video_id).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from src.collection.base import BaseCollector
from src.common.config import resolve_path
from src.common.schema import RawRecord
from src.common.source_utils import youtube_watch_url

API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


class YouTubeCollector(BaseCollector):
    """Kolektor javnih YouTube komentara; source u RawRecord je watch URL."""

    source_name = "youtube"

    @staticmethod
    def _normalize_video_id(value: str) -> str:
        """Izvuci video ID iz URL-a (v= / youtu.be/) ili vrati sirovi ID."""
        line = (value or "").strip()
        if not line or line.startswith("#"):
            return ""
        if "v=" in line:
            line = line.split("v=", 1)[1].split("&", 1)[0]
        elif "youtu.be/" in line:
            line = line.split("youtu.be/", 1)[1].split("?", 1)[0]
        elif "/shorts/" in line:
            line = line.split("/shorts/", 1)[1]
        elif "/embed/" in line:
            line = line.split("/embed/", 1)[1]
        line = line.split("?", 1)[0].split("&", 1)[0].split("/", 1)[0]
        return line.strip()

    def _load_video_ids(self) -> list[str]:
        """Učitaj jedinstvene video ID-eve iz config-a i video_ids_file."""
        ids: list[str] = list(self.source_cfg.get("video_ids") or [])
        ids_file = self.source_cfg.get("video_ids_file")
        if ids_file:
            path = resolve_path(ids_file)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    vid = self._normalize_video_id(line)
                    if vid:
                        ids.append(vid)
        seen: set[str] = set()
        unique: list[str] = []
        for vid in ids:
            if vid not in seen:
                seen.add(vid)
                unique.append(vid)
        return unique

    def _fetch_comments_for_video(
        self,
        api_key: str,
        video_id: str,
        max_comments: int,
        include_replies: bool,
        timeout: int,
    ) -> list[RawRecord]:
        """Pozovi commentThreads API i vrati do max_comments RawRecord-a."""
        records: list[RawRecord] = []
        page_token: str | None = None
        part = "snippet,replies" if include_replies else "snippet"

        while len(records) < max_comments:
            params: dict[str, Any] = {
                "part": part,
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(records)),
                "textFormat": "plainText",
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token

            response = requests.get(API_URL, params=params, timeout=timeout)
            if response.status_code == 403:
                raise RuntimeError(
                    "YouTube API 403 — proverite API key, kvotu i da li je "
                    "YouTube Data API v3 omogućen. Odgovor: "
                    f"{response.text[:300]}"
                )
            response.raise_for_status()
            payload = response.json()

            for item in payload.get("items", []):
                top = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = (top.get("textDisplay") or top.get("textOriginal") or "").strip()
                comment_id = item.get("snippet", {}).get("topLevelComment", {}).get("id")
                watch_url = youtube_watch_url(video_id)
                if text:
                    records.append(
                        RawRecord(
                            source=watch_url,
                            text=text,
                            source_item_id=comment_id,
                            metadata={"video_id": video_id, "platform": "youtube"},
                        )
                    )

                if include_replies:
                    for reply in item.get("replies", {}).get("comments", []):
                        rsn = reply.get("snippet", {})
                        rtext = (rsn.get("textDisplay") or rsn.get("textOriginal") or "").strip()
                        if not rtext:
                            continue
                        records.append(
                            RawRecord(
                                source=watch_url,
                                text=rtext,
                                source_item_id=reply.get("id"),
                                metadata={
                                    "video_id": video_id,
                                    "platform": "youtube",
                                    "is_reply": True,
                                },
                            )
                        )

                if len(records) >= max_comments:
                    break

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        return records[:max_comments]

    def collect_specific_videos(
        self,
        video_ids: list[str],
        max_records: int,
    ) -> list[RawRecord]:
        """Prikupi komentare samo za date video ID-eve / URL-ove (YOUTUBE_API_KEY)."""
        api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
        if not api_key:
            print(
                "[youtube] YOUTUBE_API_KEY nije postavljen. "
                "Preskacem kolekciju. Kopirajte .env.example -> .env i unesite kljuc."
            )
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for raw in video_ids:
            vid = self._normalize_video_id(raw)
            if vid and vid not in seen:
                seen.add(vid)
                normalized.append(vid)
        if not normalized:
            return []

        per_video = int(self.source_cfg.get("max_comments_per_video", 100))
        include_replies = bool(self.source_cfg.get("include_replies", False))
        timeout = int(self.source_cfg.get("request_timeout_seconds", 30))

        all_records: list[RawRecord] = []
        for video_id in normalized:
            if len(all_records) >= max_records:
                break
            remaining = max_records - len(all_records)
            try:
                batch = self._fetch_comments_for_video(
                    api_key=api_key,
                    video_id=video_id,
                    max_comments=min(per_video, remaining),
                    include_replies=include_replies,
                    timeout=timeout,
                )
            except Exception as exc:
                print(f"[youtube] Greska za video {video_id}: {exc}")
                continue
            all_records.extend(batch)
            print(f"[youtube] {video_id}: +{len(batch)} komentara (ukupno {len(all_records)})")

        return all_records[:max_records]

    def collect(self, max_records: int) -> list[RawRecord]:
        """Prikupi komentare za ID-eve iz config-a / video_ids_file."""
        video_ids = self._load_video_ids()
        if not video_ids:
            print(
                "[youtube] Nema video ID-eva. Dodajte ih u config/sources/youtube_video_ids.txt "
                "ili collection.youtube.video_ids u config.yaml."
            )
            return []
        return self.collect_specific_videos(video_ids, max_records=max_records)
