"""Kolektori podataka po izvorima.

Moduli u ovom paketu:
- base: apstraktni BaseCollector i text_fingerprint (16 hex)
- run_collection: orkestracija omogućenih izvora iz COLLECTOR_REGISTRY
- youtube: YouTube Data API v3 komentari
- reddit: učitavanje odobrenog Reddit ekspora (bez scrapinga)
- reviews: lokalni CSV/JSONL/TXT recenzija
- senticomments_sr: SentiComments.SR korpus sa GitHub-a
- twitter_fetch: X/Twitter replies preko twikit (NIJE u COLLECTOR_REGISTRY;
  koristi se odvojeno sa append_twitter u src.dataset)
"""

from src.collection.reddit import RedditExportCollector
from src.collection.reviews import ReviewsCollector
from src.collection.senticomments_sr import SentiCommentsSRCollector
from src.collection.youtube import YouTubeCollector

# Twitter/X nije ovde — vidi twitter_fetch + src.dataset.append_twitter.
COLLECTOR_REGISTRY = {
    "senticomments_sr": SentiCommentsSRCollector,
    "youtube": YouTubeCollector,
    "reddit": RedditExportCollector,
    "reviews": ReviewsCollector,
}

__all__ = [
    "COLLECTOR_REGISTRY",
    "SentiCommentsSRCollector",
    "YouTubeCollector",
    "RedditExportCollector",
    "ReviewsCollector",
]
