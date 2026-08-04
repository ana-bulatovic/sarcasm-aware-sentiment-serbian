"""Kolektori podataka po izvorima."""

from src.collection.reddit import RedditExportCollector
from src.collection.reviews import ReviewsCollector
from src.collection.senticomments_sr import SentiCommentsSRCollector
from src.collection.youtube import YouTubeCollector

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
