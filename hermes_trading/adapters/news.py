"""News data adapter."""
import os


async def fetch() -> dict:
    """Fetch news data (requires NewsAPI key)."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return {
            "schema_version": "1.0",
            "available": False,
            "reason": "NEWS_API_KEY not set",
        }

    return {
        "schema_version": "1.0",
        "available": True,
        "headlines": [],
    }
