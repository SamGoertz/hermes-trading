"""Macro data adapter."""


async def fetch() -> dict:
    """Fetch macro data."""
    return {
        "schema_version": "1.0",
        "fed_rate": None,
        "inflation": None,
        "market_sentiment": None,
    }
