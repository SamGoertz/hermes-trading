"""On-chain data adapter."""
import os


async def fetch() -> dict:
    """Fetch on-chain data (requires Glassnode API key)."""
    api_key = os.getenv("GLASSNODE_API_KEY")
    if not api_key:
        return {
            "schema_version": "1.0",
            "available": False,
            "reason": "GLASSNODE_API_KEY not set",
        }

    return {
        "schema_version": "1.0",
        "available": True,
        "data": None,
    }
