"""Price data adapter."""
import os
from typing import Optional

import ccxt.async_support as ccxt


class SchemaError(Exception):
    pass


async def fetch() -> dict:
    """Fetch price data via CCXT."""
    exchange = ccxt.binance()
    try:
        ticker = await exchange.fetch_ticker("BTC/USDT")
        result = {
            "schema_version": "1.0",
            "price": ticker["last"],
            "bid": ticker["bid"],
            "ask": ticker["ask"],
            "volume": ticker["quoteVolume"],
        }
        return result
    finally:
        await exchange.close()
