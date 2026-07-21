"""24/7 trading loop."""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from .score import score as score_trades


class TradingLoop:
    def __init__(self, state_dir: Path, asset: str):
        self.state_dir = state_dir
        self.asset = asset
        self.consecutive_failures = 0
        self.circuit_broken = False

    async def _load_yaml(self, path: Path) -> dict:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        return yaml.safe_load(content)

    async def _load_goal(self) -> dict:
        return await self._load_yaml(self.state_dir / "goal.yaml")

    async def _load_strategy(self) -> dict:
        return await self._load_yaml(self.state_dir / "strategy.yaml")

    async def _fetch_price(self) -> Optional[float]:
        """Fetch current stock price with retries."""
        for attempt in range(3):
            try:
                ticker = yf.Ticker(self.asset)
                data = ticker.history(period="1d")
                if not data.empty:
                    price = float(data['Close'].iloc[-1])
                    return price
                else:
                    print(f"No data for {self.asset}")
                    return None
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"Price fetch failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
        return None

    async def _load_trades(self) -> list:
        """Load trades from jsonl."""
        trades_file = self.state_dir / "trades.jsonl"
        if not trades_file.exists():
            return []
        async with aiofiles.open(trades_file, "r") as f:
            content = await f.read()
        return [json.loads(line) for line in content.strip().split("\n") if line]

    async def _append_trade(self, trade: dict):
        """Append trade to trades.jsonl."""
        trades_file = self.state_dir / "trades.jsonl"
        trade["timestamp"] = datetime.utcnow().isoformat()
        async with aiofiles.open(trades_file, "a") as f:
            await f.write(json.dumps(trade) + "\n")

    async def _write_heartbeat(self, status: str, price: Optional[float] = None):
        """Write heartbeat."""
        heartbeat_file = self.state_dir / "heartbeat.json"
        hb = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "price": price,
            "asset": self.asset,
        }
        async with aiofiles.open(heartbeat_file, "w") as f:
            await f.write(json.dumps(hb, indent=2))

    async def _check_entry(self, strategy: dict, price: float, recent_prices: list, goal: dict) -> bool:
        """Evaluate entry signal."""
        if strategy["entry"]["direction"] != "long":
            return False

        if not goal.get("trading_enabled", True):
            return False

        if strategy["entry"]["indicator"] == "rsi":
            if len(recent_prices) < 14:
                return False
            rsi = self._compute_rsi(recent_prices)
            return rsi < strategy["entry"]["threshold"]
        return False

    async def _check_risk_limits(self, goal: dict, trades: list) -> tuple:
        """Check if trading should be halted due to risk limits.
        Returns (allowed, reason)"""
        max_pos = goal.get("max_position_size_usd", 1000.0)
        max_daily_loss = goal.get("max_daily_loss_usd", 10000.0)

        today = pd.Timestamp.utcnow().date()
        today_trades = [t for t in trades if pd.Timestamp(t.get("timestamp", "")).date() == today]

        daily_loss = sum([t.get("pnl", 0) for t in today_trades if t.get("pnl", 0) < 0])

        if abs(daily_loss) > max_daily_loss:
            return False, f"Daily loss limit exceeded: ${abs(daily_loss):.2f} > ${max_daily_loss:.2f}"

        return True, ""

    def _compute_rsi(self, prices: list, period: int = 14) -> float:
        """Compute RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 0.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    async def _one_iteration(self):
        """Single loop iteration."""
        if self.circuit_broken:
            print("Circuit breaker engaged. Waiting...")
            await asyncio.sleep(60)
            return

        goal = await self._load_goal()
        strategy = await self._load_strategy()
        price = await self._fetch_price()

        if price is None:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 5:
                self.circuit_broken = True
                print("5 consecutive failures. Circuit broken.")
            await self._write_heartbeat("price_fetch_failed")
            return

        self.consecutive_failures = 0
        await self._write_heartbeat("ok", price)

        trades = await self._load_trades()
        recent_prices = [t.get("price") for t in trades[-30:] if "price" in t]
        recent_prices.append(price)

        allowed, reason = await self._check_risk_limits(goal, trades)
        if not allowed:
            print(f"Trading halted: {reason}")
            await self._write_heartbeat("trading_halted", price)
            return

        entry_signal = await self._check_entry(strategy, price, recent_prices, goal)
        if entry_signal:
            trade = {
                "entry_price": price,
                "entry_signal": strategy["entry"]["indicator"],
                "stop_loss_pct": strategy["stop_loss_pct"],
                "position_size_r": strategy["position_size_r"],
                "status": "open",
                "price": price,
            }
            await self._append_trade(trade)
            print(f"✓ Entry signal @ {price}")

        await asyncio.sleep(60)

    async def run(self):
        """Run the trading loop forever."""
        try:
            while True:
                await self._one_iteration()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await self.exchange.close()
