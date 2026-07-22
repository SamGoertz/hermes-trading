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

    def _compute_ema_9(self, prices: list) -> Optional[float]:
        """Compute 9-period EMA from price list."""
        if len(prices) < 9:
            return None
        ema_multiplier = 2.0 / (9 + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * ema_multiplier + ema * (1 - ema_multiplier)
        return ema

    def _detect_rsi_higher_low(self, rsi_values: list, lookback: int = 3) -> bool:
        """Check if RSI made a higher low in last N bars"""
        if len(rsi_values) < lookback + 1:
            return False
        recent_rsi = rsi_values[-lookback:]
        prev_rsi = rsi_values[-(lookback + 1)]
        return min(recent_rsi) > prev_rsi

    def _detect_price_lower_low(self, prices: list, lookback: int = 3) -> bool:
        """Check if price made a lower low in last N bars"""
        if len(prices) < lookback + 1:
            return False
        recent_prices = prices[-lookback:]
        prev_price = prices[-(lookback + 1)]
        return min(recent_prices) < prev_price

    def _detect_divergence(self, prices: list, rsi_values: list, lookback: int = 3) -> bool:
        """Bullish divergence: higher RSI low + lower price low"""
        return self._detect_rsi_higher_low(rsi_values, lookback) and \
               self._detect_price_lower_low(prices, lookback)

    def _check_volume_filters(self, current_price: float, volume: int = None) -> bool:
        """Check if symbol meets volume/price criteria"""
        if current_price < 2.0 or current_price > 20.0:
            return False
        # Note: Alpaca doesn't expose RVol easily; skip that filter
        # Check volume if available from yfinance
        if volume and volume < 500000:
            return False
        return True

    async def _check_entry_divergence(self, strategy: dict, price: float,
                                       recent_prices: list, recent_rsi: list,
                                       goal: dict) -> bool:
        """Divergence-based entry for Track B"""
        if strategy["entry"]["direction"] != "long":
            return False
        if not goal.get("trading_enabled", True):
            return False

        if strategy["entry"]["indicator"] == "divergence":
            # Check RSI < 35 in last 3 bars
            if len(recent_rsi) < 3:
                return False
            if not all(r < strategy["entry"]["threshold"] for r in recent_rsi[-3:]):
                return False

            # Check divergence
            if not self._detect_divergence(recent_prices, recent_rsi, 3):
                return False

            # Check volume/price filters
            if not self._check_volume_filters(price):
                return False

            return True
        return False

    def _check_ema_crossover(self, prices: list, ema_values: list, direction: str = "above") -> bool:
        """Check if latest price closes above or below EMA"""
        if not prices or not ema_values:
            return False
        latest_price = prices[-1]
        latest_ema = ema_values[-1]
        if direction == "above":
            return latest_price > latest_ema
        else:  # below
            return latest_price < latest_ema

    def _check_hard_stop_loss(self, entry_price: float, current_price: float, stop_loss_pct: float) -> bool:
        """Check if position hit hard stop loss"""
        loss_pct = ((entry_price - current_price) / entry_price) * 100
        return loss_pct >= stop_loss_pct

    def _calculate_stop_price(self, entry_price: float, current_price: float,
                             trailing_type: str, trailing_value: float,
                             hard_stop_pct: float) -> float:
        """Calculate stop price based on hybrid config.

        Hard stop is absolute floor. Trailing adjusts as price rises.
        Returns the most conservative stop price (highest value = hardest to breach).
        """
        # Hard stop is absolute floor
        hard_stop_price = entry_price * (1 - hard_stop_pct / 100)

        if trailing_type == "none":
            return hard_stop_price

        elif trailing_type == "percent":
            # Trailing percent: stop = current * (1 - trailing%)
            trailing_stop = current_price * (1 - trailing_value / 100)
            return max(hard_stop_price, trailing_stop)

        elif trailing_type == "dollar":
            # Trailing dollar: stop = current - trailing_$
            trailing_stop = current_price - trailing_value
            return max(hard_stop_price, trailing_stop)

        return hard_stop_price

    def _check_stop_loss_hybrid(self, entry_price: float, current_price: float,
                               strategy: dict) -> bool:
        """Check if position should exit based on hybrid stop loss config.

        Returns True if current_price <= calculated_stop_price.
        """
        stop_config = strategy.get("stop_loss", {})
        hard_stop = stop_config.get("hard_stop_pct", 4.0)
        trailing = stop_config.get("trailing", {})

        stop_price = self._calculate_stop_price(
            entry_price,
            current_price,
            trailing.get("type", "none"),
            trailing.get("value", 0),
            hard_stop
        )

        return current_price <= stop_price

    async def _check_early_exit(self, strategy: dict, prices: list, ema_values: list) -> bool:
        """Early exit if price closes below 9 EMA"""
        early_exit_config = strategy.get("early_exit", {})
        if not early_exit_config.get("enabled", False):
            return False

        if early_exit_config.get("trigger") == "closes_below_ema":
            if len(prices) < 1 or len(ema_values) < 1:
                return False
            return self._check_ema_crossover(prices, ema_values, "below")
        return False

    async def _check_entry_divergence_with_ema(self, strategy: dict, price: float,
                                                recent_prices: list, recent_rsi: list,
                                                recent_ema: list, goal: dict) -> bool:
        """Divergence + 9 EMA confirmation for Track B"""
        if strategy["entry"]["direction"] != "long":
            return False
        if not goal.get("trading_enabled", True):
            return False

        if strategy["entry"]["indicator"] == "divergence_with_ema":
            # Check RSI < 35 in last 3 bars
            if len(recent_rsi) < 3:
                return False
            if not all(r < strategy["entry"]["threshold"] for r in recent_rsi[-3:]):
                return False

            # Check divergence
            if not self._detect_divergence(recent_prices, recent_rsi, 3):
                return False

            # Check 9 EMA confirmation - price must close above 9 EMA
            if len(recent_ema) < 1:
                return False
            if not self._check_ema_crossover(recent_prices, recent_ema, "above"):
                return False

            # Check volume/price filters
            if not self._check_volume_filters(price):
                return False

            return True
        return False

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

        # Calculate 9 EMA from recent prices
        recent_ema_9 = self._compute_ema_9(recent_prices)

        # Calculate RSI values for divergence detection
        if len(recent_prices) >= 14:
            recent_rsi = []
            # Compute RSI for multiple windows to detect divergence
            for i in range(max(0, len(recent_prices) - 10), len(recent_prices)):
                window = recent_prices[:i+1] if i >= 13 else recent_prices
                if len(window) >= 14:
                    recent_rsi.append(self._compute_rsi(window))
            if not recent_rsi:
                recent_rsi = [self._compute_rsi(recent_prices)]
        else:
            recent_rsi = []

        allowed, reason = await self._check_risk_limits(goal, trades)
        if not allowed:
            print(f"Trading halted: {reason}")
            await self._write_heartbeat("trading_halted", price)
            return

        # Track A: Check RSI-based entry (original logic)
        entry_signal = await self._check_entry(strategy, price, recent_prices, goal)

        # Track B: Check divergence + EMA-based entry
        if not entry_signal and recent_ema_9 is not None and recent_rsi:
            entry_signal = await self._check_entry_divergence_with_ema(
                strategy, price, recent_prices, recent_rsi,
                [recent_ema_9], goal
            )

        if entry_signal:
            trade = {
                "entry_price": price,
                "entry_signal": strategy["entry"]["indicator"],
                "stop_loss_pct": strategy.get("stop_loss_pct", 4.0),
                "position_size_r": strategy.get("position_size_r", 1.0),
                "status": "open",
                "price": price,
            }
            await self._append_trade(trade)
            print(f"✓ Entry signal @ {price}")

        # Handle open positions - early exit and hard stop loss checks
        open_trades = [t for t in trades if t.get("status") == "open"]
        for open_trade in open_trades:
            exit_reason = None

            # Check hybrid stop loss first (hard stop + optional trailing)
            should_stop_loss = self._check_stop_loss_hybrid(
                open_trade["entry_price"],
                price,
                strategy
            )
            if should_stop_loss:
                exit_reason = "hard_stop_loss"
                print(f"✗ Hard stop loss triggered @ {price}, entry was {open_trade['entry_price']}")

            # Check early exit if no hard stop loss triggered
            if not exit_reason and recent_ema_9 is not None:
                should_early_exit = await self._check_early_exit(strategy, recent_prices, [recent_ema_9])
                if should_early_exit:
                    exit_reason = "closed_below_9ema"
                    print(f"✗ Early exit triggered (closed below 9 EMA) @ {price}")

            # Record exit if triggered
            if exit_reason:
                exit_trade = {
                    "entry_price": open_trade["entry_price"],
                    "exit_price": price,
                    "exit_signal": exit_reason,
                    "entry_signal": open_trade.get("entry_signal"),
                    "status": "closed",
                    "pnl": price - open_trade["entry_price"],
                    "pnl_pct": ((price - open_trade["entry_price"]) / open_trade["entry_price"]) * 100,
                }
                await self._append_trade(exit_trade)

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
