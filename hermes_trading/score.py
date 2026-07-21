"""Score trades against goal."""
import numpy as np
from typing import Any


def score(trades: list, goal: dict) -> float:
    """
    Score trades against goal.
    Returns composite score in [-1, +1].
    """
    if not trades:
        return 0.0

    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown = goal.get("max_drawdown", 0.08)
    min_sharpe = goal.get("min_sharpe", 1.2)

    prices = [t.get("price", 0) for t in trades]
    if len(prices) < 2:
        return 0.0

    realised_return = (prices[-1] - prices[0]) / prices[0]

    peak = np.max(prices)
    drawdown = (peak - prices[-1]) / peak if peak > 0 else 0

    rets = np.diff(prices) / prices[:-1]
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8) if len(rets) > 1 else 0

    score_return = min(realised_return / target_return, 1.0) if target_return > 0 else 0
    score_dd = 1.0 - (drawdown / max_drawdown) if max_drawdown > 0 else 1.0
    score_sharpe = min(sharpe / min_sharpe, 1.0) if min_sharpe > 0 else 0

    return np.mean([score_return, score_dd, score_sharpe])
