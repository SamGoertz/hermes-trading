"""Reflection cycle - deterministic fallback and Hermes modes."""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .score import score as score_trades


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_trades(state_dir: Path) -> list:
    trades_file = state_dir / "trades.jsonl"
    if not trades_file.exists():
        return []
    with open(trades_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_yaml(path: Path, data: dict):
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def deterministic_fallback(state_dir: Path) -> dict:
    """Deterministic fallback logic - used before Hermes."""
    goal = load_yaml(state_dir / "goal.yaml")
    strategy = load_yaml(state_dir / "strategy.yaml")
    trades = load_trades(state_dir)

    realised_score = score_trades(trades, goal)
    target_return = goal.get("target_return_30d", 0.05)
    max_drawdown = goal.get("max_drawdown", 0.08)

    hypothesis = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": strategy["version"],
        "reasoning": "",
        "variable_changed": None,
        "old_value": None,
        "new_value": None,
    }

    if realised_score < 0.5:
        if strategy["entry"]["threshold"] > 20:
            old_val = strategy["entry"]["threshold"]
            strategy["entry"]["threshold"] -= 2
            hypothesis["reasoning"] = "Return below target: loosened RSI threshold by 2"
            hypothesis["variable_changed"] = "entry.threshold"
            hypothesis["old_value"] = old_val
            hypothesis["new_value"] = strategy["entry"]["threshold"]
        elif strategy["stop_loss_pct"] > 1.0:
            old_val = strategy["stop_loss_pct"]
            strategy["stop_loss_pct"] -= 0.2
            hypothesis["reasoning"] = "Return below target: tightened stop loss by 0.2%"
            hypothesis["variable_changed"] = "stop_loss_pct"
            hypothesis["old_value"] = old_val
            hypothesis["new_value"] = strategy["stop_loss_pct"]

    version_int = int(strategy["version"])
    version_int += 1
    strategy["version"] = str(version_int).zfill(2)

    return strategy, hypothesis


def main():
    parser = argparse.ArgumentParser(description="Reflection cycle")
    parser.add_argument("--fallback", action="store_true", help="Use deterministic fallback")
    parser.add_argument("--hermes", action="store_true", help="Use Hermes mode")
    args = parser.parse_args()

    state_dir = Path.home() / "hermes-trading" / "state"
    if not state_dir.exists():
        print(f"Error: state directory not found at {state_dir}", file=sys.stderr)
        sys.exit(1)

    if args.fallback:
        strategy, hypothesis = deterministic_fallback(state_dir)
        save_yaml(state_dir / "strategy.yaml", strategy)

        history_file = state_dir / "history" / f"v{strategy['version']}.yaml"
        save_yaml(history_file, strategy)

        hypotheses_file = state_dir / "hypotheses.jsonl"
        with open(hypotheses_file, "a") as f:
            f.write(json.dumps(hypothesis) + "\n")

        print(f"[OK] Reflection complete. Strategy bumped to v{strategy['version']}")
        print(f"  Changed: {hypothesis['variable_changed']}")
        print(f"  Reasoning: {hypothesis['reasoning']}")
    elif args.hermes:
        print("Hermes mode not yet implemented")
        sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
