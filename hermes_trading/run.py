"""Entrypoint for hermes trading worker."""
import asyncio
import argparse
import sys
from pathlib import Path

from .loop import TradingLoop


async def main():
    parser = argparse.ArgumentParser(description="Hermes trading worker")
    parser.add_argument("--asset", default=None, help="Trading pair (e.g. BTC/USDT)")
    args = parser.parse_args()

    state_dir = Path(__file__).parent.parent / "state"
    if not state_dir.exists():
        print(f"Error: state directory not found at {state_dir}", file=sys.stderr)
        sys.exit(1)

    goal_file = state_dir / "goal.yaml"
    if not goal_file.exists():
        print(f"Error: goal.yaml not found at {goal_file}", file=sys.stderr)
        sys.exit(1)

    asset = args.asset
    if not asset:
        import yaml
        with open(goal_file) as f:
            goal = yaml.safe_load(f)
        asset = goal.get("asset", "BTC/USDT")

    print(f"Booting hermes-trading worker")
    print(f"Asset: {asset}")
    print(f"State dir: {state_dir}")

    loop = TradingLoop(state_dir=state_dir, asset=asset)
    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
