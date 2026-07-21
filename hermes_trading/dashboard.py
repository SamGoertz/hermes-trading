"""Web dashboard for monitoring trading agent."""
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify
import yaml

app = Flask(__name__)

# Use /app/state in Docker, ~/hermes-trading/state locally
if Path("/app/state").exists():
    STATE_DIR = Path("/app/state")
else:
    STATE_DIR = Path.home() / "hermes-trading" / "state"


def load_yaml(path):
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_trades():
    trades_file = STATE_DIR / "trades.jsonl"
    if not trades_file.exists():
        return []
    with open(trades_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_hypotheses():
    hyp_file = STATE_DIR / "hypotheses.jsonl"
    if not hyp_file.exists():
        return []
    with open(hyp_file) as f:
        return [json.loads(line) for line in f if line.strip()]


@app.route("/")
def dashboard():
    goal = load_yaml(STATE_DIR / "goal.yaml")
    strategy = load_yaml(STATE_DIR / "strategy.yaml")
    trades = load_trades()
    hypotheses = load_hypotheses()

    latest_hyp = hypotheses[-1] if hypotheses else {}
    trading_enabled = goal.get("trading_enabled", True)

    status_color = "#1a4d2e" if trading_enabled else "#8b0000"
    status_text = "TRADING ACTIVE" if trading_enabled else "TRADING STOPPED"
    status_emoji = "ACTIVE" if trading_enabled else "STOPPED"
    button_color = "#dc2626" if trading_enabled else "#22c55e"
    button_text = "STOP TRADING" if trading_enabled else "RESUME TRADING"
    status_indicator = "green" if trading_enabled else "red"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hermes Trading Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0f0f0f;
                color: #fff;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                font-size: 32px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .status {{
                background: {status_color};
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .status-text {{
                font-size: 18px;
                font-weight: bold;
            }}
            .kill-switch {{
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                background: {button_color};
                color: white;
                transition: all 0.3s;
            }}
            .kill-switch:hover {{
                transform: scale(1.05);
                opacity: 0.9;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .card {{
                background: #1a1a1a;
                border: 1px solid #333;
                padding: 20px;
                border-radius: 8px;
            }}
            .card-title {{
                font-size: 12px;
                color: #888;
                text-transform: uppercase;
                margin-bottom: 8px;
            }}
            .card-value {{
                font-size: 32px;
                font-weight: bold;
                color: #22c55e;
            }}
            .card-meta {{
                font-size: 12px;
                color: #666;
                margin-top: 8px;
            }}
            .section {{
                background: #1a1a1a;
                border: 1px solid #333;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .section-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 15px;
                border-bottom: 1px solid #333;
                padding-bottom: 10px;
            }}
            .row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #222;
                font-size: 14px;
            }}
            .row:last-child {{
                border-bottom: none;
            }}
            .label {{
                color: #888;
            }}
            .value {{
                font-weight: bold;
            }}
            .limits {{
                background: #1a2a1a;
                border: 1px solid #2a4d2a;
            }}
            .refresh-time {{
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Hermes Trading Agent</h1>

            <div class="status">
                <div class="status-text">
                    Status: <span style="color: #22c55e;">{status_emoji}</span>
                </div>
                <button class="kill-switch" onclick="toggleTrading()">
                    {button_text}
                </button>
            </div>

            <div class="grid">
                <div class="card">
                    <div class="card-title">Strategy Version</div>
                    <div class="card-value">v{strategy.get("version", "??")}</div>
                    <div class="card-meta">Current Version</div>
                </div>
                <div class="card">
                    <div class="card-title">Total Trades</div>
                    <div class="card-value">{len(trades)}</div>
                    <div class="card-meta">Paper Trades</div>
                </div>
                <div class="card">
                    <div class="card-title">Reflections</div>
                    <div class="card-value">{len(hypotheses)}</div>
                    <div class="card-meta">Strategy Updates</div>
                </div>
            </div>

            <div class="section limits">
                <div class="section-title">Configuration - Editable</div>
                <form id="limitsForm" onsubmit="saveLimits(event)">
                    <div class="row">
                        <span class="label">Trading Asset:</span>
                        <input type="text" id="asset" value="{goal.get("asset", "AAPL")}" style="width: 100px; padding: 4px;">
                    </div>
                    <div class="row">
                        <span class="label">Max Position Size (USD):</span>
                        <input type="number" id="maxPos" value="{goal.get("max_position_size_usd", 100)}" step="10" min="0" style="width: 100px; padding: 4px;">
                    </div>
                    <div class="row">
                        <span class="label">Max Daily Loss (USD):</span>
                        <input type="number" id="maxLoss" value="{goal.get("max_daily_loss_usd", 500)}" step="50" min="0" style="width: 100px; padding: 4px;">
                    </div>
                    <div class="row">
                        <span class="label">Target Return (%):</span>
                        <input type="number" id="targetRtn" value="{goal.get("target_return_30d", 0)*100:.1f}" step="0.5" min="0" style="width: 100px; padding: 4px;">
                    </div>
                    <div class="row">
                        <span class="label">Max Drawdown (%):</span>
                        <input type="number" id="maxDD" value="{goal.get("max_drawdown", 0)*100:.1f}" step="0.5" min="0" style="width: 100px; padding: 4px;">
                    </div>
                    <div class="row" style="margin-top: 15px;">
                        <button type="submit" style="padding: 8px 16px; background: #22c55e; color: #000; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Save All Settings</button>
                    </div>
                </form>
                <div id="saveStatus" style="margin-top: 10px; font-size: 12px; color: #888;"></div>
            </div>

            <div class="section">
                <div class="section-title">Current Strategy (v{strategy.get("version", "?")})</div>
                <div class="row">
                    <span class="label">Entry Indicator:</span>
                    <span class="value">{strategy.get("entry", {}).get("indicator", "N/A").upper()}</span>
                </div>
                <div class="row">
                    <span class="label">Entry Threshold:</span>
                    <span class="value">{strategy.get("entry", {}).get("threshold", "N/A")}</span>
                </div>
                <div class="row">
                    <span class="label">Stop Loss:</span>
                    <span class="value">{strategy.get("stop_loss_pct", "N/A")}%</span>
                </div>
                <div class="row">
                    <span class="label">Position Size:</span>
                    <span class="value">{strategy.get("position_size_r", "N/A")} R</span>
                </div>
            </div>

            <div class="refresh-time">
                Last updated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}
                <br><small>Page auto-refreshes every 30 seconds</small>
            </div>
        </div>

        <script>
            function toggleTrading() {{
                const currentStatus = document.querySelector('.status-text span').innerText.includes('ACTIVE');

                if (!currentStatus) {{
                    // Trading is stopped - show warning before resuming
                    const warningMessage = `
⚠️ WARNING - RESUME TRADING ⚠️

You are about to RESUME LIVE PAPER TRADING.

ACKNOWLEDGMENT:
• This agent executes AUTOMATED trades
• Past performance does not guarantee future results
• The strategy may lose money
• You accept full responsibility for losses
• You have reviewed and approved the current risk limits
• You understand this is paper mode (simulated trades)

Type "I ACCEPT RISK" to resume trading.
Or click Cancel to keep trading stopped.`;

                    const userInput = prompt(warningMessage);
                    if (userInput !== 'I ACCEPT RISK') {{
                        alert('Trading remains stopped.');
                        return;
                    }}
                }}

                fetch('/api/toggle-trading', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}}
                }}).then(() => {{
                    location.reload();
                }}).catch(e => alert('Error: ' + e));
            }}

            function saveLimits(e) {{
                e.preventDefault();
                const data = {{
                    asset: document.getElementById('asset').value,
                    max_position_size_usd: document.getElementById('maxPos').value,
                    max_daily_loss_usd: document.getElementById('maxLoss').value,
                    target_return_30d: document.getElementById('targetRtn').value,
                    max_drawdown: document.getElementById('maxDD').value,
                }};

                fetch('/api/update-limits', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify(data)
                }}).then(r => r.json())
                  .then(result => {{
                    document.getElementById('saveStatus').innerText = 'Settings saved! Reloading...';
                    setTimeout(() => location.reload(), 1000);
                  }})
                  .catch(e => {{
                    document.getElementById('saveStatus').innerText = 'Error: ' + e;
                  }});
            }}

            // Auto-refresh every 30 seconds
            setTimeout(() => location.reload(), 30000);
        </script>
    </body>
    </html>
    """
    return html


@app.route("/api/toggle-trading", methods=["POST"])
def toggle_trading():
    """Toggle trading_enabled in goal.yaml"""
    goal = load_yaml(STATE_DIR / "goal.yaml")
    goal["trading_enabled"] = not goal.get("trading_enabled", True)

    with open(STATE_DIR / "goal.yaml", "w") as f:
        yaml.dump(goal, f, default_flow_style=False)

    return jsonify({"trading_enabled": goal["trading_enabled"]})


@app.route("/api/update-limits", methods=["POST"])
def update_limits():
    """Update risk limits and asset"""
    from flask import request
    data = request.get_json()
    goal = load_yaml(STATE_DIR / "goal.yaml")

    if "asset" in data:
        goal["asset"] = data["asset"].upper()
    if "max_position_size_usd" in data:
        goal["max_position_size_usd"] = float(data["max_position_size_usd"])
    if "max_daily_loss_usd" in data:
        goal["max_daily_loss_usd"] = float(data["max_daily_loss_usd"])
    if "target_return_30d" in data:
        goal["target_return_30d"] = float(data["target_return_30d"]) / 100.0
    if "max_drawdown" in data:
        goal["max_drawdown"] = float(data["max_drawdown"]) / 100.0

    with open(STATE_DIR / "goal.yaml", "w") as f:
        yaml.dump(goal, f, default_flow_style=False)

    return jsonify({"success": True, "goal": goal})


@app.route("/api/status")
def api_status():
    """JSON status endpoint"""
    goal = load_yaml(STATE_DIR / "goal.yaml")
    strategy = load_yaml(STATE_DIR / "strategy.yaml")
    trades = load_trades()
    hypotheses = load_hypotheses()

    return jsonify({
        "trading_enabled": goal.get("trading_enabled", True),
        "strategy_version": strategy.get("version", "??"),
        "total_trades": len(trades),
        "total_reflections": len(hypotheses),
        "asset": goal.get("asset", "BTC/USDT"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
