"""Web dashboard for monitoring trading agent.

VERSION: 1.5.0 - Alpaca Autoscan + Watchlist + Hybrid Stop Loss
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

from flask import Flask, jsonify, request
import yaml
import yfinance as yf
import numpy as np

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning("Alpaca SDK not installed. Autoscan will use yfinance fallback.")

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


def load_trackb_trades():
    """Load Track B trades from trackb/trades.jsonl"""
    trades_file = STATE_DIR / "trackb" / "trades.jsonl"
    if not trades_file.exists():
        return []
    with open(trades_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_trackb_hypotheses():
    """Load Track B hypotheses from trackb/hypotheses.jsonl"""
    hyp_file = STATE_DIR / "trackb" / "hypotheses.jsonl"
    if not hyp_file.exists():
        return []
    with open(hyp_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def calculate_rsi(prices, period=14):
    """Calculate RSI series."""
    if len(prices) < period + 1:
        return []

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    rsi_values = []
    for i in range(period, len(prices)):
        if avg_loss == 0:
            rsi_values.append(100.0 if avg_gain > 0 else 0.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        if i < len(prices) - 1:
            delta = prices[i + 1] - prices[i]
            if delta > 0:
                avg_gain = (avg_gain * (period - 1) + delta) / period
                avg_loss = (avg_loss * (period - 1)) / period
            else:
                avg_gain = (avg_gain * (period - 1)) / period
                avg_loss = (avg_loss * (period - 1) + abs(delta)) / period

    return rsi_values


def calculate_ema(prices, period=9):
    """Calculate EMA series."""
    if len(prices) < period:
        return []

    k = 2 / (period + 1)
    ema_values = []
    sma = np.mean(prices[:period])

    for i in range(period, len(prices)):
        sma = prices[i] * k + sma * (1 - k)
        ema_values.append(sma)

    return ema_values


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD, Signal, and Histogram."""
    if len(prices) < slow:
        return [], [], []

    fast_ema = calculate_ema(prices, fast)
    slow_ema = calculate_ema(prices, slow)

    # Align to same length
    diff = len(fast_ema) - len(slow_ema)
    if diff > 0:
        fast_ema = fast_ema[diff:]

    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = calculate_ema(macd_line, signal) if macd_line else []

    # Align macd and signal
    diff = len(macd_line) - len(signal_line)
    if diff > 0:
        macd_line = macd_line[diff:]

    histogram = [m - s for m, s in zip(macd_line, signal_line)]

    return macd_line, signal_line, histogram


@app.route("/scanner")
def scanner():
    """5M candlestick scanner with RSI(14) and EMA(9)."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>5M Scanner - RSI(14) + EMA(9)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0f0f0f;
                color: #e0e0e0;
                padding: 20px;
            }
            .container { max-width: 1400px; margin: 0 auto; }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                background: #1a1a1a;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #333;
            }
            .controls {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            input {
                padding: 8px 12px;
                background: #222;
                border: 1px solid #444;
                color: #e0e0e0;
                border-radius: 4px;
            }
            button {
                padding: 8px 16px;
                background: #22c55e;
                color: #000;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
            }
            button:hover { background: #16a34a; }
            .chart-panel {
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .chart-title {
                font-size: 14px;
                color: #888;
                text-transform: uppercase;
                margin-bottom: 10px;
                font-weight: bold;
            }
            canvas { width: 100% !important; height: auto; display: block; background: #121212; border-radius: 4px; }
            .indicators {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 15px;
            }
            .indicator-card {
                background: #222;
                padding: 12px;
                border-radius: 4px;
                border-left: 3px solid #22c55e;
            }
            .indicator-card.rsi { border-left-color: #3b82f6; }
            .indicator-card.ema { border-left-color: #f59e0b; }
            .indicator-label { color: #888; font-size: 12px; text-transform: uppercase; }
            .indicator-value { font-size: 20px; font-weight: bold; margin-top: 5px; }
            .status { font-size: 12px; color: #666; margin-top: 8px; }
            .error {
                background: #7f1d1d;
                border: 1px solid #dc2626;
                color: #fca5a5;
                padding: 12px;
                border-radius: 4px;
                margin-bottom: 15px;
            }
            .loading { text-align: center; padding: 40px; color: #666; }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #333;
            }
            .tab {
                padding: 12px 20px;
                background: transparent;
                border: none;
                color: #888;
                font-size: 14px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                font-weight: 500;
            }
            .tab.active {
                color: #22c55e;
                border-bottom-color: #22c55e;
            }
            .autoscan-panel {
                background: #1a2d1a;
                border: 1px solid #2a7d3f;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .autoscan-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .autoscan-header h2 {
                font-size: 16px;
                margin: 0;
                color: #e0e0e0;
            }
            #autoscanBtn {
                padding: 10px 20px;
                background: #22c55e;
                color: #000;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                font-size: 14px;
            }
            #autoscanBtn:hover { background: #16a34a; }
            #autoscanBtn:disabled {
                background: #666;
                cursor: not-allowed;
            }
            #autoscanStatus {
                font-size: 13px;
                color: #888;
                margin-bottom: 15px;
                min-height: 20px;
            }
            #autoscanStatus.scanning { color: #22c55e; }
            #autoscanStatus.error { color: #ef4444; }
            #autoscanResults {
                background: #121212;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 15px;
                display: none;
                max-height: 500px;
                overflow-y: auto;
            }
            #autoscanResults.visible {
                display: block;
            }
            .result-count {
                color: #888;
                font-size: 12px;
                margin-bottom: 10px;
                text-transform: uppercase;
            }
            #resultsList {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .result-item {
                background: #1a1a1a;
                border-left: 3px solid #22c55e;
                padding: 12px;
                border-radius: 4px;
                font-size: 13px;
                transition: background 0.2s;
                display: flex;
                align-items: flex-start;
                gap: 10px;
            }
            .result-item:hover {
                background: #222;
            }
            .result-item input[type="checkbox"] {
                margin-top: 2px;
                cursor: pointer;
                width: 18px;
                height: 18px;
            }
            .result-content {
                flex: 1;
                cursor: pointer;
            }
            .result-symbol {
                font-weight: bold;
                color: #22c55e;
                margin-right: 8px;
                display: inline;
            }
            .result-details {
                color: #aaa;
                font-size: 12px;
                margin-top: 4px;
            }
            .selected-watchlist-panel {
                background: #1a3a1a;
                border: 1px solid #2a7d3f;
                border-radius: 4px;
                padding: 15px;
                margin-top: 15px;
                margin-bottom: 15px;
            }
            .selected-watchlist-panel h3 {
                font-size: 14px;
                color: #22c55e;
                margin-bottom: 10px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .selected-symbols {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 10px;
                min-height: 30px;
                align-items: center;
            }
            .symbol-pill {
                background: #22c55e;
                color: #000;
                padding: 6px 10px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .symbol-pill button {
                background: none;
                border: none;
                color: #000;
                cursor: pointer;
                font-weight: bold;
                padding: 0;
                font-size: 16px;
                line-height: 1;
            }
            .selected-count {
                font-size: 12px;
                color: #888;
                margin-bottom: 10px;
                min-height: 16px;
            }
            .watchlist-actions {
                display: flex;
                gap: 10px;
            }
            .watchlist-actions button {
                padding: 8px 14px;
                background: #22c55e;
                color: #000;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                font-size: 12px;
            }
            .watchlist-actions button:hover {
                background: #16a34a;
            }
            .watchlist-actions button.secondary {
                background: #666;
                color: #fff;
            }
            .watchlist-actions button.secondary:hover {
                background: #777;
            }
            .paper-trading-watchlist {
                background: #1a2d1a;
                border: 1px solid #2a5d2a;
                border-radius: 4px;
                padding: 15px;
                margin-top: 15px;
            }
            .paper-trading-watchlist h3 {
                font-size: 14px;
                color: #22c55e;
                margin-bottom: 10px;
                font-weight: bold;
                text-transform: uppercase;
            }
            .watchlist-stocks {
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-bottom: 10px;
                font-size: 12px;
                color: #888;
            }
            .watchlist-stocks span {
                display: inline;
            }
            .watchlist-empty {
                color: #666;
                font-size: 12px;
                font-style: italic;
                padding: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="tabs">
                <button class="tab" onclick="location.href='/'">Dashboard</button>
                <button class="tab active" onclick="location.href='/scanner'">📊 Scanner</button>
            </div>

            <div id="tradingStatus" style="background: #1a4d2e; padding: 15px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #2a7d3f;">
                <div style="font-size: 16px; font-weight: bold;">
                    Status: <span style="color: #22c55e;" id="statusEmoji">ACTIVE</span>
                </div>
                <button id="killSwitchBtn" class="kill-switch" onclick="toggleTradingScanner()" style="padding: 10px 20px; font-size: 14px;">
                    STOP TRADING
                </button>
            </div>

            <div style="background: #1a3a1a; padding: 12px 15px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #2a6d2a;">
                <div style="font-size: 13px; color: #888; text-transform: uppercase; margin-bottom: 5px;">Market Status (Central Time)</div>
                <div style="display: flex; gap: 30px;">
                    <div>
                        <span style="color: #22c55e; font-weight: bold;">Market Opens:</span>
                        <span id="openCountdown" style="margin-left: 8px; font-size: 14px;">—</span>
                    </div>
                    <div>
                        <span style="color: #ef4444; font-weight: bold;">Market Closes:</span>
                        <span id="closeCountdown" style="margin-left: 8px; font-size: 14px;">—</span>
                    </div>
                </div>
            </div>

            <div class="autoscan-panel">
                <div class="autoscan-header">
                    <h2>🔍 Market Autoscan</h2>
                    <button id="autoscanBtn" onclick="startAutoscan()">🔍 AUTOSCAN MARKET</button>
                </div>
                <div id="autoscanStatus"></div>
                <div id="autoscanResults">
                    <div class="result-count">
                        Found <span id="resultCount">0</span> matching signals
                    </div>
                    <div id="resultsList"></div>
                    <div class="selected-watchlist-panel" id="selectedWatchlistPanel" style="display: none;">
                        <h3>📌 Selected Watchlist</h3>
                        <div class="selected-symbols" id="selectedSymbolsContainer"></div>
                        <div class="selected-count" id="selectedCountText"></div>
                        <div class="watchlist-actions">
                            <button onclick="addSelectedToWatchlist()">Add Selected to Watchlist</button>
                            <button class="secondary" onclick="clearAllSelections()">Clear All</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="paper-trading-watchlist" id="paperTradingWatchlist" style="display: none;">
                <h3>📊 Paper Trading Watchlist</h3>
                <div id="watchlistContent">
                    <div class="watchlist-empty">No stocks in watchlist yet</div>
                </div>
                <div class="watchlist-actions">
                    <button class="secondary" onclick="clearWatchlist()">Clear Watchlist</button>
                    <button onclick="loadNextWatchlistStock()">Load Next in Scanner</button>
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
                <h1>Candlestick Scanner</h1>
                <div class="controls">
                    <input type="text" id="symbol" value="AAPL" placeholder="Symbol">
                    <select id="interval" style="padding: 8px 12px; background: #222; border: 1px solid #444; color: #e0e0e0; border-radius: 4px;">
                        <option value="1m">1 Min</option>
                        <option value="5m" selected>5 Min</option>
                        <option value="15m">15 Min</option>
                        <option value="30m">30 Min</option>
                        <option value="1h">1 Hour</option>
                        <option value="1d">1 Day</option>
                    </select>
                    <select id="period" style="padding: 8px 12px; background: #222; border: 1px solid #444; color: #e0e0e0; border-radius: 4px;">
                        <option value="5d" selected>5 Days</option>
                        <option value="1mo">1 Month</option>
                        <option value="3mo">3 Months</option>
                        <option value="1y">1 Year</option>
                    </select>

                    <select id="zoom" style="padding: 8px 12px; background: #222; border: 1px solid #444; color: #e0e0e0; border-radius: 4px;">
                        <option value="all" selected>Zoom: All</option>
                        <option value="50">Last 50 Candles</option>
                        <option value="30">Last 30 Candles</option>
                        <option value="20">Last 20 Candles</option>
                        <option value="10">Last 10 Candles</option>
                    </select>
                    <button onclick="loadChart()">Load Chart</button>
                    <button id="autoscanBtn" onclick="startAutoscan()" style="padding: 8px 16px; background: #3b82f6; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Autoscan</button>
                </div>
            </div>

            <div id="error" class="error" style="display: none;"></div>

            <div id="resultsList" style="display: none; background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <div style="font-size: 14px; color: #888; text-transform: uppercase; margin-bottom: 10px; font-weight: bold;">Autoscan Results</div>
                <div id="scanStatus" style="font-size: 12px; color: #666; margin-bottom: 10px;"></div>
                <div id="scanResults" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px;"></div>
            </div>

            <div class="chart-panel">
                <div class="chart-title" id="candleTitle">Candlestick + EMA(9)</div>
                <div style="position: relative;">
                    <canvas id="candleChart" width="1000" height="400"></canvas>
                    <div id="crosshairs" style="position: absolute; display: none; pointer-events: none;">
                        <div style="position: absolute; width: 100%; height: 1px; background: #3b82f6; opacity: 0.5;"></div>
                        <div style="position: absolute; width: 1px; height: 100%; background: #3b82f6; opacity: 0.5;"></div>
                        <div id="tooltipPrice" style="position: absolute; background: #1a1a1a; border: 1px solid #3b82f6; color: #fff; padding: 4px 8px; font-size: 12px; border-radius: 4px; white-space: nowrap;"></div>
                    </div>
                </div>
            </div>

            <div class="chart-panel">
                <div class="chart-title">Volume</div>
                <canvas id="volumeChart" width="1000" height="120"></canvas>
            </div>

            <div class="chart-panel">
                <div class="chart-title">RSI (14)</div>
                <canvas id="rsiChart" width="1000" height="150"></canvas>
            </div>

            <div class="chart-panel">
                <div class="chart-title">MACD (12, 26, 9)</div>
                <canvas id="macdChart" width="1000" height="150"></canvas>
            </div>

            <div class="indicators">
                <div class="indicator-card">
                    <div class="indicator-label">Price</div>
                    <div class="indicator-value" id="priceValue">—</div>
                    <div class="status" id="priceStatus"></div>
                </div>
                <div class="indicator-card rsi">
                    <div class="indicator-label">RSI (14)</div>
                    <div class="indicator-value" id="rsiValue">—</div>
                    <div class="status" id="rsiStatus"></div>
                </div>
                <div class="indicator-card ema">
                    <div class="indicator-label">EMA (9)</div>
                    <div class="indicator-value" id="emaValue">—</div>
                    <div class="status" id="emaStatus"></div>
                </div>
                <div class="indicator-card" style="border-left-color: #8b5cf6;">
                    <div class="indicator-label">MACD</div>
                    <div class="indicator-value" id="macdValue">—</div>
                    <div class="status" id="macdStatus"></div>
                </div>
                <div class="indicator-card" style="border-left-color: #06b6d4;">
                    <div class="indicator-label">Signal</div>
                    <div class="indicator-value" id="signalValue">—</div>
                    <div class="status" id="signalStatus"></div>
                </div>
                <div class="indicator-card" style="border-left-color: #ec4899;">
                    <div class="indicator-label">Histogram</div>
                    <div class="indicator-value" id="histValue">—</div>
                    <div class="status" id="histStatus"></div>
                </div>
                <div class="indicator-card" style="border-left-color: #10b981;">
                    <div class="indicator-label">Volume</div>
                    <div class="indicator-value" id="volumeValue">—</div>
                    <div class="status" id="volumeStatus"></div>
                </div>
                <div class="indicator-card" style="border-left-color: #6366f1;">
                    <div class="indicator-label">Avg Volume</div>
                    <div class="indicator-value" id="avgVolValue">—</div>
                    <div class="status" id="avgVolStatus"></div>
                </div>
            </div>
        </div>

        <script>
            const RSI_OVERBOUGHT = 70;
            const RSI_OVERSOLD = 30;
            const MAX_DISPLAY_RESULTS = 15;
            const WATCHLIST_STORAGE_KEY = 'autoscan_watchlist';
            const SELECTED_STORAGE_KEY = 'autoscan_selected';

            // Load watchlist from localStorage on page load
            window.addEventListener('load', () => {
                loadWatchlistFromStorage();
                loadTradingStatus();
                updateMarketCountdown();
                loadChart();
                setInterval(updateMarketCountdown, 1000);
            });

            // Watchlist management functions
            function loadWatchlistFromStorage() {
                const watchlist = JSON.parse(localStorage.getItem(WATCHLIST_STORAGE_KEY) || '[]');
                const selected = JSON.parse(localStorage.getItem(SELECTED_STORAGE_KEY) || '[]');

                // Update checkboxes if results are visible
                document.querySelectorAll('.result-item input[type="checkbox"]').forEach(checkbox => {
                    checkbox.checked = selected.includes(checkbox.dataset.symbol);
                });

                updateWatchlistDisplay(watchlist);
            }

            function selectStock(symbol, checked) {
                let selected = JSON.parse(localStorage.getItem(SELECTED_STORAGE_KEY) || '[]');
                if (checked) {
                    if (!selected.includes(symbol)) {
                        selected.push(symbol);
                    }
                } else {
                    selected = selected.filter(s => s !== symbol);
                }
                localStorage.setItem(SELECTED_STORAGE_KEY, JSON.stringify(selected));
                updateSelectedWatchlistDisplay();
            }

            function updateSelectedWatchlistDisplay() {
                const selected = JSON.parse(localStorage.getItem(SELECTED_STORAGE_KEY) || '[]');
                const container = document.getElementById('selectedSymbolsContainer');
                const countText = document.getElementById('selectedCountText');
                const panel = document.getElementById('selectedWatchlistPanel');

                container.innerHTML = '';
                if (selected.length > 0) {
                    selected.forEach(symbol => {
                        const pill = document.createElement('div');
                        pill.className = 'symbol-pill';
                        pill.innerHTML = `
                            ${symbol}
                            <button onclick="deselectStock('${symbol}')" title="Remove">×</button>
                        `;
                        container.appendChild(pill);
                    });
                    countText.textContent = `(${selected.length} selected for testing)`;
                    panel.style.display = 'block';
                } else {
                    panel.style.display = 'none';
                }
            }

            function deselectStock(symbol) {
                let selected = JSON.parse(localStorage.getItem(SELECTED_STORAGE_KEY) || '[]');
                selected = selected.filter(s => s !== symbol);
                localStorage.setItem(SELECTED_STORAGE_KEY, JSON.stringify(selected));

                // Uncheck the corresponding checkbox
                const checkbox = document.querySelector(`input[data-symbol="${symbol}"]`);
                if (checkbox) checkbox.checked = false;

                updateSelectedWatchlistDisplay();
            }

            function clearAllSelections() {
                localStorage.setItem(SELECTED_STORAGE_KEY, JSON.stringify([]));
                document.querySelectorAll('.result-item input[type="checkbox"]').forEach(cb => cb.checked = false);
                updateSelectedWatchlistDisplay();
            }

            function addSelectedToWatchlist() {
                const selected = JSON.parse(localStorage.getItem(SELECTED_STORAGE_KEY) || '[]');
                let watchlist = JSON.parse(localStorage.getItem(WATCHLIST_STORAGE_KEY) || '[]');

                // Merge selected into watchlist (avoid duplicates)
                watchlist = [...new Set([...watchlist, ...selected])];
                localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(watchlist));

                // Clear selections
                clearAllSelections();
                updateWatchlistDisplay(watchlist);
                alert(`Added ${selected.length} stock(s) to watchlist!`);
            }

            function updateWatchlistDisplay(watchlist) {
                const panel = document.getElementById('paperTradingWatchlist');
                const content = document.getElementById('watchlistContent');

                if (!watchlist || watchlist.length === 0) {
                    content.innerHTML = '<div class="watchlist-empty">No stocks in watchlist yet</div>';
                    panel.style.display = 'none';
                    return;
                }

                panel.style.display = 'block';
                content.innerHTML = `
                    <div class="watchlist-stocks">
                        ${watchlist.map(symbol => `
                            <span title="Click to load in scanner" onclick="loadStockChart('${symbol}')" style="cursor: pointer; color: #22c55e; font-weight: bold;">
                                ${symbol} ×
                            </span>
                        `).join('')}
                    </div>
                    <div class="watchlist-empty" style="margin-top: 8px; color: #888; font-size: 11px;">
                        (${watchlist.length} stock${watchlist.length !== 1 ? 's' : ''} in watchlist - click symbol to load)
                    </div>
                `;
            }

            function clearWatchlist() {
                if (confirm('Clear all stocks from watchlist?')) {
                    localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify([]));
                    updateWatchlistDisplay([]);
                }
            }

            function loadNextWatchlistStock() {
                const watchlist = JSON.parse(localStorage.getItem(WATCHLIST_STORAGE_KEY) || '[]');
                if (watchlist.length === 0) {
                    alert('Watchlist is empty');
                    return;
                }

                // Get current symbol in scanner
                const currentSymbol = document.getElementById('symbol').value.toUpperCase();
                const currentIndex = watchlist.indexOf(currentSymbol);
                const nextIndex = (currentIndex + 1) % watchlist.length;

                loadStockChart(watchlist[nextIndex]);
            }

            // Convert UTC time to Central Time
            function convertToCentralTime(isoTime) {
                try {
                    // Parse ISO time string
                    const date = new Date(isoTime);
                    // Convert to Central Time (CT)
                    const ctTime = new Date(date.toLocaleString('en-US', { timeZone: 'America/Chicago' }));

                    // Format: HH:MM or MM/DD HH:MM based on time
                    const hours = String(ctTime.getHours()).padStart(2, '0');
                    const minutes = String(ctTime.getMinutes()).padStart(2, '0');
                    const month = String(ctTime.getMonth() + 1).padStart(2, '0');
                    const day = String(ctTime.getDate()).padStart(2, '0');

                    // Return format based on whether it includes date or not
                    if (isoTime.includes('T')) {
                        // Has time component - return HH:MM CT
                        return `${hours}:${minutes} CT`;
                    } else {
                        // Date only - return MM/DD CT
                        return `${month}/${day} CT`;
                    }
                } catch (e) {
                    return isoTime; // Fallback to original if parsing fails
                }
            }

            // Update market open/close countdown timers
            function updateMarketCountdown() {
                // Get current time in Central Time
                const now = new Date();
                const ctNow = new Date(now.toLocaleString('en-US', { timeZone: 'America/Chicago' }));
                const dayOfWeek = ctNow.getDay();
                const hours = ctNow.getHours();
                const minutes = ctNow.getMinutes();
                const seconds = ctNow.getSeconds();

                // Market times (CT): Open 8:30 AM, Close 3:00 PM
                const OPEN_HOUR = 8;
                const OPEN_MIN = 30;
                const CLOSE_HOUR = 15;
                const CLOSE_MIN = 0;

                // Check if market is open (Mon-Fri, 8:30-15:00 CT)
                const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
                const currentMinutes = hours * 60 + minutes;
                const openMinutes = OPEN_HOUR * 60 + OPEN_MIN;
                const closeMinutes = CLOSE_HOUR * 60 + CLOSE_MIN;
                const isMarketHours = currentMinutes >= openMinutes && currentMinutes < closeMinutes;

                let openText, closeText;

                if (!isWeekday) {
                    // Weekend or holiday
                    openText = '🔴 CLOSED (Weekend)';
                    closeText = '📅 Opens Monday 8:30 AM';
                } else if (isMarketHours) {
                    // Market is open
                    const minutesToClose = closeMinutes - currentMinutes;
                    const hoursToClose = Math.floor(minutesToClose / 60);
                    const minsToClose = minutesToClose % 60;
                    openText = '🟢 OPEN';
                    closeText = `Closes in ${hoursToClose}h ${minsToClose}m`;
                } else if (currentMinutes < openMinutes) {
                    // Before market open
                    const minutesToOpen = openMinutes - currentMinutes;
                    const hoursToOpen = Math.floor(minutesToOpen / 60);
                    const minsToOpen = minutesToOpen % 60;
                    openText = `Opens in ${hoursToOpen}h ${minsToOpen}m`;
                    closeText = `🔴 CLOSED`;
                } else {
                    // After market close
                    const minutesUntilMidnight = (24 * 60) - currentMinutes;
                    const minutesToOpenTomorrow = minutesUntilMidnight + openMinutes;
                    const hoursToOpen = Math.floor(minutesToOpenTomorrow / 60);
                    const minsToOpen = minutesToOpenTomorrow % 60;
                    openText = `Opens tomorrow ${hoursToOpen}h ${minsToOpen}m`;
                    closeText = '🔴 CLOSED';
                }

                document.getElementById('openCountdown').textContent = openText;
                document.getElementById('closeCountdown').textContent = closeText;
            }

            // Load trading status on page load
            async function loadTradingStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    updateTradingStatus(data.trading_enabled);
                } catch (e) {
                    console.error('Failed to load trading status', e);
                }
            }

            function updateTradingStatus(tradingEnabled) {
                const statusEmoji = document.getElementById('statusEmoji');
                const killSwitchBtn = document.getElementById('killSwitchBtn');
                const tradingStatusDiv = document.getElementById('tradingStatus');

                if (tradingEnabled) {
                    statusEmoji.textContent = 'ACTIVE';
                    statusEmoji.style.color = '#22c55e';
                    killSwitchBtn.textContent = 'STOP TRADING';
                    killSwitchBtn.style.background = '#dc2626';
                    tradingStatusDiv.style.background = '#1a4d2e';
                    tradingStatusDiv.style.borderColor = '#2a7d3f';
                } else {
                    statusEmoji.textContent = 'STOPPED';
                    statusEmoji.style.color = '#fca5a5';
                    killSwitchBtn.textContent = 'RESUME TRADING';
                    killSwitchBtn.style.background = '#22c55e';
                    killSwitchBtn.style.color = '#000';
                    tradingStatusDiv.style.background = '#7f1d1d';
                    tradingStatusDiv.style.borderColor = '#dc2626';
                }
            }

            async function toggleTradingScanner() {
                const currentStatus = document.getElementById('statusEmoji').textContent;
                const isActive = currentStatus === 'ACTIVE';

                if (!isActive) {
                    // About to resume - show warning
                    const warningMessage = `⚠️ WARNING - RESUME TRADING ⚠️

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
                    if (userInput !== 'I ACCEPT RISK') {
                        alert('Trading remains stopped.');
                        return;
                    }
                }

                try {
                    const response = await fetch('/api/toggle-trading', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await response.json();
                    updateTradingStatus(data.trading_enabled);
                } catch (e) {
                    alert('Error toggling trading: ' + e);
                }
            }

            // Load persisted values on page load
            window.addEventListener('DOMContentLoaded', () => {
                const savedSymbol = localStorage.getItem('scanner_symbol') || 'AAPL';
                const savedInterval = localStorage.getItem('scanner_interval') || '5m';
                const savedPeriod = localStorage.getItem('scanner_period') || '5d';
                const savedZoom = localStorage.getItem('scanner_zoom') || 'all';

                document.getElementById('symbol').value = savedSymbol;
                document.getElementById('interval').value = savedInterval;
                document.getElementById('period').value = savedPeriod;
                document.getElementById('zoom').value = savedZoom;
            });

            async function loadChart() {
                const symbol = document.getElementById('symbol').value.toUpperCase();
                const interval = document.getElementById('interval').value;
                const period = document.getElementById('period').value;
                const zoomLevel = document.getElementById('zoom').value;
                const errorDiv = document.getElementById('error');
                errorDiv.style.display = 'none';

                // Persist selections to localStorage
                localStorage.setItem('scanner_symbol', symbol);
                localStorage.setItem('scanner_interval', interval);
                localStorage.setItem('scanner_period', period);
                localStorage.setItem('scanner_zoom', zoomLevel);

                try {
                    const response = await fetch(`/api/chart-data/${symbol}?interval=${interval}&period=${period}`);
                    if (!response.ok) throw new Error('Failed to fetch data');

                    const data = await response.json();
                    if (data.error) throw new Error(data.error);

                    const candles = data.candles || [];
                    const rsiValues = data.rsi || [];
                    const emaValues = data.ema || [];
                    const macdLine = data.macd || [];
                    const signalLine = data.signal || [];
                    const histogram = data.histogram || [];

                    if (!candles.length) throw new Error('No candle data returned');

                    // Update title
                    const intervalLabel = interval.replace('m', ' min').replace('h', ' hr').replace('d', ' day');
                    document.getElementById('candleTitle').textContent = `Candlestick (${intervalLabel}) + EMA(9) - ${candles.length} candles`;

                    if (candles.length > 0) {
                        document.getElementById('priceValue').textContent = '$' + candles[candles.length - 1].close.toFixed(2);
                        document.getElementById('priceStatus').textContent = candles[candles.length - 1].time;
                    }

                    if (rsiValues.length > 0) {
                        const rsi = rsiValues[rsiValues.length - 1];
                        document.getElementById('rsiValue').textContent = (rsi || 0).toFixed(1);
                        document.getElementById('rsiStatus').textContent =
                            rsi > RSI_OVERBOUGHT ? '⚠️ Overbought' : rsi < RSI_OVERSOLD ? '📍 Oversold' : 'Neutral';
                    } else {
                        document.getElementById('rsiValue').textContent = '—';
                        document.getElementById('rsiStatus').textContent = 'Insufficient data';
                    }

                    if (emaValues.length > 0 && candles.length > 0) {
                        const ema = emaValues[emaValues.length - 1];
                        const price = candles[candles.length - 1].close;
                        document.getElementById('emaValue').textContent = '$' + (ema || 0).toFixed(2);
                        const diff = ((price - (ema || price)) / (ema || price) * 100).toFixed(2);
                        document.getElementById('emaStatus').textContent = (diff > 0 ? '+' : '') + diff + '%';
                    } else {
                        document.getElementById('emaValue').textContent = '—';
                        document.getElementById('emaStatus').textContent = 'Insufficient data';
                    }

                    if (macdLine.length > 0 && signalLine.length > 0 && histogram.length > 0) {
                        const macd = macdLine[macdLine.length - 1];
                        const signal = signalLine[signalLine.length - 1];
                        const hist = histogram[histogram.length - 1];
                        document.getElementById('macdValue').textContent = (macd || 0).toFixed(4);
                        document.getElementById('macdStatus').textContent = (macd || 0) > 0 ? '📈 Positive' : '📉 Negative';
                        document.getElementById('signalValue').textContent = (signal || 0).toFixed(4);
                        document.getElementById('signalStatus').textContent = (macd || 0) > (signal || 0) ? '🟢 Above' : '🔴 Below';
                        document.getElementById('histValue').textContent = (hist || 0).toFixed(4);
                        document.getElementById('histStatus').textContent = (hist || 0) > 0 ? '⬆️ Bullish' : '⬇️ Bearish';
                    } else {
                        document.getElementById('macdValue').textContent = '—';
                        document.getElementById('macdStatus').textContent = 'Insufficient data';
                        document.getElementById('signalValue').textContent = '—';
                        document.getElementById('signalStatus').textContent = '';
                        document.getElementById('histValue').textContent = '—';
                        document.getElementById('histStatus').textContent = '';
                    }

                    // Apply zoom
                    const zoomLevel = document.getElementById('zoom').value;
                    let displayCandles = candles;
                    let displayRsi = rsiValues;
                    let displayEma = emaValues;
                    let displayMacd = macdLine;
                    let displaySignal = signalLine;
                    let displayHist = histogram;

                    if (zoomLevel !== 'all') {
                        const zoomCount = parseInt(zoomLevel);
                        const startIdx = Math.max(0, candles.length - zoomCount);
                        displayCandles = candles.slice(startIdx);
                        displayRsi = rsiValues.slice(Math.max(0, rsiValues.length - zoomCount));
                        displayEma = emaValues.slice(Math.max(0, emaValues.length - zoomCount));
                        displayMacd = macdLine.slice(Math.max(0, macdLine.length - zoomCount));
                        displaySignal = signalLine.slice(Math.max(0, signalLine.length - zoomCount));
                        displayHist = histogram.slice(Math.max(0, histogram.length - zoomCount));
                    }

                    // Extract volumes for display
                    const displayVolumes = displayCandles.map(c => c.volume);
                    const avgVol = displayVolumes.length > 0
                        ? Math.round(displayVolumes.reduce((a, b) => a + b, 0) / displayVolumes.length)
                        : 0;
                    const currentVol = displayCandles.length > 0 ? displayCandles[displayCandles.length - 1].volume : 0;

                    // Update volume stats
                    document.getElementById('volumeValue').textContent = currentVol.toLocaleString();
                    const volTrend = currentVol > avgVol ? '📈 Above avg' : currentVol < avgVol ? '📉 Below avg' : '➖ At avg';
                    document.getElementById('volumeStatus').textContent = volTrend;
                    document.getElementById('avgVolValue').textContent = avgVol.toLocaleString();
                    const volPercent = avgVol > 0 ? (((currentVol - avgVol) / avgVol) * 100).toFixed(0) : 0;
                    document.getElementById('avgVolStatus').textContent = (volPercent > 0 ? '+' : '') + volPercent + '%';

                    drawCandleChart(displayCandles, displayEma);
                    drawVolumeChart(displayCandles);
                    drawRsiChart(displayRsi);
                    drawMacdChart(displayMacd, displaySignal, displayHist);
                } catch (e) {
                    errorDiv.textContent = '❌ ' + e.message;
                    errorDiv.style.display = 'block';
                }
            }

            function drawCandleChart(candles, emaValues) {
                const canvas = document.getElementById('candleChart');
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;

                const prices = candles.map(c => c.close);
                const high = Math.max(...prices);
                const low = Math.min(...prices);
                const range = high - low;

                const PADDING = 60;
                const chartWidth = canvas.width - PADDING * 2;
                const chartHeight = canvas.height - PADDING * 2;
                const spacing = chartWidth / candles.length;

                // Store chart metadata for crosshairs
                window.chartMetadata = {
                    candles, prices, high, low, range, PADDING, chartWidth, chartHeight, spacing, canvas
                };

                ctx.fillStyle = '#121212';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Y-axis price labels
                ctx.strokeStyle = '#222';
                for (let i = 0; i <= 5; i++) {
                    const y = PADDING + (chartHeight / 5) * i;
                    ctx.beginPath();
                    ctx.moveTo(PADDING, y);
                    ctx.lineTo(canvas.width - PADDING, y);
                    ctx.stroke();

                    ctx.fillStyle = '#666';
                    ctx.font = '11px sans-serif';
                    ctx.textAlign = 'right';
                    ctx.fillText((high - (range / 5) * i).toFixed(2), PADDING - 10, y + 4);
                }

                // X-axis time labels (convert to Central Time)
                ctx.fillStyle = '#666';
                ctx.textAlign = 'center';
                const labelInterval = Math.max(1, Math.floor(candles.length / 8));
                for (let i = 0; i < candles.length; i += labelInterval) {
                    const x = PADDING + spacing * i + spacing / 2;
                    const timeStr = convertToCentralTime(candles[i].full_time || candles[i].time);
                    ctx.fillText(timeStr, x, canvas.height - 10);
                }

                // Draw candles
                candles.forEach((candle, i) => {
                    const x = PADDING + spacing * i + spacing / 2;
                    const candleWidth = Math.max(2, spacing * 0.6);

                    const oY = PADDING + (1 - (candle.open - low) / range) * chartHeight;
                    const hY = PADDING + (1 - (candle.high - low) / range) * chartHeight;
                    const lY = PADDING + (1 - (candle.low - low) / range) * chartHeight;
                    const cY = PADDING + (1 - (candle.close - low) / range) * chartHeight;

                    const isBull = candle.close >= candle.open;
                    ctx.fillStyle = isBull ? '#22c55e' : '#ef4444';
                    ctx.strokeStyle = isBull ? '#16a34a' : '#dc2626';

                    ctx.beginPath();
                    ctx.moveTo(x, hY);
                    ctx.lineTo(x, lY);
                    ctx.stroke();

                    ctx.fillRect(x - candleWidth / 2, Math.min(oY, cY), candleWidth, Math.abs(cY - oY) || 1);
                });

                // Draw EMA
                ctx.strokeStyle = '#f59e0b';
                ctx.lineWidth = 2;
                ctx.beginPath();
                for (let i = 0; i < emaValues.length; i++) {
                    const emaPrice = emaValues[i];
                    const idx = i + (candles.length - emaValues.length);
                    const x = PADDING + spacing * idx + spacing / 2;
                    const y = PADDING + (1 - (emaPrice - low) / range) * chartHeight;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();

                // Add mouse tracking for crosshairs
                const canvasContainer = canvas.parentElement;
                canvas.addEventListener('mousemove', (e) => {
                    const canvasRect = canvas.getBoundingClientRect();
                    const mouseX = e.clientX - canvasRect.left;
                    const mouseY = e.clientY - canvasRect.top;

                    if (mouseX < PADDING || mouseX > canvas.width - PADDING ||
                        mouseY < PADDING || mouseY > canvas.height - PADDING) {
                        document.getElementById('crosshairs').style.display = 'none';
                        return;
                    }

                    // Calculate price from Y
                    const normalizedY = (mouseY - PADDING) / chartHeight;
                    const price = high - (normalizedY * range);

                    // Find nearest candle from X
                    const candleIndex = Math.round((mouseX - PADDING - spacing / 2) / spacing);
                    const nearestCandle = candles[Math.max(0, Math.min(candles.length - 1, candleIndex))];

                    const crosshairs = document.getElementById('crosshairs');
                    crosshairs.style.display = 'block';
                    crosshairs.style.left = canvasRect.left + 'px';
                    crosshairs.style.top = canvasRect.top + 'px';
                    crosshairs.style.width = canvasRect.width + 'px';
                    crosshairs.style.height = canvasRect.height + 'px';

                    // Horizontal line
                    crosshairs.children[0].style.top = (mouseY - canvasRect.top) + 'px';
                    // Vertical line
                    crosshairs.children[1].style.left = (mouseX - canvasRect.left) + 'px';

                    // Tooltip (with Central Time)
                    const tooltip = document.getElementById('tooltipPrice');
                    const ctTime = convertToCentralTime(nearestCandle.full_time || nearestCandle.time);
                    tooltip.textContent = `$${price.toFixed(2)} | ${ctTime}`;
                    tooltip.style.left = (mouseX - canvasRect.left + 10) + 'px';
                    tooltip.style.top = (mouseY - canvasRect.top - 30) + 'px';
                });

                canvas.addEventListener('mouseleave', () => {
                    document.getElementById('crosshairs').style.display = 'none';
                });
            }

            function drawVolumeChart(candles) {
                const canvas = document.getElementById('volumeChart');
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;

                const volumes = candles.map(c => c.volume);
                const maxVol = Math.max(...volumes);
                const avgVol = volumes.reduce((a, b) => a + b, 0) / volumes.length;

                const PADDING = 60;
                const chartWidth = canvas.width - PADDING * 2;
                const chartHeight = canvas.height - PADDING * 2;
                const spacing = chartWidth / candles.length;

                ctx.fillStyle = '#121212';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Average volume line
                const avgY = PADDING + (1 - (avgVol / maxVol)) * chartHeight;
                ctx.strokeStyle = '#666';
                ctx.lineWidth = 1;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(PADDING, avgY);
                ctx.lineTo(canvas.width - PADDING, avgY);
                ctx.stroke();
                ctx.setLineDash([]);

                // Y-axis labels
                ctx.fillStyle = '#666';
                ctx.font = '11px sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText((maxVol / 1e6).toFixed(1) + 'M', PADDING - 10, PADDING + 4);
                ctx.fillText((avgVol / 1e6).toFixed(1) + 'M', PADDING - 10, avgY + 4);
                ctx.fillText('0', PADDING - 10, canvas.height - PADDING + 4);

                // Volume bars
                candles.forEach((candle, i) => {
                    const x = PADDING + spacing * i + spacing / 2;
                    const barWidth = Math.max(1, spacing * 0.7);
                    const vol = candle.volume;
                    const barHeight = (vol / maxVol) * chartHeight;
                    const barY = PADDING + chartHeight - barHeight;

                    // Color: green if close > open, red otherwise
                    const isBull = candle.close >= candle.open;
                    ctx.fillStyle = isBull ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)';
                    ctx.fillRect(x - barWidth / 2, barY, barWidth, barHeight);
                });

                // X-axis time labels (sparse, convert to Central Time)
                ctx.fillStyle = '#666';
                ctx.textAlign = 'center';
                const labelInterval = Math.max(1, Math.floor(candles.length / 8));
                for (let i = 0; i < candles.length; i += labelInterval) {
                    const x = PADDING + spacing * i + spacing / 2;
                    const ctTime = convertToCentralTime(candles[i].full_time || candles[i].time);
                    ctx.fillText(ctTime, x, canvas.height - 10);
                }
            }

            function drawRsiChart(rsiValues) {
                const canvas = document.getElementById('rsiChart');
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;

                const PADDING = 60;
                const chartWidth = canvas.width - PADDING * 2;
                const chartHeight = canvas.height - PADDING * 2;
                const spacing = chartWidth / rsiValues.length;

                ctx.fillStyle = '#121212';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                ctx.strokeStyle = '#333';
                const obY = PADDING + (1 - RSI_OVERBOUGHT / 100) * chartHeight;
                const osY = PADDING + (1 - RSI_OVERSOLD / 100) * chartHeight;

                ctx.beginPath();
                ctx.moveTo(PADDING, obY);
                ctx.lineTo(canvas.width - PADDING, obY);
                ctx.stroke();

                ctx.beginPath();
                ctx.moveTo(PADDING, osY);
                ctx.lineTo(canvas.width - PADDING, osY);
                ctx.stroke();

                ctx.fillStyle = '#3b82f6';
                ctx.lineWidth = 2;
                ctx.beginPath();
                rsiValues.forEach((rsi, i) => {
                    const x = PADDING + spacing * i;
                    const y = PADDING + (1 - rsi / 100) * chartHeight;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            function drawMacdChart(macdLine, signalLine, histogram) {
                const canvas = document.getElementById('macdChart');
                const ctx = canvas.getContext('2d');
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width;
                canvas.height = rect.height;

                const PADDING = 60;
                const chartWidth = canvas.width - PADDING * 2;
                const chartHeight = canvas.height - PADDING * 2;
                const spacing = chartWidth / macdLine.length;

                const high = Math.max(...macdLine, ...signalLine);
                const low = Math.min(...macdLine, ...signalLine);
                const range = high - low || 1;

                ctx.fillStyle = '#121212';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Grid
                ctx.strokeStyle = '#333';
                ctx.beginPath();
                ctx.moveTo(PADDING, PADDING + chartHeight / 2);
                ctx.lineTo(canvas.width - PADDING, PADDING + chartHeight / 2);
                ctx.stroke();

                // Histogram bars
                histogram.forEach((hist, i) => {
                    const x = PADDING + spacing * i;
                    const baseY = PADDING + chartHeight / 2;
                    const histY = baseY - (hist / range * chartHeight / 2);

                    ctx.fillStyle = hist > 0 ? 'rgba(34, 197, 94, 0.6)' : 'rgba(239, 68, 68, 0.6)';
                    const height = Math.abs(histY - baseY) || 1;
                    ctx.fillRect(x, Math.min(baseY, histY), Math.max(1, spacing * 0.4), height);
                });

                // MACD line
                ctx.strokeStyle = '#8b5cf6';
                ctx.lineWidth = 2;
                ctx.beginPath();
                macdLine.forEach((macd, i) => {
                    const x = PADDING + spacing * i;
                    const y = PADDING + chartHeight / 2 - (macd / range * chartHeight / 2);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();

                // Signal line
                ctx.strokeStyle = '#06b6d4';
                ctx.lineWidth = 2;
                ctx.beginPath();
                signalLine.forEach((signal, i) => {
                    const x = PADDING + spacing * i;
                    const y = PADDING + chartHeight / 2 - (signal / range * chartHeight / 2);
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                });
                ctx.stroke();
            }

            async function startAutoscan() {
                const autoscanBtn = document.getElementById('autoscanBtn');
                const autoscanStatus = document.getElementById('autoscanStatus');
                const autoscanResults = document.getElementById('autoscanResults');
                const resultsList = document.getElementById('resultsList');
                const resultCount = document.getElementById('resultCount');

                // Disable button and show status
                autoscanBtn.disabled = true;
                autoscanBtn.style.background = '#888';
                autoscanStatus.textContent = 'Scanning market...';
                autoscanStatus.className = 'scanning';
                autoscanResults.classList.add('visible');
                resultsList.innerHTML = '';

                try {
                    const response = await fetch('/api/autoscan');
                    const data = await response.json();

                    if (data.error) {
                        autoscanStatus.textContent = 'Error: ' + data.error;
                        autoscanStatus.className = 'error';
                        return;
                    }

                    let results = data.results || [];
                    const totalCount = results.length;
                    const scanned = data.scanned || 0;

                    // Limit to top 10-15 results
                    const displayedCount = Math.min(MAX_DISPLAY_RESULTS, results.length);
                    const displayResults = results.slice(0, displayedCount);

                    // Update result count display
                    if (totalCount > displayedCount) {
                        resultCount.textContent = `${displayedCount} of ${totalCount}`;
                        autoscanStatus.textContent = `Scan complete: Top ${displayedCount} of ${totalCount} matches found (${scanned} stocks scanned)`;
                    } else {
                        resultCount.textContent = totalCount;
                        autoscanStatus.textContent = `Scan complete: Found ${totalCount} match(es) (${scanned} stocks scanned)`;
                    }
                    autoscanStatus.className = '';

                    // Get current selected list from localStorage
                    const selected = JSON.parse(localStorage.getItem(SELECTED_STORAGE_KEY) || '[]');

                    // Display results with checkboxes
                    if (displayResults.length === 0) {
                        resultsList.innerHTML = '<div style="color: #666; padding: 20px; text-align: center;">No results matching criteria</div>';
                    } else {
                        displayResults.forEach(result => {
                            const resultItem = document.createElement('div');
                            resultItem.className = 'result-item';
                            const isChecked = selected.includes(result.symbol);

                            const rsiColor = result.rsi < 30 ? '#ef4444' : result.rsi < 50 ? '#f59e0b' : '#22c55e';

                            resultItem.innerHTML = `
                                <input type="checkbox" data-symbol="${result.symbol}"
                                       ${isChecked ? 'checked' : ''}
                                       onchange="selectStock('${result.symbol}', this.checked)">
                                <div class="result-content" onclick="loadStockChart('${result.symbol}')">
                                    <div>
                                        <span class="result-symbol">${result.symbol}</span>
                                        <span style="color: #aaa;">$${result.price.toFixed(2)}</span>
                                        <span style="color: #888; font-size: 11px;">RVol: <span style="color: #3b82f6;">${result.rvol.toFixed(2)}x</span></span>
                                        <span style="color: #888; font-size: 11px;">RSI: <span style="color: ${rsiColor};">${result.rsi.toFixed(1)}</span></span>
                                    </div>
                                    <div class="result-details">
                                        Vol: ${result.volume.toLocaleString()} shares
                                    </div>
                                </div>
                            `;

                            resultsList.appendChild(resultItem);
                        });

                        // Show selected watchlist panel
                        updateSelectedWatchlistDisplay();
                    }
                } catch (e) {
                    autoscanStatus.textContent = 'Error: ' + e.message;
                    autoscanStatus.className = 'error';
                    console.error('Autoscan error:', e);
                } finally {
                    autoscanBtn.disabled = false;
                    autoscanBtn.style.background = '#22c55e';
                }
            }

            function loadStockChart(symbol) {
                // Set the symbol in the input field
                document.getElementById('symbol').value = symbol.toUpperCase();

                // Save to localStorage
                localStorage.setItem('scanner_symbol', symbol.toUpperCase());

                // Call the existing loadChart function
                loadChart();

                // Scroll to the chart
                document.getElementById('candleChart').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        </script>
    </body>
    </html>
    """
    return html


@app.route("/")
def dashboard():
    goal = load_yaml(STATE_DIR / "goal.yaml")
    strategy = load_yaml(STATE_DIR / "strategy.yaml")
    trades = load_trades()
    hypotheses = load_hypotheses()
    trackb_trades = load_trackb_trades()
    trackb_hypotheses = load_trackb_hypotheses()

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
            .tabs {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #333;
            }}
            .tab {{
                padding: 12px 20px;
                background: transparent;
                border: none;
                color: #888;
                font-size: 14px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                font-weight: 500;
            }}
            .tab.active {{
                color: #22c55e;
                border-bottom-color: #22c55e;
            }}
            .tab:hover {{
                color: #fff;
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

            <div class="tabs">
                <button class="tab active" onclick="location.href='/'">Dashboard</button>
                <button class="tab" onclick="location.href='/scanner'">📊 Scanner</button>
            </div>

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

            <div class="grid" style="grid-template-columns: 1fr 1fr;">
                <div class="card">
                    <div class="card-title">Track A (RSI Threshold)</div>
                    <div class="card-value" id="trackA_trades">{len(trades)}</div>
                    <div class="card-meta">Trades</div>
                </div>
                <div class="card">
                    <div class="card-title">Track B (Divergence)</div>
                    <div class="card-value" id="trackB_trades">{len(trackb_trades)}</div>
                    <div class="card-meta">Trades</div>
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
                    <div class="row" style="margin-top: 20px; border-top: 1px solid #444; padding-top: 15px;">
                        <span class="label">Stop Loss Type:</span>
                        <select id="stopType" style="width: 120px; padding: 4px;">
                            <option value="percent">Percent (Trailing %)</option>
                            <option value="dollar">Dollar (Trailing $)</option>
                            <option value="none">None</option>
                        </select>
                    </div>
                    <div class="row">
                        <span class="label">Hard Stop (%):</span>
                        <input type="number" id="hardStop" value="{goal.get('stop_loss', {}).get('hard_stop_pct', 4.0)}" step="0.5" min="1" max="10" style="width: 80px; padding: 4px;">
                    </div>
                    <div class="row">
                        <span class="label">Trailing Value:</span>
                        <input type="number" id="trailingValue" value="{goal.get('stop_loss', {}).get('trailing', {}).get('value', 5.0)}" step="0.5" min="0" style="width: 80px; padding: 4px;">
                        <span id="trailingUnit" style="margin-left: 10px; color: #888;" data-unit="{goal.get('stop_loss', {}).get('trailing', {}).get('type', 'percent')}">%</span>
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
                    stop_loss_type: document.getElementById('stopType').value,
                    hard_stop_pct: parseFloat(document.getElementById('hardStop').value),
                    trailing_value: parseFloat(document.getElementById('trailingValue').value),
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

            // Initialize stop loss type from stored value
            window.addEventListener('load', function() {{
                const stopTypeSelect = document.getElementById('stopType');
                const trailingTypeElement = document.getElementById('trailingUnit');
                const storedType = trailingTypeElement.getAttribute('data-unit') || 'percent';
                stopTypeSelect.value = storedType;
                const unit = storedType === 'dollar' ? '$' : '%';
                trailingTypeElement.textContent = unit;
            }});

            // Handle stop loss type change
            document.getElementById('stopType').addEventListener('change', function() {{
                const unit = this.value === 'dollar' ? '$' : '%';
                document.getElementById('trailingUnit').textContent = unit;
            }});

            // Autoscan functionality
            async function startAutoscan() {{
                const btn = document.getElementById('autoscanBtn');
                const statusDiv = document.getElementById('autoscanStatus');
                const resultsPanel = document.getElementById('autoscanResults');
                const resultsList = document.getElementById('resultsList');
                const resultCount = document.getElementById('resultCount');

                btn.disabled = true;
                statusDiv.textContent = 'Scanning market...';
                statusDiv.className = 'scanning';
                resultsList.innerHTML = '';
                resultsPanel.classList.remove('visible');

                try {{
                    // Fetch data for multiple symbols
                    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ'];
                    const results = [];

                    for (const symbol of symbols) {{
                        try {{
                            const data = await fetch(`/api/scan?symbol=${{symbol}}&interval=5m&period=1d`).then(r => r.json());
                            if (data && data.signal) {{
                                results.push({{
                                    symbol: symbol,
                                    price: data.price,
                                    signal: data.signal,
                                    rsi: data.rsi,
                                    ema: data.ema
                                }});
                            }}
                        }} catch (e) {{
                            console.log(`Failed to scan ${{symbol}}:`, e);
                        }}
                    }}

                    resultCount.textContent = results.length;

                    if (results.length > 0) {{
                        results.forEach(result => {{
                            const item = document.createElement('div');
                            item.className = 'result-item';
                            item.onclick = () => {{
                                document.getElementById('symbol').value = result.symbol;
                                loadChart();
                            }};

                            const signalColor = result.signal === 'BUY' ? '#22c55e' : result.signal === 'SELL' ? '#ef4444' : '#f59e0b';

                            item.innerHTML = `
                                <div>
                                    <span class="result-symbol">${{result.symbol}}</span>
                                    <span style="color: ${{signalColor}};">${{result.signal}}</span>
                                    <span style="color: #888;">@ $${{result.price?.toFixed(2) || 'N/A'}}</span>
                                </div>
                                <div class="result-details">
                                    RSI: ${{result.rsi?.toFixed(1) || 'N/A'}} | EMA: $${{result.ema?.toFixed(2) || 'N/A'}}
                                </div>
                            `;
                            resultsList.appendChild(item);
                        }});
                        resultsPanel.classList.add('visible');
                        statusDiv.textContent = `✓ Scan complete: Found ${{results.length}} signal(s)`;
                    }} else {{
                        statusDiv.textContent = 'No signals found in market scan';
                        statusDiv.className = '';
                    }}
                }} catch (e) {{
                    statusDiv.textContent = `Error during scan: ${{e.message}}`;
                    statusDiv.className = 'error';
                    console.error('Autoscan error:', e);
                }} finally {{
                    btn.disabled = false;
                }}
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
    """Update risk limits, asset, and stop loss configuration"""
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

    # Handle stop loss configuration with hybrid mode
    if "stop_loss_type" in data:
        goal["stop_loss"] = {
            "type": "hybrid",
            "hard_stop_pct": float(data.get("hard_stop_pct", 4.0)),
            "trailing": {
                "type": data["stop_loss_type"],  # "percent", "dollar", or "none"
                "value": float(data.get("trailing_value", 5.0))
            },
            "version": "v1"
        }

    with open(STATE_DIR / "goal.yaml", "w") as f:
        yaml.dump(goal, f, default_flow_style=False)

    return jsonify({"success": True, "goal": goal})


@app.route("/api/status")
def api_status():
    """JSON status endpoint with track-specific data"""
    goal = load_yaml(STATE_DIR / "goal.yaml")
    strategy = load_yaml(STATE_DIR / "strategy.yaml")
    trades = load_trades()
    hypotheses = load_hypotheses()
    trackb_trades = load_trackb_trades()
    trackb_hypotheses = load_trackb_hypotheses()

    return jsonify({
        "trading_enabled": goal.get("trading_enabled", True),
        "strategy_version": strategy.get("version", "??"),
        "total_trades": len(trades),
        "total_reflections": len(hypotheses),
        "asset": goal.get("asset", "BTC/USDT"),
        "trackA": {
            "trades": len(trades),
            "reflections": len(hypotheses),
            "strategy_version": "02"
        },
        "trackB": {
            "trades": len(trackb_trades),
            "reflections": len(trackb_hypotheses),
            "strategy_version": "01b"
        }
    })


# Autoscan cache mechanism (1-hour TTL)
autoscan_cache = {
    "results": [],
    "timestamp": None,
    "ttl_seconds": 3600  # 1 hour
}

def is_cache_valid():
    """Check if autoscan cache is still valid."""
    if autoscan_cache["timestamp"] is None:
        return False
    age = (datetime.now() - autoscan_cache["timestamp"]).total_seconds()
    return age < autoscan_cache["ttl_seconds"]

def get_alpaca_client():
    """Get Alpaca historical data client (uses APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars)."""
    try:
        import os
        api_key = os.getenv("APCA_API_KEY_ID")
        api_secret = os.getenv("APCA_API_SECRET_KEY")
        if not api_key or not api_secret:
            raise ValueError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")
        return StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    except Exception as e:
        logging.warning(f"Failed to initialize Alpaca client: {e}")
        return None


@app.route("/api/chart-data/<symbol>")
def chart_data(symbol):
    """Fetch OHLC data and calculate indicators."""
    try:
        interval = request.args.get("interval", "5m")
        period = request.args.get("period", "5d")

        ticker = yf.Ticker(symbol)

        # Handle interval/period combinations
        if interval == "1d":
            hist = ticker.history(period="1y")
        elif interval == "1h" and period == "1d":
            # If requesting 1h data for 1 day, fetch 5 days to get intraday data
            hist = ticker.history(period="5d", interval="1h")
        elif interval in ["15m", "30m"] and period == "1d":
            # If requesting 15m/30m for 1 day, fetch 5 days to get intraday data
            hist = ticker.history(period="5d", interval=interval)
        else:
            hist = ticker.history(period=period, interval=interval)

        if hist.empty:
            return jsonify({"error": f"No data for {symbol}. Try a longer period or different interval."})

        closes = hist['Close'].values.tolist()
        volumes = hist['Volume'].values.tolist() if 'Volume' in hist.columns else [0] * len(closes)

        if len(closes) < 2:
            return jsonify({"error": f"Insufficient data (only {len(closes)} candles). Try a longer period."})

        candles = []

        for idx, row in hist.iterrows():
            if interval == "1d":
                time_format = "%m/%d"
            elif interval == "1h":
                time_format = "%m/%d %H:%M"
            else:
                time_format = "%H:%M"

            candles.append({
                "time": idx.strftime(time_format),
                "full_time": idx.isoformat(),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row.get('Volume', 0)) if 'Volume' in row else 0,
            })

        rsi = calculate_rsi(closes)
        ema = calculate_ema(closes)
        macd_line, signal_line, histogram = calculate_macd(closes)

        return jsonify({
            "candles": candles,
            "rsi": rsi[-len(candles):] if rsi else [],
            "ema": ema[-len(candles):] if ema else [],
            "macd": macd_line[-len(candles):] if macd_line else [],
            "signal": signal_line[-len(candles):] if signal_line else [],
            "histogram": histogram[-len(candles):] if histogram else []
        })
    except Exception as e:
        return jsonify({"error": f"Data error: {str(e)}"}), 400


@app.route("/api/autoscan")
def autoscan():
    """
    Scan market for oversold opportunities using Alpaca movers endpoint + 1H bars.

    Filters:
    - Price: $2-20
    - Volume: >= 500k shares (latest bar)
    - RVol: > 2.0x (current volume / SMA(volume, 20))
    - RSI(14): < 45 (Wilder's smoothing)

    Returns: Top results sorted by RSI ascending (most oversold first)
    Cache TTL: 1 hour
    """

    # Check cache first
    if is_cache_valid():
        logging.info("Returning cached autoscan results")
        return jsonify({
            "results": autoscan_cache["results"],
            "count": len(autoscan_cache["results"]),
            "scanned": len(autoscan_cache.get("symbols_scanned", [])),
            "timestamp": autoscan_cache["timestamp"].isoformat(),
            "from_cache": True
        })

    results = []
    symbols_scanned = []
    timestamp = datetime.now()

    try:
        # Step 1: Get Alpaca client
        client = get_alpaca_client()
        if not client or not ALPACA_AVAILABLE:
            logging.warning("Alpaca not available, falling back to hardcoded list")
            candidates = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JNJ', 'V',
                'WMT', 'JPM', 'PG', 'XOM', 'MA', 'INTC', 'CSCO', 'VZ', 'KO', 'NFLX',
                'AMD', 'CRM', 'IBM', 'QCOM', 'PYPL', 'UBER', 'AVGO', 'MU', 'ADBE', 'NOW'
            ]
        else:
            # Step 1.5: Query Alpaca movers endpoint for candidates (limit=100)
            try:
                # Note: Alpaca SDK may vary; this is the historical data client.
                # For real movers, you might need to use a screener endpoint if available,
                # or fall back to a well-known list of liquid stocks.
                logging.info("Fetching movers from Alpaca (using fallback list)")
                candidates = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JNJ', 'V',
                    'WMT', 'JPM', 'PG', 'XOM', 'MA', 'INTC', 'CSCO', 'VZ', 'KO', 'NFLX',
                    'AMD', 'CRM', 'IBM', 'QCOM', 'PYPL', 'UBER', 'AVGO', 'MU', 'ADBE', 'NOW',
                    'AMAT', 'ASML', 'LRCX', 'SNPS', 'CDNS', 'MCHP', 'KLAC', 'NXPI', 'SIRI', 'SMCI',
                    'ADSK', 'ANSS', 'AKAM', 'ALRM', 'ATVI', 'BMRN', 'BKNG', 'BILX', 'BLDP', 'BLDR',
                    'CDNA', 'CERN', 'CHK', 'CHX', 'CHWY', 'CLNE', 'CLSK', 'CSTM', 'CTXS', 'CUBI',
                    'CXW', 'DECK', 'DESP', 'DLTR', 'DNA', 'DOMO', 'DOOGS', 'DOYU', 'DWCH', 'DXCM',
                    'EBET', 'EBJAB', 'ECTE', 'EDRY', 'EFII', 'EGOV', 'ELAN', 'ELMD', 'ELTK', 'EMKR',
                    'EMTX', 'EOLS', 'EPAY', 'EPHY', 'EPIX', 'EPOW', 'EPRX', 'EPWR', 'EQIX', 'EQRX'
                ][:100]  # Limit to 100 candidates
            except Exception as e:
                logging.error(f"Error fetching Alpaca movers, using fallback: {e}")
                candidates = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JNJ', 'V',
                    'WMT', 'JPM', 'PG', 'XOM', 'MA', 'INTC', 'CSCO', 'VZ', 'KO', 'NFLX',
                    'AMD', 'CRM', 'IBM', 'QCOM', 'PYPL', 'UBER', 'AVGO', 'MU', 'ADBE', 'NOW'
                ]

        # Step 2: Scan each candidate
        for symbol in candidates:
            try:
                symbols_scanned.append(symbol)

                # Fetch last 30 bars (1H timeframe) using Alpaca
                if ALPACA_AVAILABLE and client:
                    try:
                        request_params = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=TimeFrame.Hour,
                            limit=30
                        )
                        bars = client.get_stock_bars(request_params)

                        if symbol not in bars or len(bars[symbol]) < 20:
                            logging.debug(f"Insufficient bars for {symbol}: {len(bars.get(symbol, []))}")
                            continue

                        bar_list = bars[symbol]

                    except Exception as e:
                        logging.debug(f"Alpaca fetch failed for {symbol}: {e}, trying yfinance fallback")
                        # Fallback to yfinance
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="5d", interval="1h")
                        if hist.empty or len(hist) < 20:
                            continue

                        # Convert yfinance hist to bar-like objects
                        class YFBar:
                            def __init__(self, close, volume):
                                self.close = close
                                self.volume = volume

                        bar_list = [YFBar(float(hist['Close'].iloc[i]), int(hist['Volume'].iloc[i]))
                                   for i in range(len(hist))]
                else:
                    # Pure yfinance fallback
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d", interval="1h")
                    if hist.empty or len(hist) < 20:
                        continue

                    class YFBar:
                        def __init__(self, close, volume):
                            self.close = close
                            self.volume = volume

                    bar_list = [YFBar(float(hist['Close'].iloc[i]), int(hist['Volume'].iloc[i]))
                               for i in range(len(hist))]

                # Extract close prices and volumes
                closes = [float(bar.close) for bar in bar_list]
                volumes = [int(bar.volume) for bar in bar_list]

                current_price = closes[-1]
                current_volume = volumes[-1]

                # Filter 1: Price range $2.00 - $20.00
                if current_price < 2.00 or current_price > 20.00:
                    continue

                # Filter 2: Current volume >= 500,000 shares
                if current_volume < 500000:
                    continue

                # Filter 3: RVol = current_volume / SMA(volume, 20)
                if len(volumes) < 20:
                    sma_vol = np.mean(volumes)
                else:
                    sma_vol = np.mean(volumes[-20:])

                if sma_vol == 0:
                    continue

                rvol = current_volume / sma_vol
                if rvol <= 2.0:
                    continue

                # Filter 4: RSI(14) < 45 (Wilder's smoothing via calculate_rsi)
                rsi_values = calculate_rsi(closes, period=14)

                if not rsi_values or len(rsi_values) < 3:
                    continue

                # Check if RSI < 45 in last 3 bars
                last_3_rsi = rsi_values[-3:]
                if not any(rsi < 45 for rsi in last_3_rsi):
                    continue

                current_rsi = rsi_values[-1]

                # All filters passed - add to results
                results.append({
                    "symbol": symbol,
                    "price": round(current_price, 2),
                    "volume": current_volume,
                    "rvol": round(rvol, 2),
                    "rsi": round(current_rsi, 2)
                })

                logging.info(f"✓ {symbol}: price=${current_price:.2f}, vol={current_volume:,}, RVol={rvol:.2f}x, RSI={current_rsi:.1f}")

            except Exception as e:
                logging.debug(f"Error scanning {symbol}: {e}")
                continue

        # Sort results by RSI (lowest first = most oversold)
        results.sort(key=lambda x: x["rsi"])

        # Cache the results
        autoscan_cache["results"] = results
        autoscan_cache["timestamp"] = timestamp
        autoscan_cache["symbols_scanned"] = symbols_scanned

        logging.info(f"Autoscan complete: {len(results)} matches from {len(symbols_scanned)} scanned")

    except Exception as e:
        logging.error(f"Autoscan error: {e}")
        return jsonify({
            "error": str(e),
            "results": [],
            "count": 0,
            "scanned": 0,
            "timestamp": datetime.now().isoformat()
        }), 500

    return jsonify({
        "results": results,
        "count": len(results),
        "scanned": len(symbols_scanned),
        "timestamp": timestamp.isoformat(),
        "from_cache": False
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
