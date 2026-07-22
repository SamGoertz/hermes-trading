"""Web dashboard for monitoring trading agent."""
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request
import yaml
import yfinance as yf
import numpy as np

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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="tabs">
                <button class="tab" onclick="location.href='/'">Dashboard</button>
                <button class="tab active" onclick="location.href='/scanner'">📊 Scanner</button>
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
                        <option value="1d">1 Day</option>
                        <option value="5d" selected>5 Days</option>
                        <option value="1mo">1 Month</option>
                        <option value="3mo">3 Months</option>
                        <option value="1y">1 Year</option>
                    </select>
                    <button onclick="loadChart()">Load Chart</button>
                </div>
            </div>

            <div id="error" class="error" style="display: none;"></div>

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
            </div>
        </div>

        <script>
            const RSI_OVERBOUGHT = 70;
            const RSI_OVERSOLD = 30;

            async function loadChart() {
                const symbol = document.getElementById('symbol').value.toUpperCase();
                const interval = document.getElementById('interval').value;
                const period = document.getElementById('period').value;
                const errorDiv = document.getElementById('error');
                errorDiv.style.display = 'none';

                try {
                    const response = await fetch(`/api/chart-data/${symbol}?interval=${interval}&period=${period}`);
                    if (!response.ok) throw new Error('Failed to fetch data');

                    const data = await response.json();
                    if (data.error) throw new Error(data.error);

                    const candles = data.candles;
                    const rsiValues = data.rsi;
                    const emaValues = data.ema;
                    const macdLine = data.macd;
                    const signalLine = data.signal;
                    const histogram = data.histogram;

                    // Update title
                    const intervalLabel = interval.replace('m', ' min').replace('h', ' hr').replace('d', ' day');
                    document.getElementById('candleTitle').textContent = `Candlestick (${intervalLabel}) + EMA(9)`;

                    document.getElementById('priceValue').textContent = '$' + candles[candles.length - 1].close.toFixed(2);
                    document.getElementById('priceStatus').textContent = candles[candles.length - 1].time;

                    const rsi = rsiValues[rsiValues.length - 1];
                    document.getElementById('rsiValue').textContent = rsi.toFixed(1);
                    document.getElementById('rsiStatus').textContent =
                        rsi > RSI_OVERBOUGHT ? '⚠️ Overbought' : rsi < RSI_OVERSOLD ? '📍 Oversold' : 'Neutral';

                    const ema = emaValues[emaValues.length - 1];
                    const price = candles[candles.length - 1].close;
                    document.getElementById('emaValue').textContent = '$' + ema.toFixed(2);
                    const diff = ((price - ema) / ema * 100).toFixed(2);
                    document.getElementById('emaStatus').textContent = (diff > 0 ? '+' : '') + diff + '%';

                    const macd = macdLine[macdLine.length - 1];
                    const signal = signalLine[signalLine.length - 1];
                    const hist = histogram[histogram.length - 1];
                    document.getElementById('macdValue').textContent = macd.toFixed(4);
                    document.getElementById('macdStatus').textContent = macd > 0 ? '📈 Positive' : '📉 Negative';
                    document.getElementById('signalValue').textContent = signal.toFixed(4);
                    document.getElementById('signalStatus').textContent = macd > signal ? '🟢 Above' : '🔴 Below';
                    document.getElementById('histValue').textContent = hist.toFixed(4);
                    document.getElementById('histStatus').textContent = hist > 0 ? '⬆️ Bullish' : '⬇️ Bearish';

                    drawCandleChart(candles, emaValues);
                    drawRsiChart(rsiValues);
                    drawMacdChart(macdLine, signalLine, histogram);
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

                // X-axis time labels
                ctx.fillStyle = '#666';
                ctx.textAlign = 'center';
                const labelInterval = Math.max(1, Math.floor(candles.length / 8));
                for (let i = 0; i < candles.length; i += labelInterval) {
                    const x = PADDING + spacing * i + spacing / 2;
                    ctx.fillText(candles[i].time, x, canvas.height - 10);
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

                    // Tooltip
                    const tooltip = document.getElementById('tooltipPrice');
                    tooltip.textContent = `$${price.toFixed(2)} | ${nearestCandle.time}`;
                    tooltip.style.left = (mouseX - canvasRect.left + 10) + 'px';
                    tooltip.style.top = (mouseY - canvasRect.top - 30) + 'px';
                });

                canvas.addEventListener('mouseleave', () => {
                    document.getElementById('crosshairs').style.display = 'none';
                });
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

            window.addEventListener('load', loadChart);
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


@app.route("/api/chart-data/<symbol>")
def chart_data(symbol):
    """Fetch OHLC data and calculate indicators."""
    try:
        interval = request.args.get("interval", "5m")
        period = request.args.get("period", "5d")

        ticker = yf.Ticker(symbol)

        # For daily charts, use period without interval to get cleaner data
        if interval == "1d":
            hist = ticker.history(period="1y")  # Get 1 year of daily data
        else:
            hist = ticker.history(period=period, interval=interval)

        if hist.empty:
            return jsonify({"error": f"No data for {symbol}"})

        closes = hist['Close'].values.tolist()
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
                "full_time": idx.isoformat(),  # Store full ISO time for crosshairs
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
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
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
