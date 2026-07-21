FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY hermes_trading ./hermes_trading
COPY state ./state
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    pyyaml \
    httpx \
    aiofiles \
    numpy \
    pandas \
    rich \
    flask \
    alpaca-trade-api
ENV HERMES_TRADING_MODE=paper
ENV HERMES_TRADING_I_ACCEPT_RISK=false
EXPOSE 5000
CMD ["python", "-m", "hermes_trading.run"]
