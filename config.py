"""
config.py
=========

Central configuration for the AI Portfolio Advisory System.

Single source of truth for every magic number, path, ticker, and threshold
in the project. No other module should hard-code parameters — they should
all read from here. This makes the system easy to tune from one place and
keeps experiments reproducible.

Sections
--------
1. Paths
2. Universe + benchmark
3. Date range
4. Indicator parameters
5. Signal thresholds
6. Backtest parameters
7. Data quality gates
8. Cache settings
9. Sector mapping
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROC_DIR: Path = DATA_DIR / "processed"
LOG_DIR: Path = BASE_DIR / "logs"

for _p in (DATA_DIR, RAW_DIR, PROC_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Universe + benchmark
# ---------------------------------------------------------------------------
# 25 large-cap NSE stocks across 8 sectors. The .NS suffix is the yfinance
# convention for National Stock Exchange of India tickers.
UNIVERSE: list[str] = [
    # Information Technology
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    # Banking & Financial Services
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    # Energy
    "RELIANCE.NS", "ONGC.NS", "POWERGRID.NS",
    # FMCG
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
    # Automobiles
    # NOTE: Tata Motors demerged into separate Passenger Vehicles (TMPV) and
    # Commercial Vehicles (TMCV) entities, leaving the legacy TATAMOTORS.NS
    # with empty yfinance history. Replaced with the two new tickers.
    "MARUTI.NS", "TMPV.NS", "TMCV.NS", "M&M.NS",
    # Pharma
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    # Metals
    "TATASTEEL.NS", "HINDALCO.NS",
    # Telecom
    "BHARTIARTL.NS",
]

# NIFTY 50 — the standard benchmark for Indian large-cap equities.
# Every alpha/beta/outperformance calculation references this.
BENCHMARK: str = "^NSEI"


# ---------------------------------------------------------------------------
# 3. Date range
# ---------------------------------------------------------------------------
# Six years gives us enough data to span at least one bear-bull cycle
# (the COVID crash of Mar-2020 + subsequent recovery + 2022 correction).
START_DATE: str = "2019-01-01"
# Auto-extends to today so the pipeline always covers the most recent bars.
# Override in the sidebar if you want a frozen end date for reproducible runs.
END_DATE: str = date.today().isoformat()


# ---------------------------------------------------------------------------
# 4. Indicator parameters
# ---------------------------------------------------------------------------
# Trend MAs — 20/50/200 are the standard short/medium/long-term lookbacks
# used by virtually every desk and trading platform.
MA_SHORT: int = 20
MA_MEDIUM: int = 50
MA_LONG: int = 200

# EMA — exponential MAs weight recent prices more heavily. 12/26 are the
# MACD defaults popularised by Gerald Appel.
EMA_FAST: int = 12
EMA_SLOW: int = 26

# MACD — Moving Average Convergence Divergence. The 9-day EMA of the
# MACD line is the "signal" line. Histogram = MACD − Signal.
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# RSI — Wilder's 14-period default. Above 70 = overbought, below 30 = oversold.
RSI_PERIOD: int = 14

# Bollinger Bands — 20-day SMA with ±2 standard-deviation envelope.
BB_PERIOD: int = 20
BB_STD: float = 2.0

# Stochastic Oscillator — fast %K, slow %D smoothing.
STOCH_K: int = 14
STOCH_D: int = 3

# ATR — Average True Range, the canonical volatility measure for stop-losses.
ATR_PERIOD: int = 14

# Rate of Change lookback.
ROC_PERIOD: int = 10

# Volatility window — rolling std dev of daily returns (annualised later).
VOL_WINDOW: int = 20

# Volume MA — to detect when volume is unusually high.
VOLUME_MA_PERIOD: int = 20


# ---------------------------------------------------------------------------
# 5. Signal thresholds
# ---------------------------------------------------------------------------
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0

# Sub-engine voting weights — must sum to 1.0. Trend and Momentum carry the
# most weight because the system is biased toward trend-following with
# momentum confirmation. Mean-Reversion and Volume are filters/tiebreakers.
SIGNAL_WEIGHTS: dict[str, float] = {
    "trend": 0.30,
    "momentum": 0.30,
    "mean_reversion": 0.20,
    "volume": 0.20,
}

# Regime gates — applied after the composite score is computed. The gates
# (and position-size multipliers) embed the core philosophy: "buy the dips
# in uptrends, take profits in downtrends, do less in chop."
REGIME_GATES: dict[str, dict[str, float]] = {
    "BULL":     {"buy_gate": 0.45, "sell_gate": 0.60, "size_mult": 1.00},
    "SIDEWAYS": {"buy_gate": 0.55, "sell_gate": 0.55, "size_mult": 0.60},
    "BEAR":     {"buy_gate": 0.75, "sell_gate": 0.45, "size_mult": 0.80},
}


# ---------------------------------------------------------------------------
# 6. Backtest parameters
# ---------------------------------------------------------------------------
INITIAL_CAPITAL: float = 100_000.0       # ₹1,00,000 starting equity
TRANSACTION_COST: float = 0.001          # 0.1% per trade (STT + brokerage)
POSITION_SIZE_PCT: float = 0.10          # Cap any single position at 10% NAV
MAX_OPEN_POSITIONS: int = 8              # Cap on concurrent open positions
BASE_ALLOCATION: float = 0.05            # 5% base allocation, scaled by confidence

# India 10-year G-sec is the standard risk-free proxy for Indian markets.
# Used in Sharpe, Sortino, Jensen's Alpha.
RISK_FREE_RATE: float = 0.065

# Trading-day count for annualisation (NSE).
TRADING_DAYS: int = 252


# ---------------------------------------------------------------------------
# 7. Data quality gates
# ---------------------------------------------------------------------------
MIN_TRADING_DAYS: int = 100              # Reject tickers with <100 bars
# Raised from 5% to 12% to tolerate ticker-specific data gaps caused by
# corporate actions (e.g. TATAMOTORS.NS had a 2024 DVR-to-ordinary merger
# that left yfinance with sparse history around the conversion date).
# MIN_TRADING_DAYS lowered to 100 so post-demerger new listings (TMPV.NS,
# TMCV.NS) with only ~5-6 months of history still load.
MAX_MISSING_PCT: float = 0.12
MAX_DAILY_RETURN: float = 0.50           # Flag any single-day move >50% as suspect


# ---------------------------------------------------------------------------
# 8. Cache settings
# ---------------------------------------------------------------------------
CACHE_ENABLED: bool = True
CACHE_FORMAT: str = "csv"                # "csv" or "parquet"
# Refetch if cache is older than this. Set to 1 so historical data refreshes
# daily — combined with END_DATE = today, this keeps the panel current.
CACHE_MAX_AGE_DAYS: int = 1

# Live (delayed-intraday) quote cache TTL in seconds. yfinance fast_info is
# ~15-20 minutes delayed for NSE; 60s of in-process caching is plenty to
# keep the dashboard snappy without hammering Yahoo.
LIVE_QUOTE_TTL_SECONDS: int = 60


# ---------------------------------------------------------------------------
# 9. Sector mapping
# ---------------------------------------------------------------------------
# Used by PortfolioAnalyzer for sector exposure and concentration checks.
NSE_SECTORS: dict[str, str] = {
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT", "TECHM.NS": "IT",

    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking", "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking",

    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "POWERGRID.NS": "Energy",

    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",

    "MARUTI.NS": "Auto", "TMPV.NS": "Auto", "TMCV.NS": "Auto",
    "TATAMOTORS.NS": "Auto",                  # legacy alias (may be empty)
    "M&M.NS": "Auto",

    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",

    "TATASTEEL.NS": "Metals", "HINDALCO.NS": "Metals",

    "BHARTIARTL.NS": "Telecom",
}


# ---------------------------------------------------------------------------
# Regime label conventions — referenced across regime.py, signals.py,
# backtest.py, recommend.py.
# ---------------------------------------------------------------------------
REGIME_LABELS: dict[int, str] = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}
REGIME_CODES: dict[str, int] = {v: k for k, v in REGIME_LABELS.items()}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
