"""
AI Portfolio Advisory System — `src` package.

Submodules (all public):
    data_loader  — fetches & caches OHLCV from yfinance
    features     — 32 technical indicators across 6 groups
    regime       — BEAR/SIDEWAYS/BULL labeller (MA / KMeans / HMM)
    signals      — 4-engine voting → regime-gated BUY/SELL/HOLD
    backtest     — event-driven portfolio simulator + tearsheet
    portfolio    — risk analyser for user holdings
    recommend    — ranked Buy / Exit suggestions for the universe
"""

__version__ = "0.1.0"
