"""
src/features.py
===============

Phase 2 — Feature engineering.

Takes the 5-column OHLCV bars produced by ``data_loader`` and enriches
them with **32 technical indicators across 6 groups**, returning a wide
DataFrame the signal engine can vote on.

Design principle: **features are *facts* about the data, signals are
*decisions* about what to do.** They are deliberately separated so the
signal engine can be rewritten without re-deriving the math, and so an
analyst can poke at any indicator in isolation.

Indicator groups
----------------
1.  **Trend**          (10 columns)  — MAs, EMAs, MACD, crossover booleans.
2.  **Momentum**       (4 columns)   — RSI, ROC, Stochastic %K/%D.
3.  **Volatility**     (7 columns)   — Bollinger Bands, %B, ATR, ann. vol.
4.  **Volume**         (3 columns)   — OBV, vol MA, vol ratio.
5.  **Returns**        (4 columns)   — simple/log, 5d, 20d.
6.  **Price structure**(4 columns)   — H-L%, gap%, above-MA200 flags.

Net: 5 (OHLCV) + 32 (indicators) = **37 columns**.

Why these specific indicators
-----------------------------
- **MAs (20/50/200)**: the universal short/medium/long-term lens used by
  every desk. The 50/200 crossover ("golden cross" / "death cross") is
  the canonical trend-regime indicator.
- **MACD**: Trend + momentum in one. Histogram-flips lead price by ~2 bars
  on average and are early warnings.
- **RSI**: Bounded oscillator (0–100) — the only momentum gauge that
  doesn't blow up when prices drift.
- **Bollinger Bands**: A volatility-adjusted price envelope. BB_Width
  contractions (squeeze) often precede breakouts; %B (BB_Pct) tells you
  where price sits inside the band.
- **ATR**: The gold-standard volatility unit for stop-loss placement
  (Turtle Traders, Van Tharp, etc.). Used in :mod:`signals` to size stops.
- **OBV**: Cumulative signed volume — confirms whether a price move is
  backed by participation.

We use the `ta` library where it gives us exactly what we want, but
compute simple things (MAs, returns, booleans) directly in pandas to
keep the dependency surface small and the math transparent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator
except ImportError as e:  # pragma: no cover
    raise ImportError("`ta` is required. Install with: pip install ta") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

logging.basicConfig(level=C.LOG_LEVEL, format=C.LOG_FORMAT)
logger = logging.getLogger("features")


class FeatureEngineer:
    """Compute 32 indicators on OHLCV data.

    Methods
    -------
    compute(df)            → enriched DataFrame for a single ticker
    compute_universe(data) → dict[ticker, enriched DataFrame]

    All methods return *copies* — input DataFrames are never mutated.
    """

    # Indicator columns this engine produces (excludes OHLCV).
    INDICATOR_COLUMNS: list[str] = [
        # Trend
        "MA_20", "MA_50", "MA_200", "EMA_12", "EMA_26",
        "MACD", "MACD_Signal", "MACD_Hist",
        "MA20_above_MA50", "MA50_above_MA200",
        # Momentum
        "RSI_14", "ROC_10", "Stoch_K", "Stoch_D",
        # Volatility
        "BB_Upper", "BB_Mid", "BB_Lower", "BB_Width", "BB_Pct",
        "ATR_14", "Volatility_20",
        # Volume
        "OBV", "Vol_MA_20", "Vol_Ratio",
        # Returns (Daily_Return + Adj_Return already added by data_loader)
        "Return_5d", "Return_20d",
        # Price structure
        "High_Low_Pct", "Gap_Pct", "Close_above_MA200", "Dist_from_MA200_pct",
    ]

    # ------------------------------------------------------------------
    # Single-ticker pipeline
    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 32 indicators to a single ticker's OHLCV DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain Open, High, Low, Close, Volume — and ideally
            Daily_Return / Adj_Return (added by :class:`DataLoader`).

        Returns
        -------
        pd.DataFrame
            Original columns + 32 indicators. Initial rows that don't
            yet have a 200-day MA are kept (they're masked as NaN); the
            backtester / signal engine handles NaNs explicitly.
        """
        out = df.copy()

        close = out["Close"]
        high = out["High"]
        low = out["Low"]
        volume = out["Volume"]

        # ----- 1. TREND -----
        # Simple moving averages. The 200-day MA is THE long-term trend
        # filter — being above MA200 is the most basic "is this a stock
        # I should even consider buying?" check.
        out["MA_20"] = close.rolling(C.MA_SHORT).mean()
        out["MA_50"] = close.rolling(C.MA_MEDIUM).mean()
        out["MA_200"] = close.rolling(C.MA_LONG).mean()

        # Exponential MAs weight recent data more — used by MACD and as a
        # faster trend lens for short-term setups.
        out["EMA_12"] = EMAIndicator(close, window=C.EMA_FAST).ema_indicator()
        out["EMA_26"] = EMAIndicator(close, window=C.EMA_SLOW).ema_indicator()

        # MACD — fast EMA minus slow EMA, with a 9-day signal line.
        # Histogram = MACD - Signal; histogram zero-crossings often lead
        # actual MACD crossovers by a bar or two.
        macd = MACD(close, window_slow=C.MACD_SLOW, window_fast=C.MACD_FAST,
                    window_sign=C.MACD_SIGNAL)
        out["MACD"] = macd.macd()
        out["MACD_Signal"] = macd.macd_signal()
        out["MACD_Hist"] = macd.macd_diff()

        # Crossover booleans — concrete regime markers used downstream.
        # MA50 > MA200 ("golden cross") = textbook long-term uptrend.
        out["MA20_above_MA50"] = (out["MA_20"] > out["MA_50"]).astype(int)
        out["MA50_above_MA200"] = (out["MA_50"] > out["MA_200"]).astype(int)

        # ----- 2. MOMENTUM -----
        # RSI — Wilder smoothing, 14-period default. <30 = oversold,
        # >70 = overbought. The single most-used oscillator on the planet.
        out["RSI_14"] = RSIIndicator(close, window=C.RSI_PERIOD).rsi()

        # Rate of Change — pure % move over N bars. A clean momentum
        # gauge with no smoothing.
        out["ROC_10"] = ROCIndicator(close, window=C.ROC_PERIOD).roc()

        # Stochastic — where Close sits inside the recent High-Low range.
        # %K is the raw line, %D is its 3-period MA (smoothed).
        stoch = StochasticOscillator(high=high, low=low, close=close,
                                     window=C.STOCH_K, smooth_window=C.STOCH_D)
        out["Stoch_K"] = stoch.stoch()
        out["Stoch_D"] = stoch.stoch_signal()

        # ----- 3. VOLATILITY -----
        # Bollinger Bands — 20-SMA ± 2 std-dev envelope.
        bb = BollingerBands(close, window=C.BB_PERIOD, window_dev=C.BB_STD)
        out["BB_Upper"] = bb.bollinger_hband()
        out["BB_Mid"] = bb.bollinger_mavg()
        out["BB_Lower"] = bb.bollinger_lband()
        # Width = (Upper - Lower) / Mid — volatility scaled by price.
        # Compressed width = "the squeeze," often a coiled-spring setup.
        out["BB_Width"] = (out["BB_Upper"] - out["BB_Lower"]) / out["BB_Mid"]
        # %B = where price sits in the band. 0 = at lower band, 1 = upper.
        out["BB_Pct"] = bb.bollinger_pband()

        # ATR — the canonical absolute-volatility measure. Used to size
        # stops and targets (2× ATR is a common stop, 4× ATR a target).
        out["ATR_14"] = AverageTrueRange(high=high, low=low, close=close,
                                         window=C.ATR_PERIOD).average_true_range()

        # Annualised volatility from daily log returns. sqrt(252) is the
        # standard annualisation factor for daily-frequency data.
        if "Adj_Return" in out.columns:
            log_ret = out["Adj_Return"]
        else:
            log_ret = np.log1p(close.pct_change())
        out["Volatility_20"] = log_ret.rolling(C.VOL_WINDOW).std() * np.sqrt(C.TRADING_DAYS)

        # ----- 4. VOLUME -----
        # On-Balance Volume — cumulative volume signed by daily direction.
        # Confirms whether price moves are backed by real participation.
        out["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

        out["Vol_MA_20"] = volume.rolling(C.VOLUME_MA_PERIOD).mean()
        # Vol_Ratio > 1.5 → above-average volume; >3 → climactic.
        # Replace 0 with NaN to avoid divide-by-zero on a halt day.
        out["Vol_Ratio"] = volume / out["Vol_MA_20"].replace(0, np.nan)

        # ----- 5. RETURNS (multi-period) -----
        # Use simple returns: linear over short windows is close enough
        # to log, and these are interpreted as "how much did I make?"
        out["Return_5d"] = close.pct_change(5)
        out["Return_20d"] = close.pct_change(20)

        # ----- 6. PRICE STRUCTURE -----
        # Intraday range as a % of close — a coarse intraday-vol gauge.
        out["High_Low_Pct"] = (high - low) / close.replace(0, np.nan)
        # Overnight gap — Close vs prior Close (after auto_adjust, this
        # captures genuine overnight info shock, not splits/dividends).
        out["Gap_Pct"] = (out["Open"] - close.shift(1)) / close.shift(1).replace(0, np.nan)

        # The single most important trend filter in long-only investing.
        out["Close_above_MA200"] = (close > out["MA_200"]).astype(int)
        # Continuous version — by how much % is price above/below MA200.
        out["Dist_from_MA200_pct"] = (close - out["MA_200"]) / out["MA_200"].replace(0, np.nan)

        # Cast the integer-flag columns to a stable int dtype (the booleans
        # above are float because they were computed against NaN-laden MAs).
        for col in ("MA20_above_MA50", "MA50_above_MA200", "Close_above_MA200"):
            out[col] = out[col].fillna(0).astype(int)

        return out

    # ------------------------------------------------------------------
    # Universe pipeline
    # ------------------------------------------------------------------
    def compute_universe(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Apply :meth:`compute` to every ticker in a universe dict."""
        enriched: dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            try:
                enriched[ticker] = self.compute(df)
            except Exception as e:
                logger.error("[%s] feature computation failed: %s", ticker, e)
        logger.info("Features computed for %d/%d tickers", len(enriched), len(data))
        return enriched


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny synthetic OHLCV frame and verify shape/columns.
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n)))
    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
    df["Daily_Return"] = df["Close"].pct_change()
    df["Adj_Return"] = np.log1p(df["Daily_Return"])

    enriched = FeatureEngineer().compute(df)
    print(f"Input columns:  {len(df.columns)}  →  Output columns: {len(enriched.columns)}")
    print(f"New indicators: {len(enriched.columns) - len(df.columns)}")
    print("\nTail (last 3 rows of key columns):")
    print(enriched[["Close", "MA_50", "MA_200", "RSI_14", "MACD_Hist",
                    "BB_Pct", "ATR_14", "Volatility_20"]].tail(3).round(2))
