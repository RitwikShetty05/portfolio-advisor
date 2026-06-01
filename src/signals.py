"""
src/signals.py
==============

Phase 4 — Signal generation.

Architecture
------------
Four independent **sub-engines** each cast a vote in ``[-1, +1]``:

    +---------------------+--------+-----------------------------------+
    | Sub-engine          | Weight | Reads                             |
    +=====================+========+===================================+
    | Trend               | 0.30   | MA crossovers, MACD histogram     |
    | Momentum            | 0.30   | RSI, MACD, ROC                    |
    | Mean Reversion      | 0.20   | Bollinger %B, Stochastic          |
    | Volume Confirmation | 0.20   | OBV slope, Volume ratio           |
    +---------------------+--------+-----------------------------------+

The weighted sum becomes ``Score_Raw`` (also in [-1, +1]). The score is
then **gated by the prevailing regime**:

    BULL      : buy ≥ 0.30, sell ≤ −0.50 — easy to buy, hard to sell
    SIDEWAYS  : buy ≥ 0.40, sell ≤ −0.40 — tighter both ways, smaller size
    BEAR      : buy ≥ 0.50, sell ≤ −0.25 — very hard to buy contra-trend

    (Gates recalibrated 2026-06 to the composite score's achievable range —
    its empirical max is ≈0.66 — so the strategy is no longer locked out of
    non-bull regimes. See ``config.REGIME_GATES`` for the calibration note.)

When the regime detector provides a continuous ``Regime_Prob_Bull``
posterior, it's used as a **soft gate** that smoothly tilts thresholds —
the moment a bear regime softens you don't have to wait for a discrete
flip to start buying.

Why four engines, not one?
--------------------------
Real traders triangulate. A "BUY" backed by three different lenses
(trending price, healthy momentum, confirming volume) survives noise
better than any single indicator. Weighting them rather than ANDing
them means a weak signal in one engine can still combine with strong
signals elsewhere — closer to how an experienced PM actually thinks.

Outputs
-------
Every row gets these columns appended:

    Signal           : -1 / 0 / +1
    Confidence       : float in [0, 1] — magnitude of the composite score
    Signal_Strength  : "Strong Buy" / "Buy" / "Hold" / "Sell" / "Strong Sell"
    Score_Trend      : sub-engine vote
    Score_Momentum   : sub-engine vote
    Score_MeanRev    : sub-engine vote
    Score_Volume     : sub-engine vote
    Score_Raw        : weighted composite, before gating
    Size_Mult        : regime-driven position-size multiplier in [0, 1]

Also provides :func:`add_entry_exit_levels` which annotates BUY rows
with the ATR-based stop and the 1:1 / 1:2 reward-multiple targets used
by both the backtester and the recommendation engine.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

logging.basicConfig(level=C.LOG_LEVEL, format=C.LOG_FORMAT)
logger = logging.getLogger("signals")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clip(s: pd.Series, lo: float = -1.0, hi: float = 1.0) -> pd.Series:
    """Clip a series to [lo, hi]. Used to keep sub-engine votes bounded."""
    return s.clip(lower=lo, upper=hi)


def _bucket_strength(signal: int, conf: float) -> str:
    """Map (signal, confidence) → human label for the Signal_Strength column."""
    if signal == 0:
        return "Hold"
    if signal > 0:
        return "Strong Buy" if conf >= 0.75 else "Buy"
    return "Strong Sell" if conf >= 0.75 else "Sell"


# ---------------------------------------------------------------------------
# Signal engine
# ---------------------------------------------------------------------------
class SignalEngine:
    """Generate BUY / SELL / HOLD signals with confidence and size multiplier."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        gates: dict[str, dict[str, float]] | None = None,
        use_soft_gate: bool = True,
    ) -> None:
        self.weights = weights or dict(C.SIGNAL_WEIGHTS)
        # Sanity: weights must sum to 1.0 within a small tolerance.
        total = sum(self.weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            logger.warning("Signal weights sum to %.4f, not 1.0 — renormalising.", total)
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.gates = gates or {k: dict(v) for k, v in C.REGIME_GATES.items()}
        self.use_soft_gate = use_soft_gate

    # ------------------------------------------------------------------
    # Sub-engine 1 — Trend
    # ------------------------------------------------------------------
    @staticmethod
    def _engine_trend(df: pd.DataFrame) -> pd.Series:
        """+1 = strong uptrend, -1 = strong downtrend.

        Components (each in [-1, 1], averaged):
            * MA20 vs MA50 (1 if bullish, -1 if bearish)
            * MA50 vs MA200 (golden/death cross — the long-term trend)
            * MACD histogram sign × tanh(scaled magnitude)
            * Close vs MA200 (binary trend filter)
        """
        # Boolean MA cross signals (-1 or +1).
        ma_short = 2 * df["MA20_above_MA50"].astype(float) - 1
        ma_long = 2 * df["MA50_above_MA200"].astype(float) - 1

        # MACD histogram → bounded contribution via tanh.
        # The scale (2.0 / 20-day rolling std of MACD_Hist) puts most days
        # in the (-1, +1) range without saturating; a regime change still
        # pushes it toward ±1.
        hist = df["MACD_Hist"]
        scale = hist.rolling(20, min_periods=5).std().replace(0, np.nan)
        macd_contrib = np.tanh(hist / scale.replace(0, 1.0))
        macd_contrib = macd_contrib.fillna(0.0)

        # Above-MA200 filter — the simplest, strongest long-term trend cue.
        above_200 = 2 * df["Close_above_MA200"].astype(float) - 1

        score = (ma_short + ma_long + macd_contrib + above_200) / 4.0
        return _clip(score)

    # ------------------------------------------------------------------
    # Sub-engine 2 — Momentum
    # ------------------------------------------------------------------
    @staticmethod
    def _engine_momentum(df: pd.DataFrame) -> pd.Series:
        """+1 = healthy momentum, -1 = momentum exhausted/turning down.

        Components:
            * RSI mapped to [-1, +1] with a positive bias around 55–60.
            * MACD line sign and slope (vs its 3-day MA).
            * ROC_10 squashed via tanh.
        """
        # RSI piecewise-linear mapping:
        #   <30 → -1 (oversold but punished as falling)
        #   30–50 → linearly to 0
        #   50–70 → linearly to +1
        #   >70 → +1 minus a slight overbought penalty
        rsi = df["RSI_14"].fillna(50)
        rsi_score = np.where(
            rsi < 30, -1.0,
            np.where(rsi < 50, (rsi - 50) / 20.0,
                     np.where(rsi <= 70, (rsi - 50) / 20.0,
                              1.0 - (rsi - 70) / 30.0))
        )
        rsi_score = pd.Series(rsi_score, index=df.index)

        # MACD line direction — combine sign with slope-of-line.
        macd = df["MACD"].fillna(0.0)
        macd_slope = macd - macd.shift(3)
        macd_sign = np.sign(macd)
        macd_score = np.tanh(macd_sign * 0.5 + macd_slope.fillna(0.0) * 2.0)

        roc = df["ROC_10"].fillna(0.0)
        roc_score = np.tanh(roc / 5.0)  # 5% ROC over 10d → ~tanh(1) ≈ 0.76

        score = (rsi_score + macd_score + roc_score) / 3.0
        return _clip(score)

    # ------------------------------------------------------------------
    # Sub-engine 3 — Mean reversion
    # ------------------------------------------------------------------
    @staticmethod
    def _engine_mean_reversion(df: pd.DataFrame) -> pd.Series:
        """Contrarian — fades extremes.

        Inputs:
            * BB %B (Bollinger position). Near 0 = oversold → buy; near
              1 = overbought → sell.
            * Stochastic (average of %K, %D). Same logic, different lookback.

        We linearly map the [0, 1]-range inputs to [+1, -1] so the engine
        naturally opposes overextensions. Note this is *intentionally* a
        weaker sub-engine (20% weight) — fading strong trends is dangerous;
        we want it as a tiebreaker, not a primary signal.
        """
        bbp = df["BB_Pct"].clip(lower=0, upper=1).fillna(0.5)
        bb_score = 1.0 - 2.0 * bbp                  # 0→+1, 1→-1

        # Stochastic is 0..100 in `ta`. Rescale to [0, 1] first.
        stoch_avg = ((df["Stoch_K"].fillna(50) + df["Stoch_D"].fillna(50)) / 200.0)
        stoch_score = 1.0 - 2.0 * stoch_avg

        score = (bb_score + stoch_score) / 2.0
        return _clip(score)

    # ------------------------------------------------------------------
    # Sub-engine 4 — Volume confirmation
    # ------------------------------------------------------------------
    @staticmethod
    def _engine_volume(df: pd.DataFrame) -> pd.Series:
        """Does volume confirm the direction of price?

        Inputs:
            * OBV slope (10-day diff, normalised).
            * Vol_Ratio — current volume vs 20d MA. Capped, tanh'd.

        Note: this engine is *direction-agnostic on its own* — it amplifies
        whichever direction price is going. We apply price direction here
        by multiplying with the sign of Return_5d so it actually adds
        information instead of just being a magnifier of noise.
        """
        obv = df["OBV"].ffill().fillna(0.0)
        obv_slope = obv.diff(10)
        obv_scale = obv_slope.rolling(60, min_periods=10).std().replace(0, np.nan)
        obv_score = np.tanh(obv_slope / obv_scale).fillna(0.0)

        vol_ratio = df["Vol_Ratio"].fillna(1.0).clip(upper=5.0)
        vol_score = np.tanh((vol_ratio - 1.0))      # 1.0 → 0, 2.0 → ~0.76

        # Apply direction: same-sign with 5-day return.
        direction = np.sign(df["Return_5d"].fillna(0.0))
        score = direction * (0.5 * obv_score + 0.5 * vol_score)
        return _clip(score)

    # ------------------------------------------------------------------
    # Main: compose the engines and gate by regime
    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add signal columns to one ticker's enriched + regime-labelled DF."""
        out = df.copy()
        out["Score_Trend"] = self._engine_trend(out)
        out["Score_Momentum"] = self._engine_momentum(out)
        out["Score_MeanRev"] = self._engine_mean_reversion(out)
        out["Score_Volume"] = self._engine_volume(out)

        out["Score_Raw"] = (
            self.weights["trend"]          * out["Score_Trend"]
            + self.weights["momentum"]     * out["Score_Momentum"]
            + self.weights["mean_reversion"] * out["Score_MeanRev"]
            + self.weights["volume"]       * out["Score_Volume"]
        )

        # Resolve gates per-row from regime label, possibly tilted by the
        # HMM's continuous P(Bull).
        signals = np.zeros(len(out), dtype=int)
        confidence = np.zeros(len(out), dtype=float)
        size_mult = np.zeros(len(out), dtype=float)

        regime_labels = out["Regime_Label"].fillna("SIDEWAYS").values
        prob_bull = (
            out["Regime_Prob_Bull"].values if "Regime_Prob_Bull" in out.columns
            else np.full(len(out), np.nan)
        )
        score_raw = out["Score_Raw"].values

        for i, lbl in enumerate(regime_labels):
            gate = self.gates.get(lbl, self.gates["SIDEWAYS"])
            buy_g, sell_g, sm = gate["buy_gate"], gate["sell_gate"], gate["size_mult"]

            # Soft gate: when P(Bull) is known, tilt the thresholds linearly
            # by up to 0.10 in either direction. Bullish posteriors make
            # the buy threshold easier and the sell threshold harder.
            if self.use_soft_gate and not np.isnan(prob_bull[i]):
                tilt = 0.10 * (prob_bull[i] - 0.5) * 2.0      # in [-0.10, +0.10]
                buy_g = max(0.10, buy_g - tilt)
                sell_g = max(0.10, sell_g + tilt)

            s = score_raw[i]
            if s >= buy_g:
                signals[i] = +1
            elif s <= -sell_g:
                signals[i] = -1
            # confidence is the bounded magnitude of the raw score.
            confidence[i] = float(min(1.0, abs(s)))
            size_mult[i] = sm if signals[i] != 0 else 0.0

        out["Signal"] = signals
        out["Confidence"] = confidence
        out["Size_Mult"] = size_mult
        out["Signal_Strength"] = [
            _bucket_strength(s, c) for s, c in zip(signals, confidence)
        ]
        return out

    def generate_universe(
        self, regimed: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Apply :meth:`generate` to every ticker."""
        out: dict[str, pd.DataFrame] = {}
        for t, df in regimed.items():
            try:
                out[t] = self.generate(df)
            except Exception as e:
                logger.error("[%s] signal generation failed: %s", t, e)
        logger.info("Signals generated for %d/%d tickers", len(out), len(regimed))
        return out


# ---------------------------------------------------------------------------
# add_entry_exit_levels — ATR-based stop & target annotation
# ---------------------------------------------------------------------------
def add_entry_exit_levels(df: pd.DataFrame, atr_mult_stop: float = 2.0,
                          atr_mult_t1: float = 2.0,
                          atr_mult_t2: float = 4.0) -> pd.DataFrame:
    """Annotate BUY/SELL rows with stop-loss and target prices.

    Levels are computed on **every** row (so the same column exists across
    the panel for plotting), but they are only *meaningful* on rows where
    ``Signal != 0``. Convention used:

        Stop_Loss  = Entry − 2 × ATR  (Turtle Traders standard)
        Target_1   = Entry + 2 × ATR  (1:1 R/R — partial profit)
        Target_2   = Entry + 4 × ATR  (1:2 R/R — runner)
        Risk_Reward = (Target_2 − Entry) / (Entry − Stop_Loss)

    For SELL signals the geometry flips: stops above entry, targets below.
    """
    out = df.copy()
    if "ATR_14" not in out.columns:
        raise ValueError("ATR_14 column required (run features.compute first).")

    entry = out["Close"]
    atr = out["ATR_14"].fillna(0.0)
    sig = out["Signal"]

    stop = entry.copy()
    t1 = entry.copy()
    t2 = entry.copy()

    # Long-side
    long_mask = (sig > 0)
    stop[long_mask] = entry[long_mask] - atr_mult_stop * atr[long_mask]
    t1[long_mask] = entry[long_mask] + atr_mult_t1 * atr[long_mask]
    t2[long_mask] = entry[long_mask] + atr_mult_t2 * atr[long_mask]

    # Short-side (geometry flipped). Kept for completeness — this system
    # is long-only by default but the maths is symmetric.
    short_mask = (sig < 0)
    stop[short_mask] = entry[short_mask] + atr_mult_stop * atr[short_mask]
    t1[short_mask] = entry[short_mask] - atr_mult_t1 * atr[short_mask]
    t2[short_mask] = entry[short_mask] - atr_mult_t2 * atr[short_mask]

    out["Entry_Price"] = entry
    out["Stop_Loss"] = stop
    out["Target_1"] = t1
    out["Target_2"] = t2

    # Risk/reward — denominator guarded against zero ATR (illiquid stock).
    risk = (entry - stop).abs().replace(0, np.nan)
    reward = (t2 - entry).abs()
    out["Risk_Reward"] = (reward / risk).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from features import FeatureEngineer  # type: ignore
    from regime import RegimeDetector     # type: ignore

    rng = np.random.default_rng(11)
    n = 500
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.014, n)))
    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n)),
        "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "Low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Adj_Return"] = np.log1p(df["Daily_Return"])

    enriched = FeatureEngineer().compute(df)
    regimed = RegimeDetector(method="hmm").fit_transform(enriched)
    signaled = SignalEngine().generate(regimed)
    signaled = add_entry_exit_levels(signaled)

    print("Signal distribution:")
    print(signaled["Signal_Strength"].value_counts().to_string())
    print("\nLast 3 rows with all signal columns:")
    print(signaled[["Close", "Regime_Label", "Score_Raw", "Confidence",
                    "Signal", "Signal_Strength", "Size_Mult",
                    "Stop_Loss", "Target_1", "Target_2", "Risk_Reward"]].tail(3).round(3))
