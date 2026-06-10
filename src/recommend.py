"""
src/recommend.py
================

Phase 6B — Recommendation engine.

Screens the full universe and produces two ranked lists:

* **Short-term opportunities** (2–8 week holding period). Pure momentum
  plays — we want stocks that are *moving* and confirmed by volume.

* **Long-term opportunities** (3–18 month holding period). Quality
  compounders — we want stocks in steady uptrends with controlled
  volatility, only suggested when the broader regime is bullish.

Also produces **exit recommendations** for the user's current holdings:
positions to consider trimming or closing based on signal, regime, or
overextension.

Scoring philosophy
------------------
Each opportunity type has its own weighted scoring function so the same
ticker can rank #1 for short-term and #15 for long-term (e.g. a stock
hitting an overbought, high-volatility breakout).

Short-term weights
~~~~~~~~~~~~~~~~~~
    20-day momentum         30%
    Signal confidence       25%
    RSI score (peak at 55)  20%
    Volume confirmation     15%
    Regime bonus            10%   (BULL=1.0, SIDEWAYS=0.5, BEAR=0)

Long-term weights
~~~~~~~~~~~~~~~~~
    Above MA200 (binary)    25%
    Volatility (inverted)   25%   (low vol = high quality)
    Trend quality           20%   (peaks at +5% to +20% above MA200)
    Signal confidence       15%
    Positive 20-day return  15%

Filters
-------
Short-term : Signal ≥ 0, Confidence ≥ 0.30, Return_20d > 0, Regime ≥ SIDEWAYS,
             composite Score ≥ 0.25 (absolute quality floor — being the only
             stock to clear the gates does not by itself make a stock a pick).
Long-term  : Close > MA200, Signal ≥ 0, Regime = BULL, Volatility_20 < 0.45.
             (No extra floor needed: the binary above-MA200 term already
             contributes 0.25, and the BULL + MA200 gates encode coherence.)
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
logger = logging.getLogger("recommend")


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------
def _rsi_score_short(rsi: float) -> float:
    """Triangular score peaked at RSI=55 (healthy momentum without overbought).

    RSI=55 → 1.0;  RSI=30 or 80 → 0.0;  outside → 0.0.
    """
    if pd.isna(rsi):
        return 0.0
    if 30 <= rsi <= 55:
        return float((rsi - 30) / 25)
    if 55 < rsi <= 80:
        return float((80 - rsi) / 25)
    return 0.0


def _trend_quality_score(dist_pct: float) -> float:
    """Triangular score peaked at +5% to +20% above MA200.

    A stock far above MA200 (>30%) is *too* extended; right at MA200 is
    only starting to recover. The sweet spot for a long-term entry is
    "trend established but not euphoric."
    """
    if pd.isna(dist_pct):
        return 0.0
    if dist_pct < 0:
        return 0.0
    if dist_pct < 0.05:
        return float(dist_pct / 0.05)                       # ramp 0 → 1
    if dist_pct <= 0.20:
        return 1.0                                           # plateau
    if dist_pct <= 0.40:
        return float((0.40 - dist_pct) / 0.20)              # ramp 1 → 0
    return 0.0


def _holding_period(vol: float, kind: str) -> str:
    """Auto-adjust suggested holding window by realised volatility."""
    if kind == "short_term":
        if pd.isna(vol) or vol < 0.20: return "6–8 weeks"
        if vol < 0.35: return "4–6 weeks"
        return "2–4 weeks"
    # long-term
    if pd.isna(vol) or vol < 0.20: return "12–18 months"
    if vol < 0.30: return "6–12 months"
    return "3–6 months"


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """Rank Buy candidates (short- + long-term) and flag Exits."""

    REGIME_BONUS = {"BULL": 1.0, "SIDEWAYS": 0.5, "BEAR": 0.0}

    def __init__(
        self,
        max_short_term: int = 10,
        max_long_term: int = 10,
    ) -> None:
        self.max_short_term = int(max_short_term)
        self.max_long_term = int(max_long_term)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        signaled: dict[str, pd.DataFrame],
        current_holdings: dict[str, float] | None = None,
    ) -> dict:
        """Produce ranked Short/Long-term buys and Exit alerts."""
        snapshots = self._latest_snapshots(signaled)
        short_term = self._rank_short_term(snapshots)
        long_term = self._rank_long_term(snapshots, exclude=set(short_term["ticker"])
                                         if not short_term.empty else set())

        exits = pd.DataFrame()
        if current_holdings:
            exits = self._exit_alerts(snapshots, current_holdings)

        return {
            "short_term": short_term.head(self.max_short_term).reset_index(drop=True),
            "long_term": long_term.head(self.max_long_term).reset_index(drop=True),
            "exits": exits.reset_index(drop=True),
        }

    # ------------------------------------------------------------------
    # Build the "today" snapshot table
    # ------------------------------------------------------------------
    @staticmethod
    def _latest_snapshots(signaled: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """One row per ticker — the most recent fully-populated bar."""
        rows = []
        for t, df in signaled.items():
            if df.empty:
                continue
            last = df.dropna(subset=["Close", "RSI_14", "MA_200"]).tail(1)
            if last.empty:
                continue
            row = last.iloc[0].to_dict()
            row["ticker"] = t
            row["as_of"] = last.index[0]
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Short-term scoring
    # ------------------------------------------------------------------
    def _rank_short_term(self, snap: pd.DataFrame) -> pd.DataFrame:
        if snap.empty:
            return snap

        df = snap.copy()
        # Filter.
        #
        # Return_20d must be STRICTLY POSITIVE (was > −10%): this is a
        # momentum screen, and a momentum candidate that has been falling
        # for a month is a contradiction in terms. Real incident
        # (2026-06-10): WIPRO — Hold signal, 20-day return −3.9%, RSI 30 —
        # was the only name to squeak past the loose filter and therefore
        # headlined the tab as "#1" with a momentum component of exactly
        # 0.00. Last-man-standing is not a recommendation.
        df = df[
            (df["Signal"] >= 0) &
            (df["Confidence"] >= 0.30) &
            (df["Return_20d"] > 0.0) &
            (df["Regime"].isin([C.REGIME_CODES["SIDEWAYS"], C.REGIME_CODES["BULL"]]))
        ].copy()
        if df.empty:
            return df

        # Component scores
        # Momentum: ret_20d squashed via tanh — 20% over 20d → ~0.76 of max.
        mom = np.tanh(df["Return_20d"] / 0.20).clip(lower=0)
        conf = df["Confidence"].clip(lower=0, upper=1)
        rsi_s = df["RSI_14"].apply(_rsi_score_short)
        # Volume: tanh of (Vol_Ratio - 1).
        vol_s = np.tanh((df["Vol_Ratio"].fillna(1.0) - 1.0)).clip(lower=0)
        reg_s = df["Regime_Label"].map(self.REGIME_BONUS).fillna(0.0)

        df["Score"] = (
            0.30 * mom + 0.25 * conf + 0.20 * rsi_s + 0.15 * vol_s + 0.10 * reg_s
        )
        # ABSOLUTE quality floor — ranking alone isn't enough. The filter
        # above is a pass/fail gate per condition, but a stock can clear
        # every gate minimally and still be a poor idea (regime bonus 0.10
        # + threshold confidence 0.075 ≈ 0.18 with zero momentum/RSI/volume
        # contribution). Requiring Score ≥ 0.25 means at least one engine
        # beyond "the market is bullish" actually likes the stock; genuine
        # momentum candidates (e.g. +10% over 20d at conf 0.30 in BULL)
        # score ≈ 0.31 and pass comfortably. An empty list is the honest
        # output when nothing qualifies — the UI says exactly that.
        df = df[df["Score"] >= 0.25]
        if df.empty:
            return df
        df["Type"] = "Short-Term"
        df["Holding_Period"] = df["Volatility_20"].apply(
            lambda v: _holding_period(v, "short_term")
        )
        df["Rationale"] = df.apply(self._rationale_short, axis=1)
        return self._format_output(df).sort_values("Score", ascending=False)

    @staticmethod
    def _rationale_short(row) -> str:
        bits = []
        ret20 = float(row.get("Return_20d", 0.0))
        bits.append(f"20-day return {ret20*100:+.1f}%")
        rsi = float(row.get("RSI_14", 50))
        if 50 <= rsi <= 65:
            bits.append(f"RSI {rsi:.0f} (healthy momentum)")
        elif rsi > 65:
            bits.append(f"RSI {rsi:.0f} (strong, watch overbought)")
        vr = float(row.get("Vol_Ratio", 1.0) or 1.0)
        if vr > 1.5:
            bits.append(f"volume {vr:.1f}× avg confirms")
        if row.get("Regime_Label") == "BULL":
            bits.append("bullish regime tailwind")
        return "; ".join(bits) + "."

    # ------------------------------------------------------------------
    # Long-term scoring
    # ------------------------------------------------------------------
    def _rank_long_term(self, snap: pd.DataFrame,
                        exclude: set[str] | None = None) -> pd.DataFrame:
        if snap.empty:
            return snap

        df = snap.copy()
        if exclude:
            df = df[~df["ticker"].isin(exclude)]

        # Filter
        df = df[
            (df["Close_above_MA200"] == 1) &
            (df["Signal"] >= 0) &
            (df["Regime_Label"] == "BULL") &
            (df["Volatility_20"] < 0.45)
        ].copy()
        if df.empty:
            return df

        # Component scores
        above_ma200 = df["Close_above_MA200"].astype(float)            # binary
        # Inverted volatility — < 0.20 (annualised) is great, > 0.45 was filtered out.
        vol_score = (1.0 - (df["Volatility_20"] / 0.45)).clip(lower=0, upper=1)
        trend_q = df["Dist_from_MA200_pct"].apply(_trend_quality_score)
        conf = df["Confidence"].clip(lower=0, upper=1)
        # Positive 20-day momentum — pass through if positive, zero otherwise.
        mom_pos = (df["Return_20d"].clip(lower=0) / 0.15).clip(upper=1)

        df["Score"] = (
            0.25 * above_ma200 + 0.25 * vol_score + 0.20 * trend_q
            + 0.15 * conf + 0.15 * mom_pos
        )
        df["Type"] = "Long-Term"
        df["Holding_Period"] = df["Volatility_20"].apply(
            lambda v: _holding_period(v, "long_term")
        )
        df["Rationale"] = df.apply(self._rationale_long, axis=1)
        return self._format_output(df).sort_values("Score", ascending=False)

    @staticmethod
    def _rationale_long(row) -> str:
        bits = []
        dist = float(row.get("Dist_from_MA200_pct", 0.0))
        bits.append(f"{dist*100:+.1f}% above MA200")
        vol = float(row.get("Volatility_20", 0.0))
        if vol < 0.20:
            bits.append(f"low annualised vol ({vol*100:.0f}%)")
        elif vol < 0.30:
            bits.append(f"moderate vol ({vol*100:.0f}%)")
        bits.append("bullish regime")
        return "; ".join(bits) + "."

    # ------------------------------------------------------------------
    # Output formatting (shared between short & long)
    # ------------------------------------------------------------------
    @staticmethod
    def _format_output(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # ±0.5% entry zone for limit orders.
        close = df["Close"]
        df["Entry_Low"] = close * (1.0 - 0.005)
        df["Entry_High"] = close * (1.0 + 0.005)

        # Recompute trade levels from ATR for EVERY ranked candidate. The
        # Stop/Target columns carried in from add_entry_exit_levels are only
        # populated on actual BUY/SELL bars, so a Hold-ranked watchlist name
        # would otherwise show Stop = T1 = T2 = Close and R/R 0.0 (looks
        # broken). Standard 2×ATR stop, 2×/4×ATR targets → a clean 1:2 reward.
        if "ATR_14" in df.columns:
            atr = df["ATR_14"].fillna(0.0)
            df["Stop_Loss"] = close - 2.0 * atr
            df["Target_1"] = close + 2.0 * atr
            df["Target_2"] = close + 4.0 * atr
            risk = (close - df["Stop_Loss"]).abs().replace(0, np.nan)
            df["Risk_Reward"] = ((df["Target_2"] - close).abs() / risk).fillna(0.0)

        cols = [
            "ticker", "Type", "Score", "Signal_Strength", "Confidence",
            "Close", "Entry_Low", "Entry_High",
            "Stop_Loss", "Target_1", "Target_2", "Risk_Reward",
            "Holding_Period", "RSI_14", "Volatility_20", "Regime_Label",
            "Rationale", "as_of",
        ]
        present = [c for c in cols if c in df.columns]
        return df[present]

    # ------------------------------------------------------------------
    # Exit alerts on current holdings
    # ------------------------------------------------------------------
    def _exit_alerts(self, snap: pd.DataFrame,
                     holdings: dict[str, float]) -> pd.DataFrame:
        if snap.empty:
            return pd.DataFrame()

        held = snap[snap["ticker"].isin(holdings.keys())].copy()
        if held.empty:
            return held

        rows = []
        for _, r in held.iterrows():
            t = r["ticker"]
            triggers: list[str] = []
            severity = None
            action = None
            # Protective stop for an EXISTING LONG holding: 2×ATR(14) below
            # the latest close — the same Turtle-style multiple the entry
            # levels use elsewhere.
            #
            # Why we don't read the snapshot's Stop_Loss column here:
            # ``add_entry_exit_levels`` only computes meaningful levels on
            # signal bars. On a Hold bar Stop_Loss degenerates to the close
            # itself ("your stop is the current price" — useless), and on a
            # SELL bar the geometry is the SHORT side (stop *above* price),
            # which is the wrong direction for protecting a long position
            # the user already owns. Same bug class as the degenerate trade
            # levels fixed in _format_output — this is the Exit-tab twin.
            close_px = float(r["Close"])
            atr = float(r.get("ATR_14", 0.0) or 0.0)
            stop = close_px - 2.0 * atr if atr > 0 else close_px * 0.95

            # 1) Active SELL signal
            if r.get("Signal", 0) == -1 and float(r.get("Confidence", 0.0)) > 0.40:
                triggers.append(f"SELL signal (conf {float(r['Confidence']):.2f})")
                severity = "HIGH"; action = "EXIT"

            # 2) Regime turned bearish
            if r.get("Regime_Label") == "BEAR":
                triggers.append("Regime turned BEAR")
                severity = "HIGH"; action = action or "EXIT"

            # 3) Overbought + strong run = take partial profit
            rsi = float(r.get("RSI_14", 50))
            ret20 = float(r.get("Return_20d", 0.0))
            if rsi > 75 and ret20 > 0.15:
                triggers.append(f"Overbought (RSI {rsi:.0f}, +{ret20*100:.0f}% in 20d)")
                severity = severity or "MEDIUM"; action = action or "REDUCE"

            # 4) Momentum deteriorating
            if ret20 < -0.10:
                triggers.append(f"20-day return {ret20*100:.1f}%")
                severity = severity or "MEDIUM"; action = action or "REDUCE"

            if not triggers:
                continue
            rows.append({
                "ticker": t,
                "current_value": holdings[t],
                "action": action,
                "urgency": severity,
                "triggers": "; ".join(triggers),
                "suggested_stop": stop,
                "close": float(r["Close"]),
                "rsi": rsi,
                "regime": r.get("Regime_Label"),
                "as_of": r.get("as_of"),
            })
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        # Sort: HIGH urgency first, then by current_value (biggest exposures first).
        urgency_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        out["_u"] = out["urgency"].map(urgency_order).fillna(99)
        out = out.sort_values(["_u", "current_value"], ascending=[True, False]).drop(columns="_u")
        return out

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------
    @staticmethod
    def print_report(result: dict) -> None:
        line = "=" * 80
        print(line)
        print("  RECOMMENDATIONS")
        print(line)

        for kind in ("short_term", "long_term"):
            df = result.get(kind, pd.DataFrame())
            print(f"\n[{kind.upper().replace('_', ' ')}]")
            if df.empty:
                print("  No candidates pass the filter today.")
                continue
            for i, row in df.iterrows():
                rr = float(row.get("Risk_Reward", 0))
                print(f"  #{i+1:>2}  {row['ticker']:<14}  Score {row['Score']:.2f}  "
                      f"({row['Signal_Strength']}, conf {row['Confidence']:.2f})")
                print(f"        Entry ₹{row['Entry_Low']:,.1f} – ₹{row['Entry_High']:,.1f}   "
                      f"Stop ₹{row['Stop_Loss']:,.1f}   "
                      f"T1 ₹{row['Target_1']:,.1f}   T2 ₹{row['Target_2']:,.1f}   "
                      f"R/R {rr:.1f}")
                print(f"        Hold {row['Holding_Period']}  |  {row['Rationale']}")

        exits = result.get("exits", pd.DataFrame())
        print("\n[EXIT ALERTS — Current Holdings]")
        if exits.empty:
            print("  ✓ No exit alerts.")
        else:
            for _, r in exits.iterrows():
                print(f"  [{r['urgency']:<6}] {r['ticker']:<14}  {r['action']}  "
                      f"@ ₹{r['close']:,.1f}  "
                      f"(₹{r['current_value']:,.0f} held)")
                print(f"          Triggers: {r['triggers']}")
                print(f"          Suggested stop: ₹{r['suggested_stop']:,.1f}")
        print(line)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from features import FeatureEngineer
    from regime import RegimeDetector
    from signals import SignalEngine, add_entry_exit_levels

    rng = np.random.default_rng(4)
    n = 500
    idx = pd.bdate_range("2022-01-03", periods=n)
    tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS", "TATASTEEL.NS"]
    signaled = {}
    for t in tickers:
        drift = rng.normal(0.0005, 0.0003)
        rets = rng.normal(drift, 0.014, n)
        close = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": 1_000_000.0,
        }, index=idx)
        df["Daily_Return"] = df["Close"].pct_change()
        df["Adj_Return"] = np.log1p(df["Daily_Return"])
        df = FeatureEngineer().compute(df)
        df = RegimeDetector(method="hmm").fit_transform(df)
        df = SignalEngine().generate(df)
        df = add_entry_exit_levels(df)
        signaled[t] = df

    eng = RecommendationEngine()
    result = eng.generate(signaled, current_holdings={"TCS.NS": 30_000, "RELIANCE.NS": 20_000})
    eng.print_report(result)
