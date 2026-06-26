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

Conviction, sizing & track record (2026-06-26)
----------------------------------------------
Every pick carries an HONEST conviction tier (``_conviction`` — High/Medium/
Low from composite score × breadth of engine agreement × confidence; a strong
score from a single lens is capped at Medium) and a suggested position size
(``_suggested_weight`` — conviction × regime, ≤10% cap, mirroring the
backtester). ``market_summary`` powers a "what to do today" banner, and
``historical_track_record`` replays the exact short-term rule over history so
the engine's confidence is *earned* (shown hit-rate, incl. losers), not
asserted.

Filters
-------
Short-term : Signal ≥ 0, Confidence ≥ 0.30, Return_20d > 0, Regime ≥ SIDEWAYS,
             composite Score ≥ 0.28, AND ≥2 of the 3 trend-following engines
             agree (the 2026-06-26 tightening — one lens firing is noise).
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
# Conviction + position sizing — making the engine's confidence HONEST and
# ACTIONABLE rather than just loud.
# ---------------------------------------------------------------------------
# Regime participation bonus (also reused inside the short-term score).
REGIME_BONUS: dict[str, float] = {"BULL": 1.0, "SIDEWAYS": 0.5, "BEAR": 0.0}

# Short-term screen thresholds (raised 2026-06-26 as part of the "tighten the
# signal" upgrade — fewer, higher-quality ideas):
#   * confidence floor unchanged at 0.30,
#   * composite score floor 0.25 → 0.28,
#   * NEW: at least ST_MIN_AGREEMENT of the three trend-following sub-engines
#     (Trend / Momentum / Volume) must independently agree. A single lens
#     firing is noise; two-plus aligning is signal.
ST_CONF_FLOOR: float = 0.30
ST_SCORE_FLOOR: float = 0.28
ST_MIN_AGREEMENT: int = 2

# Conviction → suggested fraction of investable capital, BEFORE the regime
# haircut below. Anchored to the backtest's 10% per-stock cap so the live
# guidance and the simulated strategy speak the same language.
CONVICTION_SIZING: dict[str, float] = {"High": 0.10, "Medium": 0.06, "Low": 0.03}
# Risk-off haircut: commit less of that target as the tape weakens — exactly
# the size_mult the Backtester applies (1.0 / 0.7 / 0.5).
REGIME_SIZE_MULT: dict[str, float] = {"BULL": 1.0, "SIDEWAYS": 0.7, "BEAR": 0.5}


def _conviction(score: float, agreement: int, confidence: float) -> tuple[str, str]:
    """Map (composite score, breadth of engine agreement, confidence) → an
    HONEST conviction tier. Returns ``(label, stars)``.

    The philosophy: conviction is high only when the idea is strong on the
    composite score AND backed by *breadth* (multiple independent lenses
    agree) AND the signal magnitude (confidence) is meaningful. A high score
    from a single lens is deliberately capped at Medium — that's the
    calibration that stops the tool from sounding certain when it isn't.
    """
    if score >= 0.45 and agreement >= 3 and confidence >= 0.38:
        return "High", "★★★"
    if score >= 0.32 and agreement >= 2:
        return "Medium", "★★☆"
    return "Low", "★☆☆"


def _suggested_weight(conviction: str, regime_label: str) -> float:
    """Suggested allocation as a FRACTION of investable capital, from
    conviction × regime. Mirrors the backtester's sizing so the advice is
    consistent with what was validated. Always ≤ the 10% per-stock cap."""
    base = CONVICTION_SIZING.get(conviction, 0.03)
    mult = REGIME_SIZE_MULT.get(regime_label, 0.7)
    return round(min(base * mult, 0.10), 4)


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
    # ---- Shared short-term scoring (single source of truth) ----
    # Both the live screen (_rank_short_term) and the historical track record
    # call these, so what the cards show and what the track record measures
    # are guaranteed to be the SAME rule — honesty by construction.
    @staticmethod
    def _short_term_score(df: pd.DataFrame) -> pd.Series:
        """Composite 0..~1 short-term momentum score, vectorised over rows."""
        mom = np.tanh(df["Return_20d"] / 0.20).clip(lower=0)             # 20-day momentum
        conf = df["Confidence"].clip(lower=0, upper=1)                    # signal confidence
        rsi_s = df["RSI_14"].apply(_rsi_score_short)                      # healthy-RSI band
        vol_s = np.tanh((df["Vol_Ratio"].fillna(1.0) - 1.0)).clip(lower=0)  # volume confirm
        reg_s = df["Regime_Label"].map(REGIME_BONUS).fillna(0.0)         # regime tailwind
        return 0.30 * mom + 0.25 * conf + 0.20 * rsi_s + 0.15 * vol_s + 0.10 * reg_s

    @staticmethod
    def _engine_agreement(df: pd.DataFrame) -> pd.Series:
        """How many of the three TREND-FOLLOWING sub-engines (Trend /
        Momentum / Volume) independently lean bullish. Mean-reversion is
        excluded on purpose — it's contrarian, so it *should* fight a strong
        trend, and counting it would punish exactly the setups we want."""
        cols = [c for c in ("Score_Trend", "Score_Momentum", "Score_Volume")
                if c in df.columns]
        if not cols:
            return pd.Series(0, index=df.index)
        return sum((df[c].fillna(0.0) > 0.05).astype(int) for c in cols)

    @classmethod
    def _short_term_eligible(cls, df: pd.DataFrame) -> pd.Series:
        """Boolean mask: rows that pass the (tightened) short-term screen.

        Gates: not a Sell, confidence ≥ floor, 20-day return strictly
        positive (it's a momentum screen), regime ≥ SIDEWAYS, composite
        score ≥ floor, AND ≥2 trend-following engines agree. The agreement
        gate is the 2026-06-26 tightening — one lens firing is noise.
        """
        score = cls._short_term_score(df)
        agree = cls._engine_agreement(df)
        return (
            (df["Signal"] >= 0)
            & (df["Confidence"] >= ST_CONF_FLOOR)
            & (df["Return_20d"] > 0.0)
            & (df["Regime_Label"].isin(["SIDEWAYS", "BULL"]))
            & (score >= ST_SCORE_FLOOR)
            & (agree >= ST_MIN_AGREEMENT)
        )

    def _rank_short_term(self, snap: pd.DataFrame) -> pd.DataFrame:
        if snap.empty:
            return snap
        df = snap.copy()
        df["Score"] = self._short_term_score(df)
        df["Engine_Agreement"] = self._engine_agreement(df)
        df = df[self._short_term_eligible(df)].copy()
        if df.empty:
            return df
        df["Type"] = "Short-Term"
        df["Holding_Period"] = df["Volatility_20"].apply(
            lambda v: _holding_period(v, "short_term")
        )
        df["Rationale"] = df.apply(self._rationale_short, axis=1)
        df = self._add_conviction_sizing(df)
        # High-conviction ideas lead, then by score within a tier.
        return (self._format_output(df)
                .sort_values(["_conv_rank", "Score"], ascending=[False, False]))

    # ---- Conviction + position-size annotation (shared short & long) ----
    @staticmethod
    def _add_conviction_sizing(df: pd.DataFrame) -> pd.DataFrame:
        """Attach Conviction (label + stars), a numeric _conv_rank for
        sorting, and a Suggested_Weight (fraction of capital)."""
        df = df.copy()
        agree = df["Engine_Agreement"] if "Engine_Agreement" in df.columns \
            else RecommendationEngine._engine_agreement(df)
        convs = [
            _conviction(float(s), int(a), float(c))
            for s, a, c in zip(df["Score"], agree, df["Confidence"])
        ]
        df["Conviction"] = [c[0] for c in convs]
        df["Conviction_Stars"] = [c[1] for c in convs]
        df["_conv_rank"] = df["Conviction"].map(
            {"High": 3, "Medium": 2, "Low": 1}).fillna(1).astype(int)
        df["Suggested_Weight"] = [
            _suggested_weight(cv, str(rg))
            for cv, rg in zip(df["Conviction"], df["Regime_Label"])
        ]
        return df

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
        df["Engine_Agreement"] = self._engine_agreement(df)
        df["Holding_Period"] = df["Volatility_20"].apply(
            lambda v: _holding_period(v, "long_term")
        )
        df["Rationale"] = df.apply(self._rationale_long, axis=1)
        df = self._add_conviction_sizing(df)
        return (self._format_output(df)
                .sort_values(["_conv_rank", "Score"], ascending=[False, False]))

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
            "ticker", "Type", "Score", "Conviction", "Conviction_Stars",
            "_conv_rank", "Engine_Agreement", "Suggested_Weight",
            "Signal_Strength", "Confidence",
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
    # Market summary — drives the "what to do today" action-plan banner
    # ------------------------------------------------------------------
    @staticmethod
    def market_summary(signaled: dict[str, pd.DataFrame]) -> dict:
        """Latest-bar regime breakdown across the universe. Cheap; lets the
        UI tell the user the market backdrop in one honest line."""
        counts = {"BULL": 0, "SIDEWAYS": 0, "BEAR": 0}
        n = 0
        for df in signaled.values():
            last = df.dropna(subset=["Regime_Label"]).tail(1)
            if last.empty:
                continue
            lbl = str(last["Regime_Label"].iloc[0])
            if lbl in counts:
                counts[lbl] += 1
            n += 1
        defensive = counts["BEAR"] + counts["SIDEWAYS"]
        if n == 0:
            phase = "Unknown"
        elif counts["BULL"] >= 0.5 * n:
            phase = "Constructive"
        elif counts["BEAR"] >= 0.4 * n:
            phase = "Defensive"
        else:
            phase = "Mixed / range-bound"
        return {"counts": counts, "n_scored": n,
                "defensive": defensive, "phase": phase}

    # ------------------------------------------------------------------
    # Retrospective track record — confidence that is EARNED, not asserted
    # ------------------------------------------------------------------
    def historical_track_record(
        self,
        signaled: dict[str, pd.DataFrame],
        lookback_days: int = 504,
        max_hold: int = 40,
        atr_stop: float = 2.0,
        atr_target: float = 4.0,
    ) -> dict:
        """Replay the SHORT-TERM entry rule over recent history and simulate
        each idea's outcome with the exact stop/target the cards show.

        This is deliberately a *retrospective* computation, not a live
        journal: a public, stateless Streamlit Cloud app resets its disk on
        every redeploy, so a "log what it told me" file would be wiped and
        misleading. Replaying the rule over history needs no storage, is
        perfectly reproducible, and — crucially — shows the LOSSES too.

        Honest framing for the UI: this is the hit-rate of individual
        signals (entry → first of stop / target / opposite-signal / time
        limit), net of round-trip cost. It is NOT a portfolio return (no
        capital or position-count constraint) — so report it as signal
        quality, not "what you'd have made".

        Returns a stats dict (``{"n": 0}`` when nothing triggered).
        """
        need = ["Close", "High", "Low", "ATR_14", "Return_20d",
                "Confidence", "RSI_14", "Regime_Label", "Signal"]
        trades: list[dict] = []
        for ticker, raw in signaled.items():
            d = raw.dropna(subset=[c for c in need if c in raw.columns])
            if len(d) < 60 or not all(c in d.columns for c in need):
                continue
            d = d.tail(lookback_days + max_hold)
            elig = self._short_term_eligible(d).to_numpy()
            close = d["Close"].to_numpy(dtype=float)
            high = d["High"].to_numpy(dtype=float)
            low = d["Low"].to_numpy(dtype=float)
            atr = d["ATR_14"].fillna(0.0).to_numpy(dtype=float)
            sig = d["Signal"].to_numpy(dtype=float)
            n = len(d)
            i = 0
            while i < n - 1:
                if not elig[i] or atr[i] <= 0:
                    i += 1
                    continue
                entry = close[i]
                stop = entry - atr_stop * atr[i]
                target = entry + atr_target * atr[i]
                ret = None
                j = i
                for j in range(i + 1, min(i + 1 + max_hold, n)):
                    if low[j] <= stop:
                        ret, reason = stop / entry - 1.0, "stop"
                        break
                    if high[j] >= target:
                        ret, reason = target / entry - 1.0, "target"
                        break
                    if sig[j] == -1:
                        ret, reason = close[j] / entry - 1.0, "signal"
                        break
                if ret is None:                      # ran out of room → time exit
                    j = min(i + max_hold, n - 1)
                    ret, reason = close[j] / entry - 1.0, "time"
                ret -= 2.0 * C.TRANSACTION_COST       # round-trip friction
                trades.append({"ticker": ticker, "ret": float(ret),
                               "hold": int(j - i), "reason": reason})
                i = j + 1                             # no overlapping entries per name

        if not trades:
            return {"n": 0}
        tr = pd.DataFrame(trades)
        wins = tr.loc[tr["ret"] > 0, "ret"]
        losses = tr.loc[tr["ret"] <= 0, "ret"]
        pf = (float(wins.sum() / abs(losses.sum()))
              if len(losses) and losses.sum() < 0
              else (float("inf") if len(wins) else 0.0))
        return {
            "n": int(len(tr)),
            "win_rate": float(len(wins) / len(tr)),
            "avg_return": float(tr["ret"].mean()),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "profit_factor": pf,
            "avg_hold_days": float(tr["hold"].mean()),
            "best": float(tr["ret"].max()),
            "worst": float(tr["ret"].min()),
            "exit_reasons": tr["reason"].value_counts().to_dict(),
            "lookback_days": int(lookback_days),
        }

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

    # ---- Exercise the 2026-06-26 upgrade: conviction, sizing, summary, track record ----
    print("\n" + "=" * 80)
    print("  UPGRADE SELF-CHECKS (conviction / sizing / summary / track record)")
    print("=" * 80)

    summ = eng.market_summary(signaled)
    print(f"Market summary: phase={summ['phase']}  counts={summ['counts']}  "
          f"scored={summ['n_scored']}")

    for kind in ("short_term", "long_term"):
        df = result[kind]
        if df.empty:
            print(f"[{kind}] empty today (honest no-pick).")
            continue
        assert {"Conviction", "Conviction_Stars", "Suggested_Weight"} <= set(df.columns), \
            f"{kind} missing conviction/sizing columns"
        assert df["Conviction"].isin(["High", "Medium", "Low"]).all()
        assert (df["Suggested_Weight"] <= 0.10 + 1e-9).all(), "weight exceeds 10% cap"
        # Sorted High→Low conviction.
        ranks = df["_conv_rank"].tolist() if "_conv_rank" in df.columns else []
        assert ranks == sorted(ranks, reverse=True), f"{kind} not conviction-sorted"
        print(f"[{kind}] {len(df)} pick(s); convictions="
              f"{df['Conviction'].value_counts().to_dict()}; "
              f"weights={[f'{w*100:.0f}%' for w in df['Suggested_Weight']]}")

    tr = eng.historical_track_record(signaled)
    if tr.get("n", 0):
        print(f"Track record: n={tr['n']}  win_rate={tr['win_rate']*100:.0f}%  "
              f"avg_return={tr['avg_return']*100:+.2f}%  PF={tr['profit_factor']:.2f}  "
              f"avg_hold={tr['avg_hold_days']:.0f}d  exits={tr['exit_reasons']}")
        assert 0.0 <= tr["win_rate"] <= 1.0
    else:
        print("Track record: no historical short-term entries on synthetic data.")
    print("✓ upgrade self-checks passed")
