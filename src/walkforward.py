"""
src/walkforward.py
==================

Phase 7c — Walk-forward out-of-sample testing.

The problem with a single backtest
----------------------------------
A standard backtest fits the model (the HMM regime detector here) on the
*same* data on which it then reports performance. Even when no explicit
hyperparameters are tuned, this is contaminated: the HMM's parameters
(transitions, emissions) were chosen to explain the very returns we then
score. The reported Sharpe is therefore optimistic.

What walk-forward does
----------------------
Structurally separate training from testing by sliding a (train → test)
window forward through time::

    [────── train ──────][── test ──]
                            ↓ slide forward
                          [────── train ──────][── test ──]
                                                   ↓ slide forward
                                                 [────── train ──────][── test ──]

For each window:
  1. **Fit** the HMM on the training slice ONLY (via
     :meth:`RegimeDetector.fit_then_predict`).
  2. **Predict** regimes on the test slice using that fitted model.
  3. **Generate signals** on the test slice from those regime labels.
  4. **Backtest** the test slice in isolation.

Concatenate the test-window equity curves end-to-end — that is the true
out-of-sample (OOS) performance, the closest thing a backtest can give
to "what would have happened if we'd traded this strategy live."

Two modes
---------
* ``anchored=True``  (default) — training window *grows from the start*
  each iteration. The model has access to more history over time, which
  is realistic if you would have actually used everything available.
* ``anchored=False`` — fixed-size rolling training window. Useful for
  stress-testing whether the strategy survives "if I only had the last
  N years of data." More conservative; lower OOS sample size.

Why this is the single biggest credibility improvement
------------------------------------------------------
Most undergrad finance projects skip this step. Adding it signals you
understand *the* fundamental hazard of backtesting (overfitting to the
sample). Citing in-sample vs. OOS Sharpe degradation in an interview is
a tell that you've actually thought about this.

A typical honest OOS result will show **lower** Sharpe than in-sample —
that's expected, not a failure. The interesting question is *how much*
degradation, and whether the OOS Sharpe is still meaningfully positive.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.regime import RegimeDetector  # noqa: E402
from src.signals import SignalEngine, add_entry_exit_levels  # noqa: E402
from src.backtest import Backtester  # noqa: E402

logger = logging.getLogger("walkforward")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardWindow:
    """One train→test pair plus its backtest outcome."""
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: dict = field(default_factory=dict)
    equity_curve: pd.Series | None = None
    trade_log: pd.DataFrame | None = None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward outcome."""
    windows: List[WalkForwardWindow]
    oos_equity: pd.Series
    oos_returns: pd.Series
    oos_metrics: dict
    summary: pd.DataFrame
    in_sample_metrics: dict | None = None   # for IS vs OOS comparison


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class WalkForward:
    """Walk-forward out-of-sample tester.

    Parameters
    ----------
    train_years : float
        Length of the training window in years. Default 3.0.
    test_months : int
        Length of each test window in months. Default 6.
    step_months : int
        How far to slide the test window forward each iteration. Default
        equals ``test_months`` (non-overlapping windows).
    anchored : bool
        If True (default), the training window starts at the very first
        date and *grows*. If False, the training window has fixed length
        ``train_years`` and slides forward.
    regime_method : {'hmm', 'kmeans', 'ma_crossover'}
        Which regime detector to refit per window.
    initial_capital : float
        Capital each test window starts with. The OOS equity curve is
        then chained so absolute values don't matter — only the
        compounded return path does.
    """

    def __init__(
        self,
        train_years: float = 3.0,
        test_months: int = 6,
        step_months: int | None = None,
        anchored: bool = True,
        regime_method: str = "hmm",
        initial_capital: float = C.INITIAL_CAPITAL,
        transaction_cost: float = C.TRANSACTION_COST,
        max_positions: int = C.MAX_OPEN_POSITIONS,
    ) -> None:
        self.train_years = float(train_years)
        self.test_months = int(test_months)
        self.step_months = int(step_months if step_months is not None else test_months)
        self.anchored = bool(anchored)
        self.regime_method = regime_method
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.max_positions = int(max_positions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        enriched: Dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame | None = None,
        progress_callback=None,
    ) -> WalkForwardResult:
        """Execute the walk-forward.

        Parameters
        ----------
        enriched : dict[str, DataFrame]
            **Feature-enriched** OHLCV per ticker — i.e. what
            :class:`FeatureEngineer` produces. Must NOT yet have regime
            or signal columns (we re-compute those per window so they
            are properly out-of-sample).
        benchmark_df : DataFrame, optional
            Benchmark OHLCV (NIFTY 50). Used inside each window's
            backtest for alpha/beta calculation.
        progress_callback : callable, optional
            Called as ``progress_callback(window_idx, n_windows, msg)``
            after each window completes — used by Streamlit to render
            a progress bar.

        Returns
        -------
        WalkForwardResult
        """
        if not enriched:
            raise ValueError("Empty universe.")

        windows = self._build_windows(enriched)
        if not windows:
            raise ValueError(
                "Date range too short for the requested train/test window sizes."
            )
        logger.info("Walk-forward: %d windows planned.", len(windows))

        oos_equities: list[pd.Series] = []
        for i, w in enumerate(windows, 1):
            logger.info(
                "Window %d/%d  train [%s → %s]  test [%s → %s]",
                i, len(windows),
                w.train_start.date(), w.train_end.date(),
                w.test_start.date(), w.test_end.date(),
            )

            test_data = self._build_test_signaled(enriched, w)
            if not test_data:
                logger.warning("  Window %d skipped — no usable tickers.", i)
                if progress_callback:
                    progress_callback(i, len(windows), "skipped (no data)")
                continue

            bt = Backtester(
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost,
                max_positions=self.max_positions,
            )
            bt.run(test_data, benchmark_df=benchmark_df)
            w.metrics = bt.metrics
            w.equity_curve = bt.equity_curve
            w.trade_log = bt.trade_log
            oos_equities.append(bt.equity_curve)
            if progress_callback:
                progress_callback(i, len(windows),
                                  f"Sharpe {bt.metrics.get('sharpe', 0):.2f}")

        if not oos_equities:
            raise RuntimeError("No walk-forward windows produced output.")

        # Chain the per-window equity curves into a single OOS series.
        oos_equity = self._chain_equities(oos_equities)
        oos_returns = oos_equity.pct_change().fillna(0.0)
        oos_metrics = self._aggregate_oos_metrics(oos_equity, oos_returns)

        summary = self._summary_table(windows)
        return WalkForwardResult(
            windows=windows,
            oos_equity=oos_equity,
            oos_returns=oos_returns,
            oos_metrics=oos_metrics,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Window planning
    # ------------------------------------------------------------------
    def _build_windows(self, enriched: Dict[str, pd.DataFrame]
                       ) -> List[WalkForwardWindow]:
        # Normalise to tz-naive Timestamps to avoid TypeError when mixing
        # tz-aware (fresh yfinance fetch) and tz-naive (CSV-cached) indices
        # in the same set / sort. See data_loader._fetch_one for the
        # upstream normalisation; this is the defensive backup.
        def _strip_tz(d):
            ts = pd.Timestamp(d)
            return ts.tz_localize(None) if ts.tz is not None else ts

        all_dates = sorted({
            _strip_tz(d)
            for df in enriched.values()
            for d in df.index
        })
        if not all_dates:
            return []
        idx = pd.DatetimeIndex(all_dates)
        start, end = idx[0], idx[-1]

        # DateOffset arithmetic so calendar (not "trading day") math is used.
        # That's correct here because we slide windows by calendar months.
        train_off = pd.DateOffset(years=int(self.train_years),
                                  months=int(round((self.train_years - int(self.train_years)) * 12)))
        test_off = pd.DateOffset(months=self.test_months)
        step_off = pd.DateOffset(months=self.step_months)

        first_test_start = start + train_off
        if first_test_start >= end:
            return []

        windows: list[WalkForwardWindow] = []
        cursor = first_test_start
        i = 0
        while cursor + test_off <= end + pd.DateOffset(days=1):  # tolerance
            test_start = cursor
            test_end = min(cursor + test_off, end)
            if self.anchored:
                train_start = start
            else:
                train_start = test_start - train_off
                if train_start < start:
                    train_start = start
            train_end = test_start
            i += 1
            windows.append(WalkForwardWindow(
                idx=i, train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
            ))
            cursor = cursor + step_off
        return windows

    # ------------------------------------------------------------------
    # Per-window pipeline: regime → signals on the test slice
    # ------------------------------------------------------------------
    def _build_test_signaled(self,
                              enriched: Dict[str, pd.DataFrame],
                              w: WalkForwardWindow) -> Dict[str, pd.DataFrame]:
        """For each ticker: fit regime on train, predict on (train+test),
        slice to test only, generate signals + entry/exit levels.

        Why we predict on the *union* of train and test:
            HMM emission scaling and posterior calculation rely on having
            a contiguous index. We slice the relevant portion back out
            after prediction. The model parameters (which is what could
            leak future info) were already fixed by the train-only fit
            inside :meth:`RegimeDetector._hmm_fit_predict`.
        """
        signal_engine = SignalEngine()
        test_data: Dict[str, pd.DataFrame] = {}

        for ticker, df in enriched.items():
            # Slice once with .loc — works even when train_end == test_start.
            train_slice = df.loc[w.train_start:w.train_end]
            full_slice = df.loc[w.train_start:w.test_end]
            test_slice_idx = (full_slice.index > w.train_end) & \
                             (full_slice.index <= w.test_end)
            if len(train_slice) < 100 or test_slice_idx.sum() < 10:
                continue

            try:
                det = RegimeDetector(method=self.regime_method)
                regimed = det.fit_then_predict(train_slice, full_slice)
                signaled = signal_engine.generate(regimed)
                signaled = add_entry_exit_levels(signaled)
                # Keep only test-window rows so the backtester never sees
                # training-window signals.
                test_only = signaled.loc[test_slice_idx]
                if not test_only.empty:
                    test_data[ticker] = test_only
            except Exception as e:
                logger.error("[%s] window %d failed: %s", ticker, w.idx, e)
                continue
        return test_data

    # ------------------------------------------------------------------
    # Stitching + aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _chain_equities(equities: List[pd.Series]) -> pd.Series:
        """Concatenate per-window equity series so each picks up where the
        previous one left off. Each individual window starts at
        ``initial_capital`` (chosen above); we rescale so the OOS curve is
        a single compounded path."""
        if not equities:
            return pd.Series(dtype=float)
        chained = equities[0].copy()
        for eq in equities[1:]:
            if eq.empty:
                continue
            scale = chained.iloc[-1] / eq.iloc[0]
            adjusted = eq * scale
            # Drop the first bar of the new window if it overlaps (the
            # previous window's last bar already covers that date).
            if not chained.empty and adjusted.index[0] <= chained.index[-1]:
                adjusted = adjusted.iloc[1:]
            chained = pd.concat([chained, adjusted])
        chained = chained[~chained.index.duplicated(keep="last")].sort_index()
        return chained

    @staticmethod
    def _aggregate_oos_metrics(equity: pd.Series,
                                 returns: pd.Series) -> dict:
        if len(equity) < 2:
            return {}
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
        ann_vol = float(returns.std() * np.sqrt(C.TRADING_DAYS))
        rf_daily = (1.0 + C.RISK_FREE_RATE) ** (1.0 / C.TRADING_DAYS) - 1.0
        excess = returns - rf_daily
        sharpe = float(excess.mean() / returns.std() * np.sqrt(C.TRADING_DAYS)) \
            if returns.std() > 0 else 0.0
        downside = returns[returns < 0]
        sortino = float(excess.mean() / downside.std() * np.sqrt(C.TRADING_DAYS)) \
            if len(downside) > 1 and downside.std() > 0 else 0.0
        peak = equity.cummax()
        dd = equity / peak - 1.0
        max_dd = float(dd.min())
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
        return {
            "total_return": total_return, "cagr": cagr, "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "max_drawdown": max_dd, "years": years,
            "start": equity.index[0], "end": equity.index[-1],
        }

    @staticmethod
    def _summary_table(windows: List[WalkForwardWindow]) -> pd.DataFrame:
        rows = []
        for w in windows:
            if not w.metrics:
                continue
            rows.append({
                "window": w.idx,
                "train_start": w.train_start.date(),
                "train_end": w.train_end.date(),
                "test_start": w.test_start.date(),
                "test_end": w.test_end.date(),
                "n_trades": w.metrics.get("n_trades", 0),
                "total_return": w.metrics.get("total_return", 0.0),
                "sharpe": w.metrics.get("sharpe", 0.0),
                "max_dd": w.metrics.get("max_drawdown", 0.0),
                "win_rate": w.metrics.get("win_rate", 0.0),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import FeatureEngineer

    rng = np.random.default_rng(11)
    n = 1_500                                  # ~6 years of business days
    idx = pd.bdate_range("2019-01-02", periods=n)
    universe: dict[str, pd.DataFrame] = {}
    for ticker in ["MOCK_A.NS", "MOCK_B.NS", "MOCK_C.NS"]:
        drift = rng.normal(0.0005, 0.0003)
        rets = rng.normal(drift, 0.014, n)
        close = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": 1_000_000.0,
        }, index=idx)
        df["Daily_Return"] = df["Close"].pct_change()
        df["Adj_Return"] = np.log1p(df["Daily_Return"])
        universe[ticker] = FeatureEngineer().compute(df)

    wf = WalkForward(train_years=3.0, test_months=6, step_months=6,
                     anchored=True, regime_method="hmm")
    result = wf.run(universe)
    print(f"\nOOS windows: {len(result.summary)}")
    print(result.summary.to_string(index=False))
    print("\nAggregated OOS metrics:")
    for k, v in result.oos_metrics.items():
        if isinstance(v, float):
            print(f"  {k:<18s} {v:>10.4f}")
        else:
            print(f"  {k:<18s} {v}")
