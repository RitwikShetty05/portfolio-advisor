"""
src/backtest.py
===============

Phase 5 — Event-driven backtester + performance tearsheet.

What it does
------------
Walks day-by-day through every ticker's history in chronological order,
opening and closing positions as signals fire, while respecting:

    * Initial capital  : ₹1,00,000 (configurable)
    * Transaction cost : 0.1% per trade (NSE STT + brokerage + SEBI)
    * Position cap     : 10% of NAV per stock
    * Max positions    : 8 concurrent
    * Stop-loss        : 2× ATR — when a bar's intraday Low touches the
                         stop, the position is closed that bar AT the stop
                         price (the standard backtest fill assumption; it
                         ignores gap-down slippage below the stop, see
                         _check_stops)

Why event-driven (not vectorised)
---------------------------------
A vectorised "Signal × forward-return" backtest is fast but lies — it
assumes you can take every signal with unlimited capital, no slippage,
no position cap, no stop-loss management. Real portfolios are
event-driven: you have finite cash, finite slots, stop-losses can fire
intraday, and one position closing frees capital for the next one.

We pay a small speed cost (~seconds for years × dozens of tickers, fine)
to get **realistic** results. Junior-quant resumes claiming "Sharpe 4 in
backtest" almost always come from a vectorised backtest with look-ahead.

Metrics produced
----------------
* Returns           : Total, CAGR, Annualised Volatility
* Risk-adjusted     : Sharpe, Sortino, Calmar
* Drawdown          : Max Drawdown + date
* Trade statistics  : Win rate, profit factor, avg win/loss,
                      expectancy, max consecutive losses
* Benchmark-relative: Jensen's Alpha, Beta, R², Outperformance

Risk-free rate uses India's 10-year G-sec proxy (``config.RISK_FREE_RATE``).
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

logging.basicConfig(level=C.LOG_LEVEL, format=C.LOG_FORMAT)
logger = logging.getLogger("backtest")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class Position:
    """An open position, mark-to-market in the daily loop."""
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float                    # fractional OK — we trade ₹, not lots
    stop_loss: float
    target_1: float
    target_2: float
    entry_confidence: float
    entry_regime: str


@dataclass
class Trade:
    """A completed round-trip, used for trade statistics."""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float                       # net of transaction costs
    return_pct: float                # P&L relative to gross entry cost
    holding_days: int
    exit_reason: str                 # "stop_loss" | "signal_sell" | "end_of_data"
    entry_confidence: float
    entry_regime: str


@dataclass
class _State:
    """Internal book-keeping for the backtester's daily loop."""
    cash: float
    equity_curve: List[tuple] = field(default_factory=list)
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class Backtester:
    """Event-driven portfolio simulator with full performance reporting."""

    def __init__(
        self,
        initial_capital: float = C.INITIAL_CAPITAL,
        transaction_cost: float = C.TRANSACTION_COST,
        position_size_pct: float = C.POSITION_SIZE_PCT,
        base_allocation: float = C.BASE_ALLOCATION,
        max_positions: int = C.MAX_OPEN_POSITIONS,
        risk_free_rate: float = C.RISK_FREE_RATE,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.txn_cost = float(transaction_cost)
        self.position_size_pct = float(position_size_pct)
        self.base_allocation = float(base_allocation)
        self.max_positions = int(max_positions)
        self.risk_free_rate = float(risk_free_rate)

        # Populated by run()
        self.equity_curve: pd.Series | None = None
        self.daily_returns: pd.Series | None = None
        self.trade_log: pd.DataFrame | None = None
        self.benchmark_returns: pd.Series | None = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    def _position_size(self, nav: float, confidence: float, size_mult: float) -> float:
        """Cash to allocate to a new position. Capped at ``position_size_pct``.

        Sizing model: each position targets the per-stock cap
        ``position_size_pct`` (10% NAV), scaled DOWN by

            * the regime risk multiplier ``size_mult`` (1.0 bull / 0.7
              sideways / 0.5 bear — take less risk as the tape weakens), and
            * a gentle conviction tilt ``(0.5 + 0.5·confidence)`` so a
              stronger composite score commits more capital.

        Why this replaced the old ``base_allocation · confidence · size_mult``
        rule: with ``base_allocation = 0.05`` the product could never approach
        the 10% cap (``confidence`` is the bounded |score|, ≤≈0.66), so every
        position was ~2–4% of NAV and the book sat mostly in cash even when
        fully "loaded". Anchoring to the cap fixes the chronic under-investment
        while the regime multiplier keeps the strategy defensive in drawdowns.
        Conviction also still drives *ordering* in ``_check_buys`` (best ideas
        are funded first when slots or cash bind).
        """
        confidence = max(0.0, min(1.0, confidence))
        size_mult = max(0.0, min(1.0, size_mult))
        conviction = 0.5 + 0.5 * confidence            # in [0.5, 1.0]
        raw = self.position_size_pct * size_mult * conviction
        return nav * min(raw, self.position_size_pct)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        signaled: Dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame | None = None,
    ) -> dict:
        """Run the backtest.

        Parameters
        ----------
        signaled : dict
            ``{ticker: DataFrame}`` of OHLCV + features + regime + signal
            columns (as produced by :func:`add_entry_exit_levels`).
        benchmark_df : DataFrame, optional
            Benchmark OHLCV (e.g. NIFTY 50) for alpha/beta calculations.

        Returns
        -------
        dict
            Full metrics dict; also stored as ``self.metrics``.
        """
        if not signaled:
            raise ValueError("No tickers to backtest.")

        # Build the master trading calendar from the union of all ticker indices.
        # Normalise to tz-naive to avoid TypeError when mixing tz-aware /
        # tz-naive Timestamps in the same set (see data_loader._fetch_one).
        def _strip_tz(d):
            ts = pd.Timestamp(d)
            return ts.tz_localize(None) if ts.tz is not None else ts
        master_index = sorted({
            _strip_tz(d)
            for df in signaled.values()
            for d in df.index
        })
        master_index = pd.DatetimeIndex(master_index)
        logger.info("Backtesting %d tickers over %d trading days (%s → %s)",
                    len(signaled), len(master_index),
                    master_index.min().date(), master_index.max().date())

        state = _State(cash=self.initial_capital)

        # Idle cash earns the risk-free rate (Indian liquid-fund / T-bill
        # proxy). Modelling cash at 0% systematically penalises a regime-aware
        # strategy for its single biggest *feature* — stepping aside into cash
        # during bear/sideways tapes. A real book parks that cash at ≈the
        # risk-free rate, so we accrue it daily. This is a realism correction,
        # not a tuning knob: it is internally consistent with the same rate
        # used in the Sharpe/alpha denominators.
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / C.TRADING_DAYS) - 1.0

        for date in master_index:
            # 0. Accrue one day of risk-free interest on the idle cash balance.
            state.cash *= (1.0 + rf_daily)
            # 1. Mark-to-market open positions to know our current NAV.
            nav = self._compute_nav(state, signaled, date)

            # 2. Risk management first — stops fire BEFORE we look at new signals.
            self._check_stops(state, signaled, date)
            # 3. Then check for sell signals on currently held positions.
            self._check_sells(state, signaled, date)
            # 4. Finally, deploy capital on new buy signals.
            self._check_buys(state, signaled, date, nav)

            # 5. Re-compute NAV after the day's events, record equity.
            end_nav = self._compute_nav(state, signaled, date)
            state.equity_curve.append((date, end_nav))

        # Close anything still open at the end of the test window.
        self._close_remaining(state, signaled, master_index[-1])

        # Build outputs.
        self.equity_curve = pd.Series(
            {d: e for d, e in state.equity_curve}, name="Equity"
        ).sort_index()
        self.daily_returns = self.equity_curve.pct_change().fillna(0.0)
        self.trade_log = self._build_trade_log(state.trades)

        # Benchmark prep — align to portfolio dates and compute returns.
        if benchmark_df is not None and "Close" in benchmark_df.columns:
            bench_close = benchmark_df["Close"].reindex(self.equity_curve.index).ffill()
            self.benchmark_returns = bench_close.pct_change().fillna(0.0)
        else:
            self.benchmark_returns = None

        self.metrics = self._compute_metrics()
        logger.info("Backtest complete: %d trades, final NAV ₹%s",
                    len(state.trades), f"{self.equity_curve.iloc[-1]:,.0f}")
        return self.metrics

    # ------------------------------------------------------------------
    # Daily-loop primitives
    # ------------------------------------------------------------------
    def _compute_nav(self, state: _State, signaled: Dict[str, pd.DataFrame],
                     date: pd.Timestamp) -> float:
        """Cash + Σ(position value at today's close)."""
        nav = state.cash
        for ticker, pos in state.positions.items():
            df = signaled.get(ticker)
            if df is None or date not in df.index:
                # Stale price — use entry price as conservative mark.
                nav += pos.shares * pos.entry_price
                continue
            close = df.at[date, "Close"]
            if pd.isna(close):
                close = pos.entry_price
            nav += pos.shares * close
        return nav

    def _check_stops(self, state: _State, signaled: Dict[str, pd.DataFrame],
                     date: pd.Timestamp) -> None:
        """Close any position whose intraday Low touched its stop."""
        # Iterate over a snapshot — _close_position mutates state.positions.
        for ticker in list(state.positions):
            pos = state.positions[ticker]
            df = signaled.get(ticker)
            if df is None or date not in df.index:
                continue
            low = df.at[date, "Low"]
            if pd.isna(low):
                continue
            if low <= pos.stop_loss:
                # Fill assumption: executed AT the stop level. This is the
                # standard backtest convention — exact for any day that merely
                # trades through the stop, mildly optimistic on overnight
                # gap-downs (a real stop-market order would fill at the open,
                # below the stop). With large-cap NSE names and a 2×ATR buffer
                # the gap error is second-order; flagged here for honesty.
                self._close_position(state, ticker, date, pos.stop_loss, "stop_loss")

    def _check_sells(self, state: _State, signaled: Dict[str, pd.DataFrame],
                     date: pd.Timestamp) -> None:
        """Close on SELL signal at today's close."""
        for ticker in list(state.positions):
            df = signaled.get(ticker)
            if df is None or date not in df.index:
                continue
            sig = df.at[date, "Signal"]
            if sig == -1:
                close = df.at[date, "Close"]
                if not pd.isna(close):
                    self._close_position(state, ticker, date, float(close), "signal_sell")

    def _check_buys(self, state: _State, signaled: Dict[str, pd.DataFrame],
                    date: pd.Timestamp, nav: float) -> None:
        """Open new positions on BUY signals, subject to cash & slot limits."""
        if len(state.positions) >= self.max_positions:
            return

        # Rank today's BUY signals by confidence — best ideas get capital first.
        candidates: list[tuple[float, str, pd.Series]] = []
        for ticker, df in signaled.items():
            if ticker in state.positions:
                continue
            if date not in df.index:
                continue
            row = df.loc[date]
            if row.get("Signal", 0) != 1:
                continue
            conf = float(row.get("Confidence", 0.0))
            if conf <= 0.0:
                continue
            candidates.append((conf, ticker, row))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for conf, ticker, row in candidates:
            if len(state.positions) >= self.max_positions:
                break

            size_mult = float(row.get("Size_Mult", 1.0))
            cash_to_deploy = self._position_size(nav, conf, size_mult)
            # Need to leave a cushion for transaction cost — solve
            # cash_to_deploy = shares * price * (1 + txn_cost).
            entry_price = float(row["Close"])
            if entry_price <= 0 or pd.isna(entry_price):
                continue

            gross_cost = cash_to_deploy
            shares = gross_cost / (entry_price * (1.0 + self.txn_cost))
            if shares <= 0:
                continue
            total_cost = shares * entry_price * (1.0 + self.txn_cost)
            if total_cost > state.cash:
                continue

            state.cash -= total_cost
            state.positions[ticker] = Position(
                ticker=ticker,
                entry_date=date,
                entry_price=entry_price,
                shares=shares,
                stop_loss=float(row.get("Stop_Loss", entry_price * 0.95)),
                target_1=float(row.get("Target_1", entry_price * 1.05)),
                target_2=float(row.get("Target_2", entry_price * 1.10)),
                entry_confidence=conf,
                entry_regime=str(row.get("Regime_Label", "SIDEWAYS")),
            )

    def _close_position(self, state: _State, ticker: str, date: pd.Timestamp,
                        exit_price: float, reason: str) -> None:
        """Close a position, record the trade, return cash to the pool."""
        pos = state.positions.pop(ticker)
        proceeds = pos.shares * exit_price * (1.0 - self.txn_cost)
        gross_entry = pos.shares * pos.entry_price
        gross_exit = pos.shares * exit_price
        entry_cost = gross_entry * (1.0 + self.txn_cost)
        exit_proceeds = gross_exit * (1.0 - self.txn_cost)
        pnl = exit_proceeds - entry_cost
        ret_pct = pnl / entry_cost if entry_cost > 0 else 0.0
        state.cash += proceeds
        state.trades.append(Trade(
            ticker=ticker,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            return_pct=ret_pct,
            holding_days=int((date - pos.entry_date).days),
            exit_reason=reason,
            entry_confidence=pos.entry_confidence,
            entry_regime=pos.entry_regime,
        ))

    def _close_remaining(self, state: _State, signaled: Dict[str, pd.DataFrame],
                         last_date: pd.Timestamp) -> None:
        """End-of-test: close any positions that survived to the final bar."""
        for ticker in list(state.positions):
            df = signaled.get(ticker)
            if df is None or last_date not in df.index:
                continue
            close = df.at[last_date, "Close"]
            if pd.isna(close):
                continue
            self._close_position(state, ticker, last_date, float(close), "end_of_data")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    @staticmethod
    def _build_trade_log(trades: List[Trade]) -> pd.DataFrame:
        if not trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in trades])

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_metrics(self) -> dict:
        equity = self.equity_curve
        rets = self.daily_returns
        n = len(rets)
        if n < 2:
            return {}

        years = (equity.index[-1] - equity.index[0]).days / 365.25
        years = max(years, 1e-9)

        # Returns
        total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1) if years > 0 else 0.0
        ann_vol = float(rets.std() * np.sqrt(C.TRADING_DAYS))

        # Daily risk-free conversion (compound).
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / C.TRADING_DAYS) - 1.0
        excess = rets - rf_daily

        sharpe = float(excess.mean() / rets.std() * np.sqrt(C.TRADING_DAYS)) \
            if rets.std() > 0 else 0.0
        downside = rets[rets < 0]
        sortino = float(excess.mean() / downside.std() * np.sqrt(C.TRADING_DAYS)) \
            if len(downside) > 1 and downside.std() > 0 else 0.0

        # Drawdown
        cum_peak = equity.cummax()
        dd = equity / cum_peak - 1.0
        max_dd = float(dd.min())
        max_dd_date = dd.idxmin()
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

        # Trade-level statistics
        tl = self.trade_log
        if tl is not None and not tl.empty:
            wins = tl[tl["pnl"] > 0]["pnl"]
            losses = tl[tl["pnl"] <= 0]["pnl"]
            n_trades = int(len(tl))
            n_wins = int(len(wins))
            win_rate = n_wins / n_trades if n_trades > 0 else 0.0
            avg_win = float(wins.mean()) if not wins.empty else 0.0
            avg_loss = float(losses.mean()) if not losses.empty else 0.0
            profit_factor = float(wins.sum() / abs(losses.sum())) \
                if not losses.empty and losses.sum() < 0 else float("inf") if not wins.empty else 0.0
            expectancy = float(tl["pnl"].mean()) if n_trades > 0 else 0.0
            # Max consecutive losing trades
            is_loss = (tl["pnl"] <= 0).astype(int).values
            max_consec = 0
            curr = 0
            for v in is_loss:
                curr = curr + 1 if v else 0
                max_consec = max(max_consec, curr)
        else:
            n_trades = n_wins = max_consec = 0
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0.0

        # Benchmark-relative metrics
        alpha = beta = r2 = outperf = float("nan")
        if self.benchmark_returns is not None and len(self.benchmark_returns) > 30:
            br = self.benchmark_returns
            # Align — same index by construction.
            x = br - rf_daily
            y = rets - rf_daily
            var_x = float(x.var())
            if var_x > 0:
                beta = float(x.cov(y) / var_x)
                # Jensen's alpha — daily intercept, annualised.
                alpha_daily = float(y.mean() - beta * x.mean())
                alpha = alpha_daily * C.TRADING_DAYS
                # R² of the regression.
                ss_res = float(((y - beta * x - alpha_daily) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            # Outperformance vs buy-and-hold benchmark.
            bench_total = float((1.0 + br).prod() - 1.0)
            outperf = total_return - bench_total

        return {
            "start": equity.index[0],
            "end": equity.index[-1],
            "years": years,
            "initial_capital": self.initial_capital,
            "final_nav": float(equity.iloc[-1]),
            "total_return": total_return,
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "max_drawdown_date": max_dd_date,
            "n_trades": n_trades,
            "n_wins": n_wins,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "max_consecutive_losses": max_consec,
            "alpha": alpha,
            "beta": beta,
            "r_squared": r2,
            "outperformance": outperf,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def tearsheet(self) -> None:
        """Print a Bloomberg-style performance report."""
        if not self.metrics:
            print("Run backtest first.")
            return
        m = self.metrics
        line = "=" * 60
        print(line)
        print(f"  PERFORMANCE TEARSHEET   ({m['start'].date()} → {m['end'].date()})")
        print(line)
        print(f"  Initial Capital       : ₹{m['initial_capital']:>14,.0f}")
        print(f"  Final NAV             : ₹{m['final_nav']:>14,.0f}")
        print(f"  Total Return          :  {m['total_return']*100:>14.2f}%")
        print(f"  CAGR                  :  {m['cagr']*100:>14.2f}%")
        print(f"  Annualised Volatility :  {m['ann_vol']*100:>14.2f}%")
        print()
        print(f"  Sharpe Ratio          :  {m['sharpe']:>14.2f}")
        print(f"  Sortino Ratio         :  {m['sortino']:>14.2f}")
        print(f"  Calmar Ratio          :  {m['calmar']:>14.2f}")
        print(f"  Max Drawdown          :  {m['max_drawdown']*100:>14.2f}%  ({m['max_drawdown_date'].date()})")
        print()
        print(f"  Trades                :  {m['n_trades']:>14d}")
        print(f"  Win Rate              :  {m['win_rate']*100:>14.2f}%")
        print(f"  Profit Factor         :  {m['profit_factor']:>14.2f}")
        print(f"  Avg Win  / Avg Loss   : ₹{m['avg_win']:>10,.0f}  /  ₹{m['avg_loss']:,.0f}")
        print(f"  Expectancy / trade    : ₹{m['expectancy']:>14,.0f}")
        print(f"  Max Consec. Losses    :  {m['max_consecutive_losses']:>14d}")
        print()
        if not np.isnan(m["alpha"]):
            print(f"  Alpha (annualised)    :  {m['alpha']*100:>14.2f}%")
            print(f"  Beta                  :  {m['beta']:>14.2f}")
            print(f"  R²                    :  {m['r_squared']:>14.2f}")
            print(f"  Outperformance        :  {m['outperformance']*100:>14.2f}%")
        print(line)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_drawdown_series(self) -> pd.Series:
        """Continuous drawdown series for plotting."""
        if self.equity_curve is None:
            raise RuntimeError("Run backtest first.")
        peak = self.equity_curve.cummax()
        return self.equity_curve / peak - 1.0

    def get_rolling_sharpe(self, window: int = 126) -> pd.Series:
        """Rolling Sharpe (default 6-month window) — useful for spotting edge decay."""
        if self.daily_returns is None:
            raise RuntimeError("Run backtest first.")
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / C.TRADING_DAYS) - 1.0
        excess = self.daily_returns - rf_daily
        roll_mean = excess.rolling(window).mean()
        roll_std = self.daily_returns.rolling(window).std()
        return (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan) * np.sqrt(C.TRADING_DAYS)


# ---------------------------------------------------------------------------
# Smoke test — fully self-contained on synthetic data.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from features import FeatureEngineer
    from regime import RegimeDetector
    from signals import SignalEngine, add_entry_exit_levels

    rng = np.random.default_rng(2)
    n = 600
    idx = pd.bdate_range("2022-01-03", periods=n)
    universe = {}
    for ticker in ["MOCK_A.NS", "MOCK_B.NS", "MOCK_C.NS"]:
        drift = rng.normal(0.0006, 0.0002)
        rets = rng.normal(drift, 0.014, n)
        close = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "Low":  close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }, index=idx)
        df["Daily_Return"] = df["Close"].pct_change()
        df["Adj_Return"] = np.log1p(df["Daily_Return"])
        df = FeatureEngineer().compute(df)
        df = RegimeDetector(method="hmm").fit_transform(df)
        df = SignalEngine().generate(df)
        df = add_entry_exit_levels(df)
        universe[ticker] = df

    bt = Backtester()
    bt.run(universe, benchmark_df=None)
    bt.tearsheet()
