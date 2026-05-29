"""
src/data_loader.py
==================

Phase 1 — Market data acquisition layer.

What this module is responsible for
-----------------------------------
1.  Pulling historical OHLCV bars for a universe of NSE stocks plus the
    NIFTY 50 benchmark from yfinance.
2.  Validating each ticker against a 4-step data-quality gate so a single
    bad ticker (e.g. a recent IPO, a stock that was suspended) doesn't
    poison downstream feature/signal/backtest code.
3.  Caching to disk so repeated runs don't hammer the Yahoo Finance API
    (and we don't get rate-limited or, worse, IP-blocked).
4.  Producing a wide, date-aligned close-price matrix used by the
    portfolio analyser to compute the covariance matrix.

Why `auto_adjust=True` matters
------------------------------
Without dividend/split adjustment, a 1:1 bonus issue looks like a -50%
overnight crash and a ₹1 dividend looks like noise — both will wreck any
returns calculation and produce nonsense signals. Always use
adjusted prices for *statistical* work. yfinance handles this for us when
``auto_adjust=True`` is passed.

Why the 4-step quality gate exists
----------------------------------
Real market data is messy. A few examples from production:
    * A stock relists after a corporate action and yfinance returns 30
      bars instead of 1500.
    * A stale Yahoo ticker has 40% missing days because the listing was
      delisted mid-period.
    * A penny-stock has zero/negative prices on illiquid days.
    * A data error reports a "1000x" overnight move.
Each of those would silently produce garbage metrics if we let them
through. The gate rejects them with a clear log message.

Why log returns AND simple returns
----------------------------------
- Log returns (``Adj_Return``) are time-additive and approximately
  normally distributed — used for volatility, Sharpe, correlation,
  covariance, and HMM training.
- Simple returns (``Daily_Return``) are wealth-additive and the correct
  thing to multiply by position size to get P&L — used in the backtester.
Mixing them up is one of the classic junior-quant mistakes; we compute
both up-front so downstream code never has to guess.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:  # pragma: no cover - hard dependency
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from e

# Project config (always read parameters from config, never hard-code).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=C.LOG_LEVEL, format=C.LOG_FORMAT)
logger = logging.getLogger("data_loader")


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class DataLoader:
    """Fetch, validate, cache, and serve OHLCV data.

    Parameters
    ----------
    universe : iterable of str, optional
        Tickers to load. Defaults to ``config.UNIVERSE``.
    start, end : str (YYYY-MM-DD), optional
        Date range. Defaults to ``config.START_DATE`` / ``config.END_DATE``.
    benchmark : str, optional
        Benchmark ticker (e.g. NIFTY 50). Defaults to ``config.BENCHMARK``.
    use_cache : bool, optional
        If True, read from ``data/raw/`` when a fresh CSV is available.
    cache_max_age_days : int, optional
        Refetch if the cached file is older than this. Defaults to
        ``config.CACHE_MAX_AGE_DAYS``.

    Public methods
    --------------
    load_universe()       → dict[str, pd.DataFrame]
    load_benchmark()      → pd.DataFrame
    get_aligned_close()   → pd.DataFrame  (wide: dates × tickers)
    get_quality_report()  → pd.DataFrame
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(
        self,
        universe: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        benchmark: str | None = None,
        use_cache: bool | None = None,
        cache_max_age_days: int | None = None,
    ) -> None:
        self.universe: list[str] = list(universe) if universe is not None else list(C.UNIVERSE)
        self.start: str = start or C.START_DATE
        self.end: str = end or C.END_DATE
        self.benchmark: str = benchmark or C.BENCHMARK
        self.use_cache: bool = C.CACHE_ENABLED if use_cache is None else use_cache
        self.cache_max_age_days: int = (
            cache_max_age_days if cache_max_age_days is not None else C.CACHE_MAX_AGE_DAYS
        )

        # Populated by load_*()
        self.data: dict[str, pd.DataFrame] = {}
        self.benchmark_df: pd.DataFrame | None = None
        self.quality: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, ticker: str) -> Path:
        # Sanitise the ticker for the filesystem (e.g. "M&M.NS" → "M_M.NS").
        safe = ticker.replace("&", "_").replace("/", "_").replace("^", "_")
        return C.RAW_DIR / f"{safe}_{self.start}_{self.end}.csv"

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=self.cache_max_age_days)

    @staticmethod
    def _most_recent_business_day() -> pd.Timestamp:
        """Most recent Mon–Fri date (approximate — ignores exchange holidays).

        Used as a sanity check on whether the cached CSV covers today's
        likely data. Holidays produce false positives (we'd refetch a Monday
        that wasn't actually a trading day) but yfinance just returns the
        same data, so it's harmless.
        """
        t = pd.Timestamp.now().normalize()
        while t.weekday() >= 5:                       # Sat, Sun
            t = t - pd.Timedelta(days=1)
        return t

    def _read_cache(self, ticker: str) -> pd.DataFrame | None:
        path = self._cache_path(ticker)
        if not (self.use_cache and self._cache_is_fresh(path)):
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Defensive tz strip — older cache CSVs may have been written
            # while tz-aware, and pandas's parse_dates round-trip can
            # restore that. Force tz-naive so downstream set/sort operations
            # don't TypeError on mixed Timestamps.
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_localize(None)
            # "Today coverage" check — if the cache stops before the most
            # recent business day, treat it as stale and refetch. This is
            # what makes today's signal show up in the heat-map during /
            # after market hours instead of being permanently one day behind.
            if not df.empty:
                target = self._most_recent_business_day()
                if df.index.max() < target:
                    logger.info(
                        "[%s] cache covers up to %s, target %s — refetching",
                        ticker, df.index.max().date(), target.date(),
                    )
                    return None
            logger.debug("[%s] cache hit (%s)", ticker, path.name)
            return df
        except Exception as e:  # malformed cache — just refetch
            logger.warning("[%s] cache read failed (%s); refetching", ticker, e)
            return None

    def _write_cache(self, ticker: str, df: pd.DataFrame) -> None:
        if not self.use_cache:
            return
        try:
            df.to_csv(self._cache_path(ticker))
        except Exception as e:  # caching is best-effort, never fatal
            logger.warning("[%s] cache write failed (%s)", ticker, e)

    # ------------------------------------------------------------------
    # yfinance fetch
    # ------------------------------------------------------------------
    def _fetch_one(self, ticker: str) -> pd.DataFrame:
        """Pull a single ticker, with cache → API fallback."""
        cached = self._read_cache(ticker)
        if cached is not None and not cached.empty:
            return cached

        # yfinance treats `end` as EXCLUSIVE for daily bars (the bar dated
        # `end` is NOT included). To get today's bar in the heat-map and
        # signal panel during/after market hours, bump end by one calendar
        # day before calling yfinance.
        try:
            end_inclusive = (pd.Timestamp(self.end)
                             + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            end_inclusive = self.end

        logger.info("[%s] fetching from yfinance %s → %s (incl.)",
                    ticker, self.start, end_inclusive)
        # auto_adjust=True → split & dividend adjusted Close, High, Low, Open.
        # progress=False keeps logs clean when looping over many tickers.
        #
        # Robustness note: yfinance's `download()` endpoint occasionally returns
        # empty DataFrames for indices (^NSEI, ^GSPC, etc.) and sometimes for
        # individual tickers when Yahoo throttles. We try `download()` first
        # (faster, supports threading), then fall back to `Ticker().history()`
        # (different code path — slower but more reliable for indices and
        # less prone to silent empty responses).
        df = pd.DataFrame()
        try:
            df = yf.download(
                tickers=ticker,
                start=self.start,
                end=end_inclusive,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:
            logger.warning("[%s] yf.download failed (%s); trying Ticker().history()", ticker, e)

        if df is None or df.empty:
            logger.info("[%s] yf.download returned empty; falling back to Ticker().history()", ticker)
            try:
                df = yf.Ticker(ticker).history(
                    start=self.start,
                    end=end_inclusive,
                    auto_adjust=True,
                )
            except Exception as e:
                raise ValueError(
                    f"yfinance returned empty data for {ticker} "
                    f"(both download() and Ticker().history() failed: {e})"
                ) from e

        if df is None or df.empty:
            raise ValueError(f"yfinance returned empty data for {ticker}")

        # yfinance sometimes returns a MultiIndex for single tickers in newer
        # versions — flatten it for a stable schema.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only what we need.
        df = df[[c for c in self.REQUIRED_COLUMNS if c in df.columns]].copy()
        # Normalise to tz-naive timestamps. yfinance returns tz-AWARE indices
        # on some endpoints (download() vs Ticker().history()) and tz-NAIVE
        # on others, plus our CSV cache round-trip strips tz info. Mixing
        # tz-aware and tz-naive Timestamps in a set/sorted() raises TypeError
        # (see Streamlit Cloud deployment).
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        df.index.name = "Date"

        self._write_cache(ticker, df)
        return df

    # ------------------------------------------------------------------
    # Quality gate
    # ------------------------------------------------------------------
    def _validate(self, ticker: str, df: pd.DataFrame) -> tuple[bool, dict]:
        """Return ``(passed, report)`` for one ticker.

        Four checks (any failure → reject the ticker):
            1. Minimum row count (``config.MIN_TRADING_DAYS``).
            2. Missing-value fraction (``config.MAX_MISSING_PCT``).
            3. No zero/negative prices.
            4. No suspect single-day returns (``config.MAX_DAILY_RETURN``).
        """
        report = {
            "rows": int(len(df)),
            "missing_pct": float(df[self.REQUIRED_COLUMNS].isna().any(axis=1).mean())
            if not df.empty else 1.0,
            "zero_or_neg_prices": int((df["Close"] <= 0).sum()) if "Close" in df else 0,
            "extreme_moves": 0,
            "passed": False,
            "reason": "",
        }

        if report["rows"] < C.MIN_TRADING_DAYS:
            report["reason"] = f"too few rows ({report['rows']} < {C.MIN_TRADING_DAYS})"
            return False, report

        if report["missing_pct"] > C.MAX_MISSING_PCT:
            report["reason"] = f"missing pct {report['missing_pct']:.1%} > {C.MAX_MISSING_PCT:.1%}"
            return False, report

        if report["zero_or_neg_prices"] > 0:
            report["reason"] = f"{report['zero_or_neg_prices']} zero/negative close prices"
            return False, report

        # Extreme single-day return check — uses simple returns.
        with np.errstate(divide="ignore", invalid="ignore"):
            simple_ret = df["Close"].pct_change()
        extreme = int((simple_ret.abs() > C.MAX_DAILY_RETURN).sum())
        report["extreme_moves"] = extreme
        # We *flag* extreme moves but don't auto-reject (Indian markets have
        # legitimate 20%+ circuit days, though >50% is essentially always a
        # data error). Configurable in `config.MAX_DAILY_RETURN`.
        if extreme > 0:
            logger.warning("[%s] %d suspect day(s) with |return| > %.0f%%",
                           ticker, extreme, C.MAX_DAILY_RETURN * 100)

        report["passed"] = True
        return True, report

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add both simple and log returns. Drop the first row (NaN)."""
        out = df.copy()
        # Forward-fill tiny gaps (e.g. exchange holiday inconsistencies)
        # before computing returns. Limit=2 prevents covering a real outage.
        out[["Open", "High", "Low", "Close"]] = (
            out[["Open", "High", "Low", "Close"]].ffill(limit=2)
        )
        out["Daily_Return"] = out["Close"].pct_change()
        # log(1 + r) — additive over time, the right input for statistics.
        out["Adj_Return"] = np.log1p(out["Daily_Return"])
        return out.dropna(subset=["Close"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_universe(self) -> dict[str, pd.DataFrame]:
        """Fetch every ticker in the universe. Bad tickers are skipped, not fatal."""
        self.data = {}
        self.quality = {}
        for ticker in self.universe:
            try:
                df = self._fetch_one(ticker)
            except Exception as e:
                logger.error("[%s] fetch failed: %s", ticker, e)
                self.quality[ticker] = {"passed": False, "reason": f"fetch error: {e}",
                                        "rows": 0, "missing_pct": 1.0,
                                        "zero_or_neg_prices": 0, "extreme_moves": 0}
                continue

            passed, report = self._validate(ticker, df)
            self.quality[ticker] = report
            if not passed:
                logger.warning("[%s] failed quality gate: %s", ticker, report["reason"])
                continue

            self.data[ticker] = self._add_returns(df)
            logger.info("[%s] loaded %d bars (%s → %s)",
                        ticker, len(self.data[ticker]),
                        self.data[ticker].index.min().date(),
                        self.data[ticker].index.max().date())

        logger.info("Universe loaded: %d/%d tickers passed",
                    len(self.data), len(self.universe))
        return self.data

    def load_benchmark(self) -> pd.DataFrame:
        """Fetch the benchmark series (e.g. NIFTY 50)."""
        df = self._fetch_one(self.benchmark)
        self.benchmark_df = self._add_returns(df)
        logger.info("Benchmark %s loaded: %d bars", self.benchmark, len(self.benchmark_df))
        return self.benchmark_df

    def get_aligned_close(self) -> pd.DataFrame:
        """Wide DataFrame of close prices, date-aligned across tickers.

        Rows with any missing values are dropped — this ensures every
        column has data on every day, which is what covariance / correlation
        / portfolio-vol formulas assume.
        """
        if not self.data:
            raise RuntimeError("Call load_universe() before get_aligned_close().")
        closes = pd.DataFrame({t: df["Close"] for t, df in self.data.items()})
        closes = closes.sort_index().dropna(how="any")
        return closes

    def get_quality_report(self) -> pd.DataFrame:
        """Per-ticker validation summary as a DataFrame."""
        if not self.quality:
            raise RuntimeError("Call load_universe() before get_quality_report().")
        return (
            pd.DataFrame(self.quality).T
            .reset_index().rename(columns={"index": "ticker"})
        )


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_universe()
    bench = loader.load_benchmark()

    print("\n=== Quality report ===")
    print(loader.get_quality_report().to_string(index=False))

    if data:
        sample = next(iter(data))
        print(f"\n=== Sample ({sample}) — last 3 rows ===")
        print(data[sample].tail(3))
        print(f"\nAligned close matrix shape: {loader.get_aligned_close().shape}")
        if bench is not None:
            print(f"Benchmark {loader.benchmark} bars: {len(bench)}")
