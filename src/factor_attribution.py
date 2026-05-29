"""
src/factor_attribution.py
=========================

Phase 7d — Multi-factor performance attribution.

Why this module exists
----------------------
A backtest that reports "Sharpe 1.2, alpha 4.6% vs NIFTY 50" hides a
critical question that every senior quant asks:

    *Is your alpha actually skill, or did you accidentally buy
    small-caps in a small-cap rally?*

A 1-factor CAPM regression (alpha + beta·MKT) only checks whether your
strategy beat *the market*. But systematic exposures to **size** (small
beats large), **value**, **momentum**, **sector tilts**, and so on are
all *known* drivers that you can earn passively. If your "alpha" is
really just a +0.4 exposure to the size factor times the year-on-year
midcap rally, you didn't have skill — you had luck (and a long midcap
tilt).

Methodology
-----------
We run a multivariate OLS regression of strategy excess returns on a
chosen set of factor excess returns:

    r_p − rf = α + β₁(r₁−rf) + β₂(r₂−rf) + … + ε

Then we report:

    * **α (intercept)** — pure, residual alpha after all factor exposures
      are accounted for. With HAC standard errors and a t-test.
    * **βᵢ** — exposure to each factor (units: 1 = 100% loading).
    * **R²** and **adjusted R²** — how much of the variance is explained
      by the factors.
    * **Attribution decomposition**: β_i · ann_mean(r_i) = annualised
      return contribution from factor i. Summing these + the alpha
      reconstructs ≈ the strategy's mean return.

Indian-market factor proxies (via yfinance)
-------------------------------------------
US researchers have polished Fama-French factor series available on
Kenneth French's website. For NSE there is no such canonical set, so we
construct proxies from publicly-quoted indices that yfinance does serve:

    Market (MKT)             — ^NSEI       — NIFTY 50
    Size (mid-cap minus large) — ^NSEMDCP50  — NIFTY Midcap 50
    IT-sector tilt           — ^CNXIT      — NIFTY IT
    Banking-sector tilt      — ^NSEBANK    — NIFTY Bank

The **size** factor is constructed as `return(NSEMDCP50) − return(NSEI)`
— a "small-minus-big" return spread analogous to Fama-French SMB.

Limitations (be honest about these in interviews)
-------------------------------------------------
* These proxies aren't orthogonalised the way the canonical FF factors
  are. Real Fama-French factors are constructed as zero-cost long-short
  portfolios sorted on book/market, market cap, etc.
* yfinance occasionally rate-limits Indian-index tickers; the module
  fails *gracefully* if a factor can't be fetched, dropping it from the
  regression and noting it.
* With only ~4 factors and ~1500 daily observations, statistical power
  is fine but the factors themselves are correlated (especially during
  crashes). Watch for high VIFs in production.

Reference
---------
* Fama, E. F., & French, K. R. (1993). Common Risk Factors in the
  Returns on Stocks and Bonds. *Journal of Financial Economics* 33.
* Newey, W. K., & West, K. D. (1987). HAC standard errors.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy required") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402

logger = logging.getLogger("factor_attribution")


# Default Indian-market factor proxies.
INDIAN_FACTOR_PROXIES: dict[str, dict] = {
    "Market":   {"ticker": "^NSEI",      "transform": "excess",
                 "label": "NIFTY 50 (market factor)"},
    "Size":     {"ticker": "^NSEMDCP50", "transform": "smb",
                 "label": "Midcap − Large (SMB-style)"},
    "IT":       {"ticker": "^CNXIT",     "transform": "excess",
                 "label": "NIFTY IT (sector tilt)"},
    "Banking":  {"ticker": "^NSEBANK",   "transform": "excess",
                 "label": "NIFTY Bank (sector tilt)"},
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FactorAttributionResult:
    """Outcome of a single factor regression."""
    alpha_daily: float
    alpha_annual: float
    alpha_se: float
    alpha_t: float
    alpha_p: float
    alpha_significant_5pct: bool
    alpha_significant_1pct: bool

    factor_names: list[str]
    betas: dict[str, float]
    se_betas: dict[str, float]
    t_stats: dict[str, float]
    p_values: dict[str, float]

    r_squared: float
    adj_r_squared: float

    # Attribution: annualised return contribution from each factor.
    attribution_annual: dict[str, float]    # β_i × ann_mean(r_i)
    strategy_mean_annual: float             # for sanity check
    unexplained_annual: float               # mean(residual) annualised

    n_obs: int
    hac_lags: int
    factors_dropped: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multivariate OLS with Newey-West HAC standard errors
# ---------------------------------------------------------------------------
def _ols_hac(y: np.ndarray, X: np.ndarray,
             hac_lags: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS with Newey-West HAC covariance.

    Returns
    -------
    coef : (k,) array
    var_coef : (k, k) HAC covariance matrix
    hac_lags : int actually used
    """
    n, k = X.shape
    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        hac_lags = max(hac_lags, 1)

    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef

    # Bartlett-kernel HAC.
    omega = (X.T * (resid ** 2)) @ X
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1)
        gamma = (X[lag:].T * (resid[lag:] * resid[:-lag])) @ X[:-lag]
        omega += w * (gamma + gamma.T)

    var_coef = XtX_inv @ omega @ XtX_inv
    return coef, var_coef, hac_lags


# ---------------------------------------------------------------------------
# Factor engine
# ---------------------------------------------------------------------------
class FactorAttribution:
    """Run a multi-factor regression against pre-defined or custom factors.

    Parameters
    ----------
    factor_proxies : dict, optional
        Map ``factor_name -> {'ticker': str, 'transform': str, 'label': str}``.
        Defaults to :data:`INDIAN_FACTOR_PROXIES`.
    risk_free_rate : float
        Annualised risk-free rate. Defaults to ``config.RISK_FREE_RATE``.
    cache_loader : DataLoader, optional
        If provided, factor tickers are fetched through the existing
        DataLoader (so its on-disk CSV cache is reused). Otherwise a
        fresh loader is created.
    """

    def __init__(
        self,
        factor_proxies: dict | None = None,
        risk_free_rate: float = C.RISK_FREE_RATE,
        cache_loader: DataLoader | None = None,
    ) -> None:
        self.factor_proxies = dict(factor_proxies or INDIAN_FACTOR_PROXIES)
        self.risk_free_rate = float(risk_free_rate)
        self.cache_loader = cache_loader
        self.factor_returns_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Factor data loading
    # ------------------------------------------------------------------
    def fetch_factors(self, start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
        """Fetch factor price series and convert to factor returns.

        Returns
        -------
        factor_returns : DataFrame
            Daily simple returns per factor (columns = factor names).
        dropped : list[str]
            Factors that failed to fetch and were skipped.
        """
        # Use the supplied DataLoader if any (cache reuse), else a fresh one.
        loader = self.cache_loader or DataLoader(
            universe=[v["ticker"] for v in self.factor_proxies.values()],
            start=start, end=end,
        )

        raw_closes: dict[str, pd.Series] = {}
        dropped: list[str] = []
        for name, meta in self.factor_proxies.items():
            ticker = meta["ticker"]
            try:
                df = loader._fetch_one(ticker)   # noqa: SLF001
                close = df["Close"].copy()
                close.index = pd.to_datetime(close.index)
                raw_closes[name] = close
            except Exception as e:
                logger.warning("Factor %s (%s) skipped: %s", name, ticker, e)
                dropped.append(name)

        if not raw_closes:
            raise RuntimeError(
                "No factor proxies could be fetched. Check yfinance "
                "connectivity / cache."
            )

        # Align all on the union of dates, then drop any rows with NaN.
        prices = pd.concat(raw_closes, axis=1).sort_index().ffill().dropna(how="any")
        rets = prices.pct_change().dropna()

        # Apply transforms (Size = MIDCAP − MARKET; everything else = raw return).
        out_cols: dict[str, pd.Series] = {}
        market_rets: pd.Series | None = rets.get("Market")
        for name, meta in self.factor_proxies.items():
            if name in dropped:
                continue
            transform = meta.get("transform", "excess")
            if transform == "smb" and market_rets is not None and name in rets.columns:
                # Size = mid-cap return − market return ("small minus big").
                out_cols[name] = rets[name] - market_rets
            elif name in rets.columns:
                out_cols[name] = rets[name]

        factor_returns = pd.DataFrame(out_cols).dropna()
        self.factor_returns_ = factor_returns
        return factor_returns, dropped

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------
    def fit(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame | None = None,
        hac_lags: int | None = None,
    ) -> FactorAttributionResult:
        """Run the multivariate regression of strategy excess returns on
        factor excess returns.

        Parameters
        ----------
        strategy_returns : Series
            Strategy daily simple returns.
        factor_returns : DataFrame, optional
            Pre-computed factor returns. If None, must have called
            :meth:`fetch_factors` already.
        hac_lags : int, optional
            Newey-West lag. Default = ⌊4·(T/100)^(2/9)⌋.
        """
        if factor_returns is None:
            if self.factor_returns_ is None:
                raise RuntimeError("Call fetch_factors() first or pass factor_returns.")
            factor_returns = self.factor_returns_

        # Daily risk-free rate (compound).
        rf_daily = (1 + self.risk_free_rate) ** (1 / C.TRADING_DAYS) - 1

        # Strategy excess.
        s = pd.Series(strategy_returns).dropna() - rf_daily
        # Factor excess (already returns, just subtract rf — except Size, which
        # is already a long-short spread so doesn't need rf subtraction).
        f = factor_returns.copy()
        for col in f.columns:
            if self.factor_proxies.get(col, {}).get("transform") != "smb":
                f[col] = f[col] - rf_daily

        # Align.
        s, f = s.align(f, join="inner", axis=0)
        f = f.dropna()
        s = s.reindex(f.index).dropna()
        f = f.reindex(s.index)
        n = len(s)
        if n < 50:
            raise ValueError(f"Need ≥50 aligned obs; got {n}.")

        # Design matrix [1, factors].
        X = np.column_stack([np.ones(n), f.values])
        y = s.values
        coef, var_coef, hac_used = _ols_hac(y, X, hac_lags)
        ses = np.sqrt(np.diag(var_coef))

        # Decompose coefficients.
        alpha_d = float(coef[0])
        alpha_se = float(ses[0])
        alpha_t = alpha_d / alpha_se if alpha_se > 0 else 0.0
        alpha_p = float(2.0 * (1.0 - stats.norm.cdf(abs(alpha_t))))

        factor_names = list(f.columns)
        betas = {n: float(coef[i + 1]) for i, n in enumerate(factor_names)}
        se_b = {n: float(ses[i + 1]) for i, n in enumerate(factor_names)}
        ts = {n: float(coef[i + 1] / ses[i + 1]) if ses[i + 1] > 0 else 0.0
              for i, n in enumerate(factor_names)}
        ps = {n: float(2.0 * (1.0 - stats.norm.cdf(abs(ts[n]))))
              for n in factor_names}

        # R².
        resid = y - X @ coef
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = X.shape[1] - 1                # number of regressors (excluding intercept)
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)

        # Attribution: each factor's contribution to mean strategy return.
        ann = C.TRADING_DAYS
        attribution_ann = {
            name: float(betas[name] * f[name].mean() * ann)
            for name in factor_names
        }
        strategy_mean_ann = float(s.mean() * ann)
        # Unexplained = mean residual (≈ alpha · 252 by construction with OLS).
        unexplained_ann = float(resid.mean() * ann)

        return FactorAttributionResult(
            alpha_daily=alpha_d,
            alpha_annual=alpha_d * ann,
            alpha_se=alpha_se,
            alpha_t=alpha_t,
            alpha_p=alpha_p,
            alpha_significant_5pct=alpha_p < 0.05,
            alpha_significant_1pct=alpha_p < 0.01,
            factor_names=factor_names,
            betas=betas,
            se_betas=se_b,
            t_stats=ts,
            p_values=ps,
            r_squared=r2,
            adj_r_squared=adj_r2,
            attribution_annual=attribution_ann,
            strategy_mean_annual=strategy_mean_ann,
            unexplained_annual=unexplained_ann,
            n_obs=n,
            hac_lags=hac_used,
        )

    # ------------------------------------------------------------------
    # Pretty printers
    # ------------------------------------------------------------------
    @staticmethod
    def to_summary_table(result: FactorAttributionResult) -> pd.DataFrame:
        """Coefficient table: factor → β, t-stat, p-value, annual contribution."""
        rows = [{
            "Factor": "α (pure alpha)",
            "Loading": f"{result.alpha_annual*100:+.2f}% / yr",
            "t-stat": result.alpha_t,
            "p-value": result.alpha_p,
            "Annual contribution": result.alpha_annual,
            "Significance": ("***" if result.alpha_significant_1pct
                              else "**" if result.alpha_significant_5pct
                              else "n.s."),
        }]
        for n in result.factor_names:
            t = result.t_stats[n]
            p = result.p_values[n]
            rows.append({
                "Factor": n,
                "Loading": f"{result.betas[n]:+.3f}",
                "t-stat": t,
                "p-value": p,
                "Annual contribution": result.attribution_annual[n],
                "Significance": ("***" if p < 0.01 else "**" if p < 0.05 else "n.s."),
            })
        df = pd.DataFrame(rows)
        return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic test: build a strategy that's known to have 0.8 market beta,
    # 0.3 size, no real alpha. See if the regression recovers those values.
    rng = np.random.default_rng(0)
    n = 1_500
    idx = pd.bdate_range("2019-01-01", periods=n)

    mkt = rng.normal(0.0005, 0.012, n)
    smb = rng.normal(0.0001, 0.008, n)
    it = rng.normal(0.0006, 0.014, n)
    bnk = rng.normal(0.0004, 0.013, n)

    # Strategy = 0.8 MKT + 0.3 SMB + 0.05 alpha + noise (no IT/Banking exposure)
    strat = 0.8 * mkt + 0.3 * smb + 0.0001 + rng.normal(0, 0.005, n)
    strat_series = pd.Series(strat, index=idx)
    factor_ret = pd.DataFrame({
        "Market": mkt, "Size": smb, "IT": it, "Banking": bnk,
    }, index=idx)

    fa = FactorAttribution(factor_proxies=INDIAN_FACTOR_PROXIES)
    fa.factor_returns_ = factor_ret
    res = fa.fit(strat_series)
    table = FactorAttribution.to_summary_table(res)
    print(table.to_string(index=False))
    print(f"\nR² = {res.r_squared:.3f}   adj-R² = {res.adj_r_squared:.3f}")
    print(f"Strategy mean (ann) = {res.strategy_mean_annual*100:.2f}%   "
          f"unexplained = {res.unexplained_annual*100:.2f}%")
