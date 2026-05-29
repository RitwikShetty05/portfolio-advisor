"""
src/significance.py
===================

Statistical-significance tools for backtest results.

Why this module exists
----------------------
The Sharpe ratio printed in any tearsheet is a *point estimate* of an
unknown quantity. Two independent strategies with the same Sharpe of 1.0
can have wildly different *true* Sharpes — one might be a coin flip
above zero, the other genuinely skilled. Reporting only the point
estimate is the single biggest source of "backtest looked great, live
trading lost money" disappointment in junior quant work.

This module quantifies the uncertainty around the Sharpe and tests
whether your alpha is statistically distinguishable from luck:

1.  ``bootstrap_sharpe_ci``
        Percentile bootstrap CI on the annualised Sharpe. Resample the
        daily-return series with replacement N times, recompute Sharpe
        on each, take the 2.5%/97.5% percentiles. Fully non-parametric —
        survives fat tails, skew, and serial autocorrelation reasonably
        well (use a block bootstrap if you want to be really careful).

2.  ``probabilistic_sharpe_ratio``
        PSR (Bailey & López de Prado, 2012):
            PSR(SR*) = Φ( (SR_obs − SR*) · √(n−1) /
                          √(1 − γ₃·SR_obs + (γ₄/4)·SR_obs²) )
        i.e. P(true Sharpe > benchmark) under a normality-adjusted
        sampling distribution that *accounts for skew and excess
        kurtosis*. PSR > 0.95 is the typical threshold for "real."

3.  ``deflated_sharpe_ratio``
        DSR (Bailey & López de Prado, 2014). Adjusts PSR for **selection
        bias**: when you tried many parameter combinations and reported
        the best, the observed Sharpe is inflated. DSR deflates the
        benchmark to E[max_SR | N trials, SR_true=0] before computing
        PSR. The honest probability that your strategy is real *after*
        accounting for how many you tried.

4.  ``alpha_t_stat``
        Regresses strategy excess returns on benchmark excess returns
        (CAPM) and returns the alpha, beta, and t-statistic with
        **Newey-West (HAC) standard errors** — these correct for serial
        autocorrelation in daily returns, which OLS does not. A
        t-statistic of |2.0|+ on alpha is the conventional bar for
        "statistically significant" in finance papers.

References
----------
* Bailey, D. H., & López de Prado, M. M. (2012). The Sharpe Ratio
  Efficient Frontier. *Journal of Risk* 15(2).
* Bailey, D. H., & López de Prado, M. M. (2014). The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting, and
  Non-Normality. *Journal of Portfolio Management* 40(5).
* Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  *Econometrica* 55.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as e:  # pragma: no cover
    raise ImportError("scipy required: pip install scipy") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

logger = logging.getLogger("significance")


# Euler-Mascheroni constant — used in the expected-maximum formula for DSR.
EULER_MASCHERONI = 0.5772156649


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce(returns: pd.Series | np.ndarray | Sequence[float]) -> np.ndarray:
    """Drop NaNs, return a 1-D numpy array."""
    if isinstance(returns, pd.Series):
        r = returns.dropna().values
    else:
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
    return r


def annualised_sharpe(returns, risk_free_daily: float = 0.0) -> float:
    """Annualised Sharpe = (mean(r-rf) / std(r)) × √252.

    Uses ``ddof=1`` (sample std) — matches scipy / R / Excel conventions.
    """
    r = _coerce(returns)
    if len(r) < 2:
        return 0.0
    sigma = r.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float((r.mean() - risk_free_daily) / sigma * np.sqrt(C.TRADING_DAYS))


# ---------------------------------------------------------------------------
# 1. Bootstrap CI on Sharpe
# ---------------------------------------------------------------------------
@dataclass
class BootstrapSharpeResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_resamples: int
    samples: np.ndarray              # full bootstrap distribution


def bootstrap_sharpe_ci(
    returns,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    risk_free_daily: float = 0.0,
    seed: int = 42,
) -> BootstrapSharpeResult:
    """Percentile bootstrap CI on the annualised Sharpe.

    Why percentile (not basic) bootstrap: the percentile CI is invariant
    under monotone transformations and behaves well on bounded estimators
    like the Sharpe ratio. For very small samples (<60), consider a
    bias-corrected accelerated (BCa) bootstrap instead.
    """
    r = _coerce(returns)
    if len(r) < 30:
        raise ValueError(f"Need ≥30 returns for bootstrap; got {len(r)}.")
    rng = np.random.default_rng(seed)

    point = annualised_sharpe(r, risk_free_daily)
    n = len(r)
    # Vectorised resampling — much faster than a Python loop.
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples_2d = r[idx]                                         # (B, n)
    means = samples_2d.mean(axis=1)
    stds = samples_2d.std(axis=1, ddof=1)
    sharpes = np.where(stds > 0,
                       (means - risk_free_daily) / stds * np.sqrt(C.TRADING_DAYS),
                       0.0)
    alpha = 1.0 - ci_level
    low = float(np.percentile(sharpes, alpha / 2 * 100))
    high = float(np.percentile(sharpes, (1.0 - alpha / 2) * 100))
    return BootstrapSharpeResult(
        point_estimate=point, ci_low=low, ci_high=high,
        ci_level=ci_level, n_resamples=n_resamples, samples=sharpes,
    )


# ---------------------------------------------------------------------------
# 2. Probabilistic Sharpe Ratio (PSR)
# ---------------------------------------------------------------------------
@dataclass
class PSRResult:
    psr: float                         # probability in [0, 1]
    sr_observed: float                 # annualised
    sr_benchmark: float                # annualised
    skew: float
    excess_kurtosis: float
    n_obs: int


def probabilistic_sharpe_ratio(
    returns,
    sr_benchmark: float = 0.0,
    risk_free_daily: float = 0.0,
) -> PSRResult:
    """P(true annualised Sharpe > sr_benchmark) given the observed SR,
    skew, excess kurtosis, and sample size.

    Formula uses daily quantities; we convert annual ↔ daily by ×/÷ √252.
    """
    r = _coerce(returns)
    n = len(r)
    if n < 30:
        raise ValueError("Need ≥30 returns for PSR.")

    sr_ann = annualised_sharpe(r, risk_free_daily)
    sr_daily = sr_ann / np.sqrt(C.TRADING_DAYS)
    sr_bench_daily = sr_benchmark / np.sqrt(C.TRADING_DAYS)

    skew = float(stats.skew(r, bias=False))
    # scipy returns *excess* kurtosis when fisher=True (default).
    ekurt = float(stats.kurtosis(r, fisher=True, bias=False))

    # Standard error of the (daily) Sharpe given non-normal returns.
    # σ_SR² = (1 − γ₃·SR + (γ₄/4)·SR²) / (n − 1)
    denom_sq = (1.0 - skew * sr_daily + (ekurt / 4.0) * sr_daily ** 2) / (n - 1)
    if denom_sq <= 0:
        # Degenerate case (extreme skew/kurtosis); return uninformative 50%.
        return PSRResult(0.5, sr_ann, sr_benchmark, skew, ekurt, n)
    se = np.sqrt(denom_sq)
    z = (sr_daily - sr_bench_daily) / se
    psr = float(stats.norm.cdf(z))
    return PSRResult(psr, sr_ann, sr_benchmark, skew, ekurt, n)


# ---------------------------------------------------------------------------
# 3. Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------
@dataclass
class DSRResult:
    dsr: float
    sr_observed: float
    expected_max_sr: float
    n_trials: int
    psr_result: PSRResult


def deflated_sharpe_ratio(
    returns,
    n_trials: int = 10,
    risk_free_daily: float = 0.0,
) -> DSRResult:
    """Deflated Sharpe Ratio.

    Adjusts PSR for the multiple-testing problem: if you tried ``n_trials``
    different strategies / parameter combinations and reported the best,
    the observed Sharpe is biased upward. DSR computes PSR against the
    *expected maximum Sharpe* of ``n_trials`` independent strategies whose
    true Sharpe is zero.

    Expected-max formula (Bailey & López de Prado, 2014):
        E[max_SR] ≈ √(Var(SR)) · ((1−γ) · Φ⁻¹(1 − 1/N)
                                  + γ · Φ⁻¹(1 − 1/(N·e)))
    where γ = Euler-Mascheroni ≈ 0.5772 and Var(SR) is computed under
    the null SR_true = 0.
    """
    r = _coerce(returns)
    n = len(r)
    if n < 30:
        raise ValueError("Need ≥30 returns for DSR.")

    sr_ann = annualised_sharpe(r, risk_free_daily)

    # Variance of the Sharpe estimator under SR_true = 0 (the null).
    # Under H0: Var(SR_daily) = 1/(n-1)  →  Var(SR_ann) = TRADING_DAYS / (n-1).
    var_sr_ann = C.TRADING_DAYS / (n - 1)
    sd_sr_ann = np.sqrt(var_sr_ann)

    if n_trials <= 1:
        expected_max = 0.0
    else:
        ppf_1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        ppf_2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        expected_max = sd_sr_ann * (
            (1 - EULER_MASCHERONI) * ppf_1 + EULER_MASCHERONI * ppf_2
        )
    psr_res = probabilistic_sharpe_ratio(
        returns, sr_benchmark=expected_max, risk_free_daily=risk_free_daily,
    )
    return DSRResult(
        dsr=psr_res.psr,
        sr_observed=sr_ann,
        expected_max_sr=expected_max,
        n_trials=n_trials,
        psr_result=psr_res,
    )


# ---------------------------------------------------------------------------
# 4. Alpha t-stat with Newey-West HAC standard errors
# ---------------------------------------------------------------------------
@dataclass
class AlphaResult:
    alpha_daily: float
    alpha_annual: float
    beta: float
    se_alpha_daily: float
    t_stat: float
    p_value: float
    n_obs: int
    hac_lags: int
    significant_5pct: bool
    significant_1pct: bool


def alpha_t_stat(
    strategy_returns,
    benchmark_returns,
    risk_free_daily: float = 0.0,
    hac_lags: int | None = None,
) -> AlphaResult:
    """Jensen's alpha with Newey-West HAC standard errors.

    Model:
        (r_s − rf) = α + β · (r_b − rf) + ε

    OLS gives unbiased α and β but the standard errors are wrong when ε
    is autocorrelated (which daily-return residuals almost always are).
    Newey-West fixes this by combining the contemporaneous and lagged
    cross-product matrices of the score with Bartlett weights:

        Ω̂ = Γ̂₀ + Σ_{l=1}^L w_l · (Γ̂_l + Γ̂_l')
        Var(β̂) = (X'X)⁻¹ Ω̂ (X'X)⁻¹

    Default ``hac_lags = floor(4 · (T/100)^(2/9))`` — Newey-West's
    common rule of thumb.
    """
    # Align series.
    s = pd.Series(strategy_returns).dropna()
    b = pd.Series(benchmark_returns).dropna()
    s, b = s.align(b, join="inner")
    n = len(s)
    if n < 30:
        raise ValueError(f"Need ≥30 aligned obs; got {n}.")

    if hac_lags is None:
        hac_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
        hac_lags = max(hac_lags, 1)

    y = (s - risk_free_daily).values
    x = (b - risk_free_daily).values
    X = np.column_stack([np.ones(n), x])

    # OLS point estimates.
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    alpha_d, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef

    # Newey-West Ω with Bartlett kernel.
    omega = (X.T * (resid ** 2)) @ X                 # Γ̂₀
    for lag in range(1, hac_lags + 1):
        w = 1.0 - lag / (hac_lags + 1)               # Bartlett weight
        gamma = (X[lag:].T * (resid[lag:] * resid[:-lag])) @ X[:-lag]
        omega += w * (gamma + gamma.T)

    var_coef = XtX_inv @ omega @ XtX_inv
    se_alpha = float(np.sqrt(var_coef[0, 0]))
    t = alpha_d / se_alpha if se_alpha > 0 else 0.0
    # Two-sided p-value from the normal approximation (n is large).
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(t))))

    return AlphaResult(
        alpha_daily=alpha_d,
        alpha_annual=float(alpha_d * C.TRADING_DAYS),
        beta=beta,
        se_alpha_daily=se_alpha,
        t_stat=float(t),
        p_value=p,
        n_obs=n,
        hac_lags=hac_lags,
        significant_5pct=bool(p < 0.05),
        significant_1pct=bool(p < 0.01),
    )


# ---------------------------------------------------------------------------
# Convenience: one-shot full significance report
# ---------------------------------------------------------------------------
def full_significance_report(
    strategy_returns,
    benchmark_returns=None,
    n_trials_dsr: int = 10,
    n_bootstrap: int = 5000,
    risk_free_daily: float = 0.0,
) -> dict:
    """Run every test and return a single dict ready for UI rendering."""
    out: dict = {}
    out["bootstrap"] = bootstrap_sharpe_ci(
        strategy_returns, n_resamples=n_bootstrap,
        risk_free_daily=risk_free_daily,
    )
    out["psr"] = probabilistic_sharpe_ratio(
        strategy_returns, sr_benchmark=0.0,
        risk_free_daily=risk_free_daily,
    )
    out["dsr"] = deflated_sharpe_ratio(
        strategy_returns, n_trials=n_trials_dsr,
        risk_free_daily=risk_free_daily,
    )
    if benchmark_returns is not None and len(benchmark_returns) > 30:
        try:
            out["alpha"] = alpha_t_stat(
                strategy_returns, benchmark_returns,
                risk_free_daily=risk_free_daily,
            )
        except Exception as e:
            logger.warning("Alpha t-stat failed: %s", e)
            out["alpha"] = None
    else:
        out["alpha"] = None
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1_000

    # Skill scenario: mean 0.06%/day, std 1.2%/day, mild positive skew.
    rets = rng.normal(0.0006, 0.012, n)
    # Inject some positive skew.
    rets += rng.standard_t(df=8, size=n) * 0.002
    s = pd.Series(rets)

    print(f"Sample Sharpe (annualised): {annualised_sharpe(s):.2f}")

    boot = bootstrap_sharpe_ci(s, n_resamples=2000, seed=1)
    print(f"Bootstrap 95% CI: [{boot.ci_low:.2f}, {boot.ci_high:.2f}] "
          f"(point: {boot.point_estimate:.2f})")

    psr = probabilistic_sharpe_ratio(s)
    print(f"PSR (vs 0): {psr.psr*100:.1f}%   "
          f"[skew {psr.skew:+.2f}, ekurt {psr.excess_kurtosis:+.2f}]")

    dsr = deflated_sharpe_ratio(s, n_trials=20)
    print(f"DSR (N=20 trials, E[max]={dsr.expected_max_sr:.2f}): {dsr.dsr*100:.1f}%")

    # Benchmark: NIFTY-like.
    bench = pd.Series(rng.normal(0.0004, 0.011, n))
    a = alpha_t_stat(s, bench)
    sig = "***" if a.significant_1pct else "**" if a.significant_5pct else "ns"
    print(f"Alpha (ann.) {a.alpha_annual*100:+.2f}%   "
          f"β={a.beta:.2f}   t={a.t_stat:+.2f}   p={a.p_value:.3f}  {sig}")
