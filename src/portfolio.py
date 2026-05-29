"""
src/portfolio.py
================

Phase 6A — Portfolio risk analyser.

Accepts a user's holdings as a simple ``{ticker: rupee_amount}`` dict and
produces a full risk dashboard — the kind of report a fund's middle-office
sends out every morning.

Sections (in print order)
-------------------------
1.  **Portfolio metrics**   — annualised return, annualised vol via
    Markowitz (σ_p = √(wᵀ Σ w)), weighted-avg individual vol (so the
    user can *see* the diversification benefit), Sharpe, Sortino, max DD.
2.  **Per-stock metrics**   — weight %, ₹ value, annualised return, annualised
    vol, Sharpe, max drawdown, sector.
3.  **Risk contributions**  — Marginal Risk Contribution per stock:
        MRC_i = (w_i × (Σw)_i) / σ_p
    A position whose risk contribution is more than 1.5× its weight is
    flagged as a "risk hog".
4.  **Diversification score** — 0–100 composite of:
        * Mean pairwise correlation (50% weight; lower = better)
        * Sector diversity (Shannon entropy) (25% weight)
        * Inverse Herfindahl-Hirschman of weights (25% weight)
5.  **VaR / CVaR**          — Historical method, 1-day 95% & 99% VaR,
    95% CVaR (Expected Shortfall), reported as both % and ₹.
6.  **Sector exposure**     — aggregated by ``config.NSE_SECTORS``.
7.  **Correlation matrix**  — full N×N matrix for heatmap plotting.
8.  **Warnings**            — auto-generated alerts with HIGH/MEDIUM
    severity (CONCENTRATION, HIGH_CORR, RISK_HOG, SECTOR_CONC).
9.  **Print report**        — Bloomberg-style console output.

All math uses **log returns** for statistics (additive, ~normal) and
**simple returns** for P&L. The covariance matrix is computed from log
returns, annualised by × ``TRADING_DAYS``.
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
logger = logging.getLogger("portfolio")


class PortfolioAnalyzer:
    """Compute every risk number a PM would want for a small equity book.

    Parameters
    ----------
    holdings : dict[str, float]
        ``{ticker: rupee_amount_invested}``. Weights are derived from
        the ratios, so absolute amounts can be in any currency.
    risk_free_rate : float, optional
        Defaults to ``config.RISK_FREE_RATE`` (India 10-yr G-sec).
    """

    def __init__(
        self,
        holdings: dict[str, float],
        risk_free_rate: float = C.RISK_FREE_RATE,
    ) -> None:
        if not holdings:
            raise ValueError("Empty holdings dict.")
        if any(v <= 0 for v in holdings.values()):
            raise ValueError("All holding amounts must be positive.")
        self.holdings = dict(holdings)
        self.total_value = float(sum(holdings.values()))
        self.weights: dict[str, float] = {
            t: v / self.total_value for t, v in self.holdings.items()
        }
        self.tickers = list(self.holdings.keys())
        self.risk_free_rate = float(risk_free_rate)

        # Populated by analyze()
        self.returns_panel: pd.DataFrame | None = None
        self.cov: pd.DataFrame | None = None
        self.corr: pd.DataFrame | None = None
        self.portfolio_metrics: dict = {}
        self.stock_metrics: pd.DataFrame | None = None
        self.risk_contrib: pd.DataFrame | None = None
        self.diversification: dict = {}
        self.var: dict = {}
        self.sector_exposure: pd.DataFrame | None = None
        self.warnings: list[dict] = []

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------
    def analyze(self, signaled: dict[str, pd.DataFrame]) -> dict:
        """Run every section. Returns a dict; also caches on the instance."""
        missing = [t for t in self.tickers if t not in signaled]
        if missing:
            raise KeyError(
                f"Tickers in holdings but not in signal data: {missing}"
            )

        # ----- Build the aligned log-return panel -----
        rets = pd.DataFrame({
            t: signaled[t]["Adj_Return"] for t in self.tickers
        }).dropna(how="any")
        if len(rets) < 30:
            raise ValueError(
                f"Only {len(rets)} aligned trading days — need at least 30."
            )
        self.returns_panel = rets

        # Annualised covariance and correlation (log returns, ×252).
        self.cov = rets.cov() * C.TRADING_DAYS
        self.corr = rets.corr()

        # Run each section.
        self._compute_portfolio_metrics()
        self._compute_stock_metrics()
        self._compute_risk_contributions()
        self._compute_diversification()
        self._compute_var()
        self._compute_sector_exposure()
        self._compute_warnings()

        return {
            "portfolio_metrics": self.portfolio_metrics,
            "stock_metrics": self.stock_metrics,
            "risk_contributions": self.risk_contrib,
            "diversification": self.diversification,
            "var": self.var,
            "sector_exposure": self.sector_exposure,
            "correlation": self.corr,
            "covariance": self.cov,
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------------
    # 1. Portfolio metrics
    # ------------------------------------------------------------------
    def _compute_portfolio_metrics(self) -> None:
        w = np.array([self.weights[t] for t in self.tickers])
        rets = self.returns_panel

        # Portfolio daily log return = w · r (approx for small returns).
        port_log = (rets * w).sum(axis=1)
        port_simple = np.expm1(port_log)            # convert to simple returns

        ann_return = float(port_log.mean() * C.TRADING_DAYS)
        # Markowitz: σ_p = √(wᵀ Σ w). Σ is already annualised.
        ann_vol = float(np.sqrt(w @ self.cov.values @ w))

        # Weighted-average individual vol (no diversification benefit).
        ind_vols = rets.std() * np.sqrt(C.TRADING_DAYS)
        weighted_ind_vol = float((w * ind_vols.values).sum())

        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / C.TRADING_DAYS) - 1.0
        excess = port_simple - rf_daily
        sharpe = float(excess.mean() / port_simple.std() * np.sqrt(C.TRADING_DAYS)) \
            if port_simple.std() > 0 else 0.0
        downside = port_simple[port_simple < 0]
        sortino = float(excess.mean() / downside.std() * np.sqrt(C.TRADING_DAYS)) \
            if len(downside) > 1 and downside.std() > 0 else 0.0

        # Historical equity curve & max drawdown (simple-return compounding).
        equity = (1.0 + port_simple).cumprod()
        dd = equity / equity.cummax() - 1.0
        max_dd = float(dd.min())

        self.portfolio_metrics = {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "weighted_avg_individual_vol": weighted_ind_vol,
            "diversification_benefit": float(weighted_ind_vol - ann_vol),
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "total_value": self.total_value,
            "n_holdings": len(self.tickers),
        }

    # ------------------------------------------------------------------
    # 2. Per-stock metrics
    # ------------------------------------------------------------------
    def _compute_stock_metrics(self) -> None:
        rets = self.returns_panel
        rf_daily = (1.0 + self.risk_free_rate) ** (1.0 / C.TRADING_DAYS) - 1.0

        rows = []
        for t in self.tickers:
            r = rets[t]
            r_simple = np.expm1(r)
            ann_ret = float(r.mean() * C.TRADING_DAYS)
            ann_vol = float(r.std() * np.sqrt(C.TRADING_DAYS))
            sharpe = float((r_simple.mean() - rf_daily) / r_simple.std() * np.sqrt(C.TRADING_DAYS)) \
                if r_simple.std() > 0 else 0.0
            equity = (1.0 + r_simple).cumprod()
            dd = float((equity / equity.cummax() - 1.0).min())
            rows.append({
                "ticker": t,
                "weight": self.weights[t],
                "value": self.holdings[t],
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": dd,
                "sector": C.NSE_SECTORS.get(t, "Other"),
            })

        self.stock_metrics = pd.DataFrame(rows).sort_values(
            "weight", ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Risk contributions
    # ------------------------------------------------------------------
    def _compute_risk_contributions(self) -> None:
        """MRC_i = w_i × (Σw)_i / σ_p.

        Risk contributions sum to σ_p by construction (Euler decomposition).
        ``risk_vs_weight = MRC%/weight`` tells you which positions are
        contributing *disproportionately* more risk than their cash weight.
        """
        w = np.array([self.weights[t] for t in self.tickers])
        sigma = self.cov.values
        sigma_p = float(np.sqrt(w @ sigma @ w))
        if sigma_p == 0:
            self.risk_contrib = pd.DataFrame()
            return

        marginal = sigma @ w                              # vector
        component = w * marginal / sigma_p                # contributions in σ-units
        contrib_pct = component / sigma_p                 # fractions, sum to 1

        rows = []
        for i, t in enumerate(self.tickers):
            rows.append({
                "ticker": t,
                "weight": self.weights[t],
                "risk_contribution": float(component[i]),
                "risk_contribution_pct": float(contrib_pct[i]),
                "risk_vs_weight": float(contrib_pct[i] / self.weights[t])
                if self.weights[t] > 0 else 0.0,
            })

        self.risk_contrib = pd.DataFrame(rows).sort_values(
            "risk_contribution_pct", ascending=False
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Diversification score (0–100)
    # ------------------------------------------------------------------
    def _compute_diversification(self) -> None:
        corr = self.corr.values.copy()
        np.fill_diagonal(corr, np.nan)
        mean_corr = float(np.nanmean(corr))
        # Lower correlation → better. Map [0, 1] → [100, 0] linearly.
        corr_score = max(0.0, min(100.0, (1.0 - mean_corr) * 100.0))

        # Sector diversity — Shannon entropy of weight-by-sector, normalised
        # by log(n_sectors_possible).
        if self.sector_exposure is None:
            sec_w = pd.Series({
                t: self.weights[t] for t in self.tickers
            }).groupby(lambda t: C.NSE_SECTORS.get(t, "Other")).sum()
        else:
            sec_w = self.sector_exposure["weight"]
        p = sec_w.values
        entropy = float(-(p * np.log(np.where(p > 0, p, 1))).sum())
        # Use number of sectors *in this portfolio*, not the universe, for
        # the normaliser — penalises N=1 fairly.
        max_entropy = np.log(max(len(p), 2))
        sector_score = float(min(100.0, entropy / max_entropy * 100.0)) \
            if max_entropy > 0 else 0.0

        # Inverse-HHI on weights. HHI = Σ w_i² ∈ [1/n, 1]. Convert to "effective N".
        hhi = float(sum(w * w for w in self.weights.values()))
        eff_n = 1.0 / hhi if hhi > 0 else 0.0
        # Map effective N to 0..100 — 1 stock → 0, ≥10 → 100.
        weight_score = float(min(100.0, max(0.0, (eff_n - 1.0) / 9.0 * 100.0)))

        composite = 0.50 * corr_score + 0.25 * sector_score + 0.25 * weight_score
        if composite >= 75:
            label = "Well Diversified"
        elif composite >= 50:
            label = "Moderately Diversified"
        elif composite >= 25:
            label = "Poorly Diversified"
        else:
            label = "Highly Concentrated"

        self.diversification = {
            "score": float(composite),
            "label": label,
            "mean_pairwise_correlation": mean_corr,
            "sector_entropy": entropy,
            "effective_n": eff_n,
            "hhi": hhi,
            "components": {
                "correlation": corr_score,
                "sector": sector_score,
                "weight": weight_score,
            },
        }

    # ------------------------------------------------------------------
    # 5. VaR / CVaR — historical method
    # ------------------------------------------------------------------
    def _compute_var(self) -> None:
        w = np.array([self.weights[t] for t in self.tickers])
        # Daily portfolio simple returns from log-return panel.
        port_simple = np.expm1((self.returns_panel * w).sum(axis=1))

        var_95 = float(np.percentile(port_simple, 5))            # 5th percentile loss
        var_99 = float(np.percentile(port_simple, 1))
        cvar_95 = float(port_simple[port_simple <= var_95].mean())

        self.var = {
            "var_95_pct": var_95,
            "var_99_pct": var_99,
            "cvar_95_pct": cvar_95,
            "var_95_inr": var_95 * self.total_value,
            "var_99_inr": var_99 * self.total_value,
            "cvar_95_inr": cvar_95 * self.total_value,
            "horizon": "1-day",
            "method": "historical",
        }

    # ------------------------------------------------------------------
    # 6. Sector exposure
    # ------------------------------------------------------------------
    def _compute_sector_exposure(self) -> None:
        rows = []
        for t in self.tickers:
            rows.append({
                "ticker": t,
                "sector": C.NSE_SECTORS.get(t, "Other"),
                "weight": self.weights[t],
                "value": self.holdings[t],
            })
        df = pd.DataFrame(rows)
        agg = df.groupby("sector").agg(weight=("weight", "sum"),
                                        value=("value", "sum"),
                                        n=("ticker", "count")).reset_index()
        self.sector_exposure = agg.sort_values("weight", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. Warnings
    # ------------------------------------------------------------------
    def _compute_warnings(self) -> None:
        warns: list[dict] = []

        # Single-stock concentration > 30% — almost always a HIGH alert.
        for t, w in self.weights.items():
            if w > 0.30:
                warns.append({
                    "type": "CONCENTRATION",
                    "severity": "HIGH",
                    "message": f"{t} is {w:.1%} of the book (>30%).",
                    "ticker": t,
                    "value": w,
                })

        # Highly correlated pairs (correlation > 0.75).
        corr = self.corr.copy()
        seen = set()
        for i, t1 in enumerate(corr.index):
            for j, t2 in enumerate(corr.columns):
                if j <= i:
                    continue
                c = corr.iat[i, j]
                if pd.isna(c) or c <= 0.75:
                    continue
                pair = tuple(sorted((t1, t2)))
                if pair in seen:
                    continue
                seen.add(pair)
                warns.append({
                    "type": "HIGH_CORR",
                    "severity": "HIGH" if c > 0.90 else "MEDIUM",
                    "message": f"{t1}↔{t2} correlation {c:.2f} (>0.75).",
                    "value": float(c),
                })

        # Risk hogs.
        if self.risk_contrib is not None and not self.risk_contrib.empty:
            for _, row in self.risk_contrib.iterrows():
                rvw = float(row["risk_vs_weight"])
                if rvw > 1.5:
                    warns.append({
                        "type": "RISK_HOG",
                        "severity": "HIGH" if rvw > 2.0 else "MEDIUM",
                        "message": (
                            f"{row['ticker']} contributes {row['risk_contribution_pct']:.1%} of risk "
                            f"on a {row['weight']:.1%} weight (×{rvw:.2f})."
                        ),
                        "ticker": row["ticker"],
                        "value": rvw,
                    })

        # Sector concentration > 50%.
        if self.sector_exposure is not None:
            for _, row in self.sector_exposure.iterrows():
                if row["weight"] > 0.50:
                    warns.append({
                        "type": "SECTOR_CONC",
                        "severity": "HIGH",
                        "message": f"{row['sector']} sector is {row['weight']:.1%} (>50%).",
                        "value": float(row["weight"]),
                    })

        self.warnings = warns

    # ------------------------------------------------------------------
    # 9. Reporting
    # ------------------------------------------------------------------
    def print_report(self) -> None:
        if not self.portfolio_metrics:
            print("Run analyze() first.")
            return
        m = self.portfolio_metrics
        d = self.diversification
        v = self.var

        line = "=" * 70
        print(line)
        print(f"  PORTFOLIO RISK DASHBOARD   (total value: ₹{self.total_value:,.0f})")
        print(line)

        # Section 1
        print("\n[1] Portfolio Metrics")
        print(f"  Annualised Return            :  {m['ann_return']*100:>8.2f}%")
        print(f"  Annualised Volatility        :  {m['ann_vol']*100:>8.2f}%")
        print(f"  Weighted-Avg Individual Vol  :  {m['weighted_avg_individual_vol']*100:>8.2f}%")
        print(f"  Diversification Benefit      :  {m['diversification_benefit']*100:>8.2f}%")
        print(f"  Sharpe Ratio                 :  {m['sharpe']:>8.2f}")
        print(f"  Sortino Ratio                :  {m['sortino']:>8.2f}")
        print(f"  Max Drawdown                 :  {m['max_drawdown']*100:>8.2f}%")

        # Section 2
        print("\n[2] Stock Metrics")
        cols = ["ticker", "weight", "ann_return", "ann_vol", "sharpe", "max_drawdown", "sector"]
        sm = self.stock_metrics[cols].copy()
        sm["weight"] = sm["weight"].apply(lambda x: f"{x:.1%}")
        sm["ann_return"] = sm["ann_return"].apply(lambda x: f"{x*100:.1f}%")
        sm["ann_vol"] = sm["ann_vol"].apply(lambda x: f"{x*100:.1f}%")
        sm["sharpe"] = sm["sharpe"].apply(lambda x: f"{x:.2f}")
        sm["max_drawdown"] = sm["max_drawdown"].apply(lambda x: f"{x*100:.1f}%")
        print(sm.to_string(index=False))

        # Section 3
        print("\n[3] Risk Contributions")
        rc = self.risk_contrib.copy()
        rc["weight"] = rc["weight"].apply(lambda x: f"{x:.1%}")
        rc["risk_contribution_pct"] = rc["risk_contribution_pct"].apply(lambda x: f"{x:.1%}")
        rc["risk_vs_weight"] = rc["risk_vs_weight"].apply(lambda x: f"{x:.2f}×")
        print(rc[["ticker", "weight", "risk_contribution_pct", "risk_vs_weight"]].to_string(index=False))

        # Section 4
        print("\n[4] Diversification")
        print(f"  Score                        :  {d['score']:>5.1f} / 100 — {d['label']}")
        print(f"  Mean Pairwise Correlation    :  {d['mean_pairwise_correlation']:>5.2f}")
        print(f"  Effective N (1/HHI)          :  {d['effective_n']:>5.2f}")

        # Section 5
        print("\n[5] Value-at-Risk (1-day, historical)")
        print(f"  VaR 95%                      :  {v['var_95_pct']*100:>6.2f}%   "
              f"(≈ ₹{abs(v['var_95_inr']):,.0f} loss)")
        print(f"  VaR 99%                      :  {v['var_99_pct']*100:>6.2f}%   "
              f"(≈ ₹{abs(v['var_99_inr']):,.0f} loss)")
        print(f"  CVaR 95% (Expected Shortfall):  {v['cvar_95_pct']*100:>6.2f}%   "
              f"(≈ ₹{abs(v['cvar_95_inr']):,.0f} loss)")

        # Section 6
        print("\n[6] Sector Exposure")
        se = self.sector_exposure.copy()
        se["weight"] = se["weight"].apply(lambda x: f"{x:.1%}")
        se["value"] = se["value"].apply(lambda x: f"₹{x:,.0f}")
        print(se.to_string(index=False))

        # Section 8 (we skip the full corr/cov matrices in printed report —
        # those are returned as DataFrames for downstream plotting).
        print("\n[7] Warnings")
        if not self.warnings:
            print("  ✓ No warnings — portfolio looks structurally healthy.")
        else:
            for w in self.warnings:
                print(f"  [{w['severity']:<6}] {w['type']:<14}  {w['message']}")
        print(line)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from features import FeatureEngineer
    from regime import RegimeDetector
    from signals import SignalEngine, add_entry_exit_levels

    rng = np.random.default_rng(3)
    n = 600
    idx = pd.bdate_range("2022-01-03", periods=n)
    tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "RELIANCE.NS"]
    signaled = {}
    for t in tickers:
        rets = rng.normal(0.0005, 0.014, n)
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

    pa = PortfolioAnalyzer({
        "TCS.NS": 35_000, "INFY.NS": 25_000,
        "HDFCBANK.NS": 25_000, "RELIANCE.NS": 15_000,
    })
    pa.analyze(signaled)
    pa.print_report()
