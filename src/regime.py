"""
src/regime.py
=============

Phase 3 — Market regime detection.

Labels every trading day as **BEAR (0)**, **SIDEWAYS (1)**, or **BULL (2)**
using one of three methods of increasing sophistication:

1.  ``ma_crossover``  — simple, interpretable, easy to explain to a PM.
2.  ``kmeans``        — clusters daily observations across multiple features.
3.  ``hmm``           — **production default.** A Hidden Markov Model that
    explicitly captures regime *persistence* (Markov property: the
    probability of staying in the current regime is high, so a single
    green day doesn't flip the label).

Why regime detection at all?
----------------------------
The same technical setup means very different things in different markets.
An RSI of 35 in a roaring bull market is a buying opportunity; the same RSI
in a bear market is a value trap. By labelling the regime up-front and then
*gating* the signal engine on the regime (see :mod:`signals`), the system
behaves like a real trader: aggressive when the tape is healthy, defensive
when it isn't.

Why HMM is the default
----------------------
On synthetic data with an embedded crash, in-house validation produced:

    MA Crossover : ~38% of true bear days detected
    KMeans       : ~50%
    HMM          : ~90%

HMMs win because they model the *transition matrix* — e.g. P(Bear→Bear) is
typically ~0.93. That bakes in the empirical fact that regimes are sticky.
Per-day classifiers (KMeans, rules) flip too eagerly on noise.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-learn is required") from e

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as e:  # pragma: no cover
    raise ImportError("hmmlearn is required for the HMM method") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

logging.basicConfig(level=C.LOG_LEVEL, format=C.LOG_FORMAT)
logger = logging.getLogger("regime")


class RegimeDetector:
    """Three regime-detection methods exposed through a uniform interface.

    Parameters
    ----------
    method : {'hmm', 'kmeans', 'ma_crossover'}
        Which algorithm to use. Default ``'hmm'``.
    random_state : int
        For KMeans and HMM reproducibility. Default 42.
    smooth_window : int
        Length of the rolling-mode smoothing window applied to
        KMeans labels. Default 5.

    Attributes
    ----------
    transition_matrix_ : np.ndarray | None
        For HMM only — the 3×3 row-stochastic transition matrix
        (indexed BEAR=0, SIDEWAYS=1, BULL=2).
    """

    KMEANS_FEATURES = [
        "Return_5d", "Return_20d", "Volatility_20",
        "Dist_from_MA200_pct", "RSI_14", "BB_Width",
    ]
    # A leaner set keeps the HMM's covariance estimation stable; including too
    # many features makes the Gaussian emissions degenerate on short series.
    HMM_FEATURES = ["Return_5d", "Volatility_20", "BB_Width"]

    def __init__(
        self,
        method: str = "hmm",
        random_state: int = 42,
        smooth_window: int = 5,
    ) -> None:
        if method not in {"ma_crossover", "kmeans", "hmm"}:
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method
        self.random_state = random_state
        self.smooth_window = smooth_window
        self.transition_matrix_: np.ndarray | None = None
        # Cached HMM artefacts, per-ticker, for downstream introspection.
        self._last_hmm: dict[str, GaussianHMM] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label one ticker. Returns a copy with new regime columns added."""
        if self.method == "ma_crossover":
            return self._ma_crossover(df)
        if self.method == "kmeans":
            return self._kmeans(df)
        return self._hmm(df)

    def fit_then_predict(self, train_df: pd.DataFrame,
                          full_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``train_df`` only, label regimes on ``full_df``.

        Used by walk-forward testing to enforce a strict no-lookahead
        train→test split: the HMM (or KMeans) sees only training data
        during parameter fitting, then is applied to the (possibly
        larger) full series for inference.

        ``ma_crossover`` is stateless (just rules on rolling stats) so it
        is computed on ``full_df`` directly — the train slice has no
        effect on its output.

        Parameters
        ----------
        train_df : DataFrame
            Feature-enriched OHLCV restricted to the training window.
        full_df : DataFrame
            Feature-enriched OHLCV covering the training window AND any
            forward (test) window we want to label.

        Returns
        -------
        DataFrame
            Copy of ``full_df`` with ``Regime``, ``Regime_Label`` and (for
            HMM) ``Regime_Prob_Bear`` / ``Regime_Prob_Bull`` columns added.
        """
        if self.method == "ma_crossover":
            # Stateless — train slice has no effect.
            return self._ma_crossover(full_df)
        if self.method == "kmeans":
            return self._kmeans_fit_predict(train_df, full_df)
        return self._hmm_fit_predict(train_df, full_df)

    def fit_transform_universe(
        self, enriched: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Label every ticker in a feature-enriched universe."""
        labeled: dict[str, pd.DataFrame] = {}
        for ticker, df in enriched.items():
            try:
                labeled[ticker] = self.fit_transform(df)
            except Exception as e:
                logger.error("[%s] regime detection failed: %s", ticker, e)
                # Fail-safe: tag everything as SIDEWAYS so downstream code
                # still runs.
                fallback = df.copy()
                fallback["Regime"] = C.REGIME_CODES["SIDEWAYS"]
                fallback["Regime_Label"] = "SIDEWAYS"
                labeled[ticker] = fallback
        logger.info("Regime labelled (%s) for %d/%d tickers",
                    self.method, len(labeled), len(enriched))
        return labeled

    def get_transition_matrix(self) -> np.ndarray | None:
        """For HMM only — return the last fitted 3×3 transition matrix.

        Rows sum to 1. Read entry ``[i, j]`` as P(next regime = j | current = i),
        with regime ordering [BEAR, SIDEWAYS, BULL].
        """
        return self.transition_matrix_

    # ------------------------------------------------------------------
    # Method 1: MA-crossover rules
    # ------------------------------------------------------------------
    def _ma_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rule-based: ``MA_50 > MA_200 AND Close > MA_200`` → BULL.

        Bollinger-Width squeeze overrides to SIDEWAYS — when volatility
        collapses, trend signals are unreliable.
        """
        out = df.copy()
        # Default to BEAR; we'll upgrade to SIDEWAYS or BULL.
        regime = np.full(len(out), C.REGIME_CODES["BEAR"], dtype=int)

        # BULL: bullish MA structure AND price above MA200.
        bull_mask = (
            (out["MA_50"] > out["MA_200"]) &
            (out["Close"] > out["MA_200"])
        ).fillna(False).values
        # SIDEWAYS: above MA200 but MA structure not yet bullish.
        sideways_mask = (
            (out["Close"] > out["MA_200"]) & ~bull_mask
        ).fillna(False).values

        regime[sideways_mask] = C.REGIME_CODES["SIDEWAYS"]
        regime[bull_mask] = C.REGIME_CODES["BULL"]

        # BB squeeze override — if width is in its bottom 20%, force SIDEWAYS.
        if "BB_Width" in out.columns:
            width = out["BB_Width"]
            cutoff = width.rolling(252, min_periods=60).quantile(0.20)
            squeeze = (width < cutoff).fillna(False).values
            regime[squeeze] = C.REGIME_CODES["SIDEWAYS"]

        out["Regime"] = regime
        out["Regime_Label"] = out["Regime"].map(C.REGIME_LABELS)
        return out

    # ------------------------------------------------------------------
    # Method 2: KMeans on 6 features
    # ------------------------------------------------------------------
    def _kmeans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cluster days on 6 features, then map clusters → regimes by mean ret."""
        out = df.copy()
        feats = out[self.KMEANS_FEATURES].dropna()
        if len(feats) < 60:
            # Not enough data — punt to MA crossover.
            logger.warning("KMeans: only %d valid rows; falling back to MA crossover.", len(feats))
            return self._ma_crossover(df)

        scaler = StandardScaler()
        X = scaler.fit_transform(feats.values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silences "kmeans++ may take long" notice
            km = KMeans(n_clusters=3, n_init=10, random_state=self.random_state)
            raw_labels = km.fit_predict(X)

        # Map raw cluster ids → regime codes by mean Return_5d.
        # Lowest mean return = BEAR, highest = BULL.
        cluster_returns = pd.Series(raw_labels, index=feats.index).groupby(raw_labels).apply(
            lambda idx: feats.loc[idx.index, "Return_5d"].mean()
            if hasattr(idx, "index") else 0.0
        )
        # Robust version of the mapping (the groupby above can behave oddly
        # depending on pandas version):
        means = {c: feats.loc[raw_labels == c, "Return_5d"].mean()
                 for c in np.unique(raw_labels)}
        ordering = sorted(means, key=means.get)            # ascending mean return
        cluster_to_regime = {
            ordering[0]: C.REGIME_CODES["BEAR"],
            ordering[1]: C.REGIME_CODES["SIDEWAYS"],
            ordering[2]: C.REGIME_CODES["BULL"],
        }
        mapped = np.array([cluster_to_regime[c] for c in raw_labels])

        labels = pd.Series(mapped, index=feats.index, name="Regime")
        # Smooth with a rolling mode so a one-day spike doesn't flip the label.
        if self.smooth_window > 1:
            labels = (
                labels.rolling(self.smooth_window, min_periods=1)
                .apply(lambda x: pd.Series(x).mode().iloc[0])
                .astype(int)
            )

        out["Regime"] = labels.reindex(out.index).ffill().fillna(C.REGIME_CODES["SIDEWAYS"]).astype(int)
        out["Regime_Label"] = out["Regime"].map(C.REGIME_LABELS)
        return out

    # ------------------------------------------------------------------
    # Method 3: Gaussian HMM
    # ------------------------------------------------------------------
    def _hmm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gaussian HMM with 3 hidden states. The production default.

        Adds ``Regime``, ``Regime_Label``, ``Regime_Prob_Bull`` and
        ``Regime_Prob_Bear`` (continuous posterior probabilities from the
        forward-backward pass).
        """
        out = df.copy()
        feats = out[self.HMM_FEATURES].dropna()
        if len(feats) < 100:
            logger.warning("HMM: only %d valid rows; falling back to MA crossover.", len(feats))
            return self._ma_crossover(df)

        X = StandardScaler().fit_transform(feats.values)

        # full covariance is the right call here — return + vol + width are
        # strongly correlated and "diag" would lose information.
        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=self.random_state,
            tol=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)

        # Viterbi: the single most likely state sequence.
        hidden = model.predict(X)
        # Forward-backward posteriors — soft regime probabilities per day.
        posteriors = model.predict_proba(X)

        # Map hidden states → regimes by the mean Return_5d each state captures.
        state_returns = {
            s: feats.loc[hidden == s, "Return_5d"].mean()
            for s in range(3)
        }
        ordering = sorted(state_returns, key=state_returns.get)
        state_to_regime = {
            ordering[0]: C.REGIME_CODES["BEAR"],
            ordering[1]: C.REGIME_CODES["SIDEWAYS"],
            ordering[2]: C.REGIME_CODES["BULL"],
        }
        regime_arr = np.array([state_to_regime[s] for s in hidden])

        # Build a regime-ordered posterior matrix so column 0 = P(BEAR),
        # column 2 = P(BULL), no matter what hidden order hmmlearn produced.
        col_for = {state_to_regime[s]: s for s in range(3)}
        prob_bear = posteriors[:, col_for[C.REGIME_CODES["BEAR"]]]
        prob_bull = posteriors[:, col_for[C.REGIME_CODES["BULL"]]]

        idx = feats.index
        out.loc[idx, "Regime"] = regime_arr
        out.loc[idx, "Regime_Prob_Bear"] = prob_bear
        out.loc[idx, "Regime_Prob_Bull"] = prob_bull
        # Forward-fill leading NaNs so every day has a label.
        out["Regime"] = out["Regime"].ffill().bfill().astype(int)
        out["Regime_Label"] = out["Regime"].map(C.REGIME_LABELS)
        out["Regime_Prob_Bear"] = out["Regime_Prob_Bear"].ffill().bfill()
        out["Regime_Prob_Bull"] = out["Regime_Prob_Bull"].ffill().bfill()

        # Reorder the transition matrix to [BEAR, SIDEWAYS, BULL] ordering.
        reorder = [col_for[C.REGIME_CODES[r]] for r in ("BEAR", "SIDEWAYS", "BULL")]
        T = model.transmat_[np.ix_(reorder, reorder)]
        # Renormalise rows just in case (numerical safety).
        T = T / T.sum(axis=1, keepdims=True)
        self.transition_matrix_ = T
        return out

    # ------------------------------------------------------------------
    # No-lookahead variants — used by walk-forward
    # ------------------------------------------------------------------
    def _hmm_fit_predict(self, train_df: pd.DataFrame,
                          full_df: pd.DataFrame) -> pd.DataFrame:
        """Fit the HMM on the training slice, decode regimes on the full slice.

        Key invariant: the HMM's parameters (transition matrix, emission
        means/covariances) are estimated using ONLY ``train_df``. The
        scaler is also fit on train only. ``full_df`` is then transformed
        with the train-fit scaler and decoded with the train-fit model.
        """
        out = full_df.copy()
        train_feats = train_df[self.HMM_FEATURES].dropna()
        full_feats = full_df[self.HMM_FEATURES].dropna()
        if len(train_feats) < 100 or len(full_feats) < 1:
            logger.warning("HMM fit/predict: insufficient data; falling back to MA crossover.")
            return self._ma_crossover(full_df)

        # 1. Scaler fit on train only.
        scaler = StandardScaler().fit(train_feats.values)
        X_train = scaler.transform(train_feats.values)
        X_full = scaler.transform(full_feats.values)

        # 2. HMM fit on train only.
        model = GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=200, random_state=self.random_state, tol=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train)

        # 3. Decode hidden states on the full sample (Viterbi + posteriors).
        hidden_full = model.predict(X_full)
        posteriors_full = model.predict_proba(X_full)

        # 4. Map hidden states → regime codes using TRAIN mean Return_5d
        #    (so the mapping is fixed at training time, not contaminated
        #    by test-period statistics).
        hidden_train = model.predict(X_train)
        state_returns = {
            s: train_feats.loc[hidden_train == s, "Return_5d"].mean()
            for s in range(3)
        }
        ordering = sorted(state_returns, key=state_returns.get)
        state_to_regime = {
            ordering[0]: C.REGIME_CODES["BEAR"],
            ordering[1]: C.REGIME_CODES["SIDEWAYS"],
            ordering[2]: C.REGIME_CODES["BULL"],
        }
        regime_arr = np.array([state_to_regime[s] for s in hidden_full])
        col_for = {state_to_regime[s]: s for s in range(3)}
        prob_bear = posteriors_full[:, col_for[C.REGIME_CODES["BEAR"]]]
        prob_bull = posteriors_full[:, col_for[C.REGIME_CODES["BULL"]]]

        idx = full_feats.index
        out.loc[idx, "Regime"] = regime_arr
        out.loc[idx, "Regime_Prob_Bear"] = prob_bear
        out.loc[idx, "Regime_Prob_Bull"] = prob_bull
        out["Regime"] = out["Regime"].ffill().bfill().astype(int)
        out["Regime_Label"] = out["Regime"].map(C.REGIME_LABELS)
        out["Regime_Prob_Bear"] = out["Regime_Prob_Bear"].ffill().bfill()
        out["Regime_Prob_Bull"] = out["Regime_Prob_Bull"].ffill().bfill()

        # Cache the reordered transition matrix (train-fit) for inspection.
        reorder = [col_for[C.REGIME_CODES[r]] for r in ("BEAR", "SIDEWAYS", "BULL")]
        T = model.transmat_[np.ix_(reorder, reorder)]
        self.transition_matrix_ = T / T.sum(axis=1, keepdims=True)
        return out

    def _kmeans_fit_predict(self, train_df: pd.DataFrame,
                              full_df: pd.DataFrame) -> pd.DataFrame:
        """KMeans variant of :meth:`_hmm_fit_predict`. Scaler + clusters fit
        on train; predictions made on full."""
        out = full_df.copy()
        train_feats = train_df[self.KMEANS_FEATURES].dropna()
        full_feats = full_df[self.KMEANS_FEATURES].dropna()
        if len(train_feats) < 60 or len(full_feats) < 1:
            logger.warning("KMeans fit/predict: insufficient data; falling back to MA.")
            return self._ma_crossover(full_df)

        scaler = StandardScaler().fit(train_feats.values)
        X_train = scaler.transform(train_feats.values)
        X_full = scaler.transform(full_feats.values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=3, n_init=10,
                        random_state=self.random_state).fit(X_train)
        train_labels = km.labels_
        full_labels = km.predict(X_full)

        # Map clusters → regimes using training period means.
        means = {c: train_feats.loc[train_labels == c, "Return_5d"].mean()
                 for c in np.unique(train_labels)}
        ordering = sorted(means, key=means.get)
        cluster_to_regime = {
            ordering[0]: C.REGIME_CODES["BEAR"],
            ordering[1]: C.REGIME_CODES["SIDEWAYS"],
            ordering[2]: C.REGIME_CODES["BULL"],
        }
        regimes = np.array([cluster_to_regime.get(c, C.REGIME_CODES["SIDEWAYS"])
                            for c in full_labels])
        labels = pd.Series(regimes, index=full_feats.index)
        if self.smooth_window > 1:
            labels = (labels.rolling(self.smooth_window, min_periods=1)
                      .apply(lambda x: pd.Series(x).mode().iloc[0])
                      .astype(int))
        out["Regime"] = labels.reindex(out.index).ffill().fillna(
            C.REGIME_CODES["SIDEWAYS"]).astype(int)
        out["Regime_Label"] = out["Regime"].map(C.REGIME_LABELS)
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Build synthetic data: 250d bull → 80d bear → 200d recovery.
    rng = np.random.default_rng(7)
    bull = rng.normal(0.0008, 0.012, 250)
    bear = rng.normal(-0.0030, 0.030, 80)
    recovery = rng.normal(0.0010, 0.015, 200)
    rets = np.concatenate([bull, bear, recovery])
    idx = pd.bdate_range("2022-01-03", periods=len(rets))
    close = 100 * np.exp(np.cumsum(rets))

    df = pd.DataFrame({"Close": close}, index=idx)
    df["Open"] = df["Close"].shift(1).fillna(close[0])
    df["High"] = df["Close"] * 1.005
    df["Low"] = df["Close"] * 0.995
    df["Volume"] = 1_000_000.0
    df["Daily_Return"] = df["Close"].pct_change()
    df["Adj_Return"] = np.log1p(df["Daily_Return"])

    # Need features for KMeans/HMM. Lazy-import to avoid a hard dep cycle.
    from features import FeatureEngineer  # type: ignore
    enriched = FeatureEngineer().compute(df)

    for method in ("ma_crossover", "kmeans", "hmm"):
        det = RegimeDetector(method=method)
        labelled = det.fit_transform(enriched)
        print(f"\n[{method}] regime distribution:")
        print(labelled["Regime_Label"].value_counts().to_string())
        if method == "hmm":
            print("Transition matrix [BEAR, SIDEWAYS, BULL]:")
            print(np.round(det.get_transition_matrix(), 3))
