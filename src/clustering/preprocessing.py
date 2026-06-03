"""
Preprocessing for clustering:
  1. Drop QoQ features for first-quarter-per-ticker (NaN by design)
  2. Median-impute any remaining nulls
  3. Clip extreme outliers at ±4 σ before scaling
  4. StandardScaler → zero mean, unit variance
  5. Correlation report (flag pairs ρ > 0.85)
  6. Quick PCA to show how many components explain 90 % variance
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run(
    df: pd.DataFrame,
    feature_cols: list[str],
    corr_threshold: float = 0.85,
) -> tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Returns
    -------
    X_scaled   : (n_samples, n_features) numpy array, ready for clustering
    scaler     : fitted StandardScaler (for inverse_transform later)
    feat_df    : clean DataFrame aligned with X_scaled rows (meta + features)
    """
    feat_df = df.copy()

    # ── impute NaN (QoQ NaN for first quarter, rare nulls) ────────────────────
    for col in feature_cols:
        if feat_df[col].isnull().any():
            feat_df[col] = feat_df[col].fillna(feat_df[col].median())

    X = feat_df[feature_cols].values.astype(float)

    # ── clip outliers at ±4σ before scaling ───────────────────────────────────
    col_means = X.mean(axis=0)
    col_stds  = X.std(axis=0) + 1e-9
    X = np.clip(X, col_means - 4 * col_stds, col_means + 4 * col_stds)
    feat_df[feature_cols] = X

    # ── StandardScaler ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── correlation report ────────────────────────────────────────────────────
    corr = pd.DataFrame(X_scaled, columns=feature_cols).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [
        (c, r, corr.loc[r, c])
        for c in upper.columns
        for r in upper.index
        if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > corr_threshold
    ]
    if high_corr:
        print(f"\nHighly correlated feature pairs (|ρ| > {corr_threshold}):")
        for a, b, rho in sorted(high_corr, key=lambda x: -x[2]):
            print(f"  {a}  ↔  {b}   ρ = {rho:.3f}")
    else:
        print(f"\nNo feature pairs exceed |ρ| = {corr_threshold} — no redundancy concern.")

    # ── PCA variance explained ────────────────────────────────────────────────
    pca_full = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    print(f"\nPCA: {n90} components explain 90% variance, {n95} explain 95%  "
          f"(out of {len(feature_cols)} features)")

    return X_scaled, scaler, feat_df
