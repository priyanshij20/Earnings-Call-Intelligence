"""
Dimensionality reduction for visualization and analysis.
  - PCA: 2D + 3D projections, scree plot data
  - t-SNE: 2D projection
  - Returns a DataFrame with all projections appended.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run(
    X_scaled: np.ndarray,
    feat_df: pd.DataFrame,
    random_state: int = 42,
    tsne_perplexity: float = 25.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns
    -------
    proj_df   : feat_df with pca_1/pca_2/pca_3/tsne_1/tsne_2 columns appended
    pca_info  : dict with 'explained_variance_ratio' and 'components' (n_components x n_features)
    """
    n_samples = X_scaled.shape[0]

    # ── PCA ───────────────────────────────────────────────────────────────────
    n_pca = min(10, n_samples, X_scaled.shape[1])
    pca = PCA(n_components=n_pca, random_state=random_state)
    coords_pca = pca.fit_transform(X_scaled)

    proj_df = feat_df.copy()
    proj_df["pca_1"] = coords_pca[:, 0]
    proj_df["pca_2"] = coords_pca[:, 1]
    proj_df["pca_3"] = coords_pca[:, 2] if n_pca >= 3 else 0.0

    pca_info = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "components": pca.components_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
    }

    print(f"\nPCA: PC1={pca.explained_variance_ratio_[0]:.1%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}  "
          f"PC3={pca.explained_variance_ratio_[2]:.1%}  "
          f"(2D = {sum(pca.explained_variance_ratio_[:2]):.1%} total)")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    perp = min(tsne_perplexity, (n_samples - 1) / 3)
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=random_state,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    coords_tsne = tsne.fit_transform(X_scaled)
    proj_df["tsne_1"] = coords_tsne[:, 0]
    proj_df["tsne_2"] = coords_tsne[:, 1]

    print(f"t-SNE: perplexity={perp:.0f}, KL={tsne.kl_divergence_:.4f}")

    return proj_df, pca_info
