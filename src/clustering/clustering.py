"""
Clustering implementations with selection diagnostics.

K-Means:
  - k=2..8, elbow (inertia) + silhouette + gap statistic
  - Returns optimal k and cluster labels

DBSCAN:
  - k-distance plot to guide epsilon selection
  - Reports outlier fraction

Hierarchical (Agglomerative):
  - Ward / complete / average linkage
  - Dendrogram data for visualization

All methods write cluster labels into the projection DataFrame.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ── K-Means ────────────────────────────────────────────────────────────────────

def _gap_statistic(X: np.ndarray, k: int, n_boot: int = 20, rng=None) -> float:
    """Compute gap statistic for a single k (Monte Carlo reference)."""
    if rng is None:
        rng = np.random.default_rng(42)

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    wk = np.log(km.inertia_ + 1e-9)

    # reference distribution: uniform over feature bounding box
    ref_wk = []
    mins, maxs = X.min(axis=0), X.max(axis=0)
    for _ in range(n_boot):
        X_ref = rng.uniform(mins, maxs, size=X.shape)
        km_ref = KMeans(n_clusters=k, n_init=5, random_state=42)
        km_ref.fit(X_ref)
        ref_wk.append(np.log(km_ref.inertia_ + 1e-9))

    return float(np.mean(ref_wk) - wk)


def kmeans_diagnostics(
    X: np.ndarray,
    k_range: range = range(2, 7),
    gap_boots: int = 20,
) -> dict:
    """
    Returns dict with keys:
      k_values, inertias, silhouette_scores, gap_stats, optimal_k, labels_per_k
    """
    rng = np.random.default_rng(42)
    inertias, sils, gaps, labels_per_k = [], [], [], {}

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels))
        gaps.append(_gap_statistic(X, k, n_boot=gap_boots, rng=rng))
        labels_per_k[k] = labels.tolist()

    # optimal k: highest silhouette (simple, interpretable)
    sil_arr = np.array(sils)
    optimal_k = list(k_range)[int(np.argmax(sil_arr))]
    print(f"\nK-Means — optimal k={optimal_k}  "
          f"silhouette={max(sils):.3f}  inertia={inertias[optimal_k - k_range.start]:.1f}")

    return {
        "k_values":          list(k_range),
        "inertias":          inertias,
        "silhouette_scores": sils,
        "gap_stats":         gaps,
        "optimal_k":         optimal_k,
        "labels_per_k":      labels_per_k,
    }


# ── DBSCAN ─────────────────────────────────────────────────────────────────────

def dbscan_fit(
    X: np.ndarray,
    eps: float | None = None,
    min_samples: int = 3,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Auto-select eps from the k-distance elbow if not provided.
    Returns (labels, eps_used, k_distances_sorted).
    """
    k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dists = np.sort(distances[:, k - 1])[::-1]

    if eps is None:
        # simple elbow: largest second-derivative point
        d2 = np.diff(np.diff(k_dists))
        eps = float(k_dists[int(np.argmin(d2)) + 1])
        eps = max(eps, 0.3)  # lower bound to avoid degenerate clustering

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"\nDBSCAN — eps={eps:.3f}  min_samples={min_samples}  "
          f"clusters={n_clusters}  outliers={n_noise} ({n_noise/len(labels):.1%})")

    return labels, eps, k_dists


# ── Hierarchical ───────────────────────────────────────────────────────────────

def hierarchical_fit(
    X: np.ndarray,
    n_clusters: int,
    linkage_method: str = "ward",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (flat_labels, linkage_matrix) for the given linkage method.
    linkage_matrix is compatible with scipy.cluster.hierarchy.dendrogram.
    """
    if linkage_method == "ward":
        Z = linkage(X, method="ward")
    else:
        dist = pdist(X)
        Z = linkage(dist, method=linkage_method)

    labels = fcluster(Z, n_clusters, criterion="maxclust") - 1  # 0-indexed
    sil = silhouette_score(X, labels)
    print(f"\nHierarchical ({linkage_method}) — k={n_clusters}  silhouette={sil:.3f}")

    return labels, Z


# ── Consensus / comparison ─────────────────────────────────────────────────────

def compare_methods(
    X: np.ndarray,
    km_labels: np.ndarray,
    db_labels: np.ndarray,
    hc_labels: np.ndarray,
) -> pd.DataFrame:
    """Agreement table: what fraction of pairs are co-assigned in all three methods?"""
    n = len(km_labels)
    agree_km_hc, agree_km_db, agree_hc_db = 0, 0, 0
    total = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            same_km = km_labels[i] == km_labels[j]
            same_db = db_labels[i] == db_labels[j]
            same_hc = hc_labels[i] == hc_labels[j]
            agree_km_hc += int(same_km == same_hc)
            agree_km_db += int(same_km == same_db)
            agree_hc_db += int(same_hc == same_db)

    return pd.DataFrame({
        "pair":      ["KMeans-Hierarchical", "KMeans-DBSCAN", "Hierarchical-DBSCAN"],
        "agreement": [agree_km_hc / total, agree_km_db / total, agree_hc_db / total],
    })
