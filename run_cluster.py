"""
Standalone clustering pipeline runner — writes all outputs to files.
Run: python run_cluster.py
"""
import os
# Must be set before numpy/scipy import to avoid BLAS thread-detection hang on macOS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

FEAT_DIR = ROOT / "data" / "features"
LOG_FILE = ROOT / "cluster_run.log"


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


log("=== Clustering Pipeline Start ===")
log("Python: " + sys.version)

try:
    import numpy as np
    log(f"numpy {np.__version__}")
except Exception as e:
    log(f"FAIL numpy: {e}"); sys.exit(1)

try:
    import pandas as pd
    log(f"pandas {pd.__version__}")
except Exception as e:
    log(f"FAIL pandas: {e}"); sys.exit(1)

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    import sklearn
    log(f"sklearn {sklearn.__version__}")
except Exception as e:
    log(f"FAIL sklearn: {e}"); sys.exit(1)

try:
    from scipy import stats
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    import scipy
    log(f"scipy {scipy.__version__}")
except Exception as e:
    log(f"FAIL scipy: {e}"); sys.exit(1)

log("All imports OK — running pipeline...")

# ── Feature Engineering ────────────────────────────────────────────────────────
from src.clustering.feature_engineering import run as fe_run
df, feature_cols = fe_run(FEAT_DIR, FEAT_DIR / "cluster_features.csv")
log(f"Features: {df.shape[0]} rows x {len(feature_cols)} features")

# ── Preprocessing ──────────────────────────────────────────────────────────────
from src.clustering.preprocessing import run as pp_run
X_scaled, scaler, feat_df = pp_run(df, feature_cols)
log(f"Scaled X: {X_scaled.shape}")

# ── Dimensionality Reduction ───────────────────────────────────────────────────
from src.clustering.dimensionality import run as dim_run
proj_df, pca_info = dim_run(X_scaled, feat_df)
log(f"Projections: {list(proj_df.columns[-5:])}")

# ── Clustering ─────────────────────────────────────────────────────────────────
from src.clustering import clustering as clust
km_diag = clust.kmeans_diagnostics(X_scaled, k_range=range(2, 7))
optimal_k = km_diag["optimal_k"]
log(f"K-Means optimal k={optimal_k}")
proj_df["cluster_km"] = km_diag["labels_per_k"][optimal_k]

db_labels, eps_used, k_dists = clust.dbscan_fit(X_scaled, min_samples=3)
proj_df["cluster_db"] = db_labels
log(f"DBSCAN eps={eps_used:.3f}")

hc_labels, linkage_matrix = clust.hierarchical_fit(X_scaled, n_clusters=optimal_k)
proj_df["cluster_hc"] = hc_labels

# ── Interpretation ─────────────────────────────────────────────────────────────
from src.clustering import interpretation
profiles = interpretation.cluster_profiles(proj_df, feature_cols, cluster_col="cluster_km")
archetype_map = profiles["archetype"].to_dict()
proj_df["archetype"] = proj_df["cluster_km"].map(archetype_map)
log(f"Archetypes: {archetype_map}")

returns_stats = interpretation.returns_by_cluster(proj_df, cluster_col="cluster_km")

interpretation.save_results(
    proj_df=proj_df,
    profiles=profiles,
    km_diag=km_diag,
    pca_info=pca_info,
    returns_stats=returns_stats,
    out_path=FEAT_DIR / "cluster_results.json",
)

proj_df.to_csv(FEAT_DIR / "cluster_projections.csv", index=False)
log(f"Saved cluster_projections.csv ({proj_df.shape})")
log("=== Pipeline Complete ===")
