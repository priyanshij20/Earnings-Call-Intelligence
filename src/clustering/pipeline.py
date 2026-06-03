"""
End-to-end clustering pipeline runner.
Produces:
  data/features/cluster_features.csv  — 18-feature company-quarter matrix
  data/features/cluster_results.json  — cluster assignments, profiles, diagnostics

Usage:
  python -m src.clustering.pipeline
  python src/clustering/pipeline.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.clustering import feature_engineering, preprocessing, dimensionality
from src.clustering import clustering as clust
from src.clustering import interpretation


ROOT     = Path(__file__).parent.parent.parent
FEAT_DIR = ROOT / "data" / "features"


def run():
    print("=" * 60)
    print("STEP 1+2 — Feature engineering")
    print("=" * 60)
    df, feature_cols = feature_engineering.run(
        feat_dir=FEAT_DIR,
        out_path=FEAT_DIR / "cluster_features.csv",
    )

    print("\n" + "=" * 60)
    print("STEP 3 — Preprocessing (scale + correlation check)")
    print("=" * 60)
    X_scaled, scaler, feat_df = preprocessing.run(df, feature_cols)

    print("\n" + "=" * 60)
    print("STEP 4 — Dimensionality reduction (PCA + t-SNE)")
    print("=" * 60)
    proj_df, pca_info = dimensionality.run(X_scaled, feat_df)

    print("\n" + "=" * 60)
    print("STEP 5 — Clustering (K-Means, DBSCAN, Hierarchical)")
    print("=" * 60)

    # K-Means
    km_diag = clust.kmeans_diagnostics(X_scaled, k_range=range(2, 7))
    optimal_k = km_diag["optimal_k"]
    proj_df["cluster_km"] = km_diag["labels_per_k"][optimal_k]

    # DBSCAN
    db_labels, eps_used, k_dists = clust.dbscan_fit(X_scaled, min_samples=3)
    proj_df["cluster_db"] = db_labels

    # Hierarchical (Ward, same k as K-Means for comparability)
    hc_labels, linkage_matrix = clust.hierarchical_fit(X_scaled, n_clusters=optimal_k)
    proj_df["cluster_hc"] = hc_labels

    # agreement table
    # exclude DBSCAN noise points (-1) from comparison
    mask = proj_df["cluster_db"] != -1
    if mask.sum() > optimal_k:
        agree = clust.compare_methods(
            X_scaled[mask],
            np.array(km_diag["labels_per_k"][optimal_k])[mask],
            db_labels[mask],
            hc_labels[mask],
        )
        print("\nPairwise method agreement (fraction of point-pairs co-assigned same way):")
        print(agree.to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 6 — Cluster interpretation")
    print("=" * 60)
    profiles = interpretation.cluster_profiles(proj_df, feature_cols, cluster_col="cluster_km")
    print("\nCluster archetypes:")
    for cl in profiles.index:
        size = (proj_df["cluster_km"] == cl).sum()
        print(f"  Cluster {cl}: {profiles.loc[cl, 'archetype']}  (n={size})")

    # attach archetype to proj_df for temporal analysis
    archetype_map = profiles["archetype"].to_dict()
    proj_df["archetype"] = proj_df["cluster_km"].map(archetype_map)

    temporal = interpretation.temporal_migration(proj_df)
    print("\nSample temporal migration (AMZN):")
    print(temporal[temporal["ticker"] == "AMZN"].to_string(index=False))

    returns_stats = interpretation.returns_by_cluster(proj_df, cluster_col="cluster_km")

    interpretation.save_results(
        proj_df=proj_df,
        profiles=profiles,
        km_diag=km_diag,
        pca_info=pca_info,
        returns_stats=returns_stats,
        out_path=FEAT_DIR / "cluster_results.json",
    )

    # also save the full proj_df for the dashboard
    proj_df.to_csv(FEAT_DIR / "cluster_projections.csv", index=False)
    print(f"Saved cluster_projections.csv ({proj_df.shape})")

    print("\n✓ Pipeline complete.")
    return proj_df, profiles, km_diag


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    run()
