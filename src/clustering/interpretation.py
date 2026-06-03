"""
Cluster interpretation and validation.

  - Centroid profiles (mean of each feature per cluster)
  - Archetype naming heuristics based on centroid signature
  - Temporal migration: does a company change cluster over time?
  - Business validation: Kruskal-Wallis test on post-earnings returns by cluster
  - Serialise results to cluster_results.json for dashboard consumption
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats


ARCHETYPE_RULES = [
    # Rules are checked in order; first match wins.
    # z-scores are relative to the cross-cluster mean (0 = average cluster).
    (
        "Crisis Communicators",
        lambda z: z.get("lm_uncertainty", 0) > 0.3 and z.get("sentiment_overall", 0) < -0.2,
    ),
    (
        "Cautious Hedgers",
        lambda z: z.get("lm_uncertainty", 0) > 0.2 and z.get("fls_hedging_count", 0) > 0.2,
    ),
    (
        "Confident Optimists",
        lambda z: z.get("sentiment_overall", 0) > 0.25 and z.get("modal_confidence", 0) > 0.1,
    ),
    (
        "Forward-Looking Visionaries",
        lambda z: z.get("fls_ratio", 0) > 0.3 and z.get("fls_hedging_count", 0) < 0.1,
    ),
    (
        "Measured Reporters",
        lambda z: abs(z.get("sentiment_overall", 0)) < 0.25 and z.get("lm_uncertainty", 0) < 0.1,
    ),
    (
        "Uncertainty Avoiders",
        lambda z: z.get("lm_uncertainty", 0) < -0.3 and z.get("modal_confidence", 0) > 0.0,
    ),
    (
        "Verbose Strategists",
        lambda z: z.get("avg_sentence_length", 0) > 0.3 and z.get("vocab_diversity", 0) > 0.1,
    ),
]


def _name_archetype(centroid_z: dict) -> str:
    for name, cond in ARCHETYPE_RULES:
        if cond(centroid_z):
            return name
    return "Linguistic Strategists"


def cluster_profiles(
    proj_df: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "cluster_km",
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by cluster with mean feature values (raw) and
    z-scored relative to cross-cluster mean for radar charts.
    """
    grp = proj_df.groupby(cluster_col)[feature_cols].mean()

    # z-score each feature across clusters (so radar arms are comparable)
    mu  = grp.mean()
    sig = grp.std().replace(0, 1)
    grp_z = (grp - mu) / sig

    grp.columns     = [f"{c}_mean" for c in grp.columns]
    grp_z.columns   = [f"{c}_z" for c in grp_z.columns]

    profiles = pd.concat([grp, grp_z], axis=1)
    profiles.index.name = "cluster"

    # assign archetype names
    archetype_map = {}
    for cl in profiles.index:
        z_dict = {col: profiles.loc[cl, f"{col}_z"] for col in feature_cols}
        archetype_map[cl] = _name_archetype(z_dict)

    # deduplicate names if same archetype assigned to multiple clusters
    seen = {}
    for cl, name in archetype_map.items():
        if name in seen:
            seen[name] += 1
            archetype_map[cl] = f"{name} {seen[name]}"
        else:
            seen[name] = 1

    profiles["archetype"] = pd.Series(archetype_map)
    return profiles


def temporal_migration(proj_df: pd.DataFrame, cluster_col: str = "cluster_km") -> pd.DataFrame:
    """
    For each ticker, list cluster assignment per quarter in chronological order.
    Returns a DataFrame with columns: ticker, quarter, date, cluster, archetype (if present).
    """
    cols = ["ticker", "quarter", "date", cluster_col]
    if "archetype" in proj_df.columns:
        cols.append("archetype")
    return (
        proj_df[cols]
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def returns_by_cluster(
    proj_df: pd.DataFrame,
    cluster_col: str = "cluster_km",
    return_col: str = "return_3d",
) -> dict:
    """
    Kruskal-Wallis test: are 3-day returns significantly different across clusters?
    Returns dict with group means, KW statistic, and p-value.
    """
    groups = [
        proj_df.loc[proj_df[cluster_col] == cl, return_col].dropna().values
        for cl in sorted(proj_df[cluster_col].dropna().unique())
    ]
    groups = [g for g in groups if len(g) >= 3]

    if len(groups) < 2:
        return {"kw_stat": None, "p_value": None, "group_means": {}}

    kw_stat, p_val = stats.kruskal(*groups)
    cluster_ids = sorted(proj_df[cluster_col].dropna().unique())
    group_means = {
        int(cl): float(proj_df.loc[proj_df[cluster_col] == cl, return_col].mean())
        for cl in cluster_ids
    }
    print(f"\nKruskal-Wallis ({return_col} by cluster): "
          f"H={kw_stat:.3f}  p={p_val:.4f}")

    return {"kw_stat": float(kw_stat), "p_value": float(p_val), "group_means": group_means}


def save_results(
    proj_df: pd.DataFrame,
    profiles: pd.DataFrame,
    km_diag: dict,
    pca_info: dict,
    returns_stats: dict,
    out_path: Path,
) -> None:
    """Serialise everything the dashboard needs into a single JSON."""
    # cluster membership list
    membership = (
        proj_df[["transcript_id", "ticker", "quarter", "date",
                 "sector", "cluster_km", "cluster_hc", "cluster_db",
                 "pca_1", "pca_2", "pca_3", "tsne_1", "tsne_2"]]
        .copy()
    )
    membership["date"] = membership["date"].astype(str)

    archetype_series = profiles["archetype"]

    payload = {
        "membership":            membership.to_dict(orient="records"),
        "profiles":              profiles.reset_index().to_dict(orient="records"),
        "km_diagnostics": {
            "k_values":          km_diag["k_values"],
            "inertias":          km_diag["inertias"],
            "silhouette_scores": km_diag["silhouette_scores"],
            "gap_stats":         km_diag["gap_stats"],
            "optimal_k":         km_diag["optimal_k"],
        },
        "pca_info":              pca_info,
        "returns_by_cluster":    returns_stats,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved cluster_results.json → {out_path}")
