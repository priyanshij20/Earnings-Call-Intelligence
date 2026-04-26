"""
Merges FinBERT, LM dictionary, FLS, and return features into a single matrix.
Output: data/features/nlp_features.csv
"""

import yaml
import pandas as pd
from pathlib import Path


def run(cfg: dict) -> pd.DataFrame:
    feat_dir = Path(cfg["paths"]["features"])

    finbert = pd.read_csv(feat_dir / "finbert_features.csv")
    lm      = pd.read_csv(feat_dir / "lm_features.csv")
    fls     = pd.read_csv(feat_dir / "fls_features.csv")
    returns = pd.read_csv(feat_dir / "returns.csv")

    # Merge on transcript_id (finbert/lm/fls) and ticker+quarter (returns)
    df = finbert.merge(lm.drop(columns=["date"], errors="ignore"),
                       on=["transcript_id", "ticker", "quarter"], how="inner")
    df = df.merge(fls.drop(columns=["date"], errors="ignore"),
                  on=["transcript_id", "ticker", "quarter"], how="inner")
    df = df.merge(returns[["ticker", "quarter", "return_1d", "return_3d", "return_7d"]],
                  on=["ticker", "quarter"], how="inner")

    df.to_csv(feat_dir / "nlp_features.csv", index=False)
    print(f"Feature matrix: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:", list(df.columns))
    print("\nSample:")
    print(df[["ticker", "quarter", "sentiment_overall", "sentiment_divergence",
              "lm_uncertainty", "fls_count", "return_1d", "return_3d"]].head(10).to_string())

    # Sanity checks
    print("\n--- Sanity Checks ---")
    corr_finbert_lm = df["sentiment_overall"].corr(df["lm_positive"])
    print(f"FinBERT vs LM positive correlation: {corr_finbert_lm:.3f} (expect positive)")
    avg_fls = df["fls_count"].mean()
    print(f"Avg FLS per transcript: {avg_fls:.1f} (expect 10–40)")
    corr_1d = df["sentiment_overall"].corr(df["return_1d"])
    print(f"Sentiment vs 1-day return (Pearson): {corr_1d:.3f} (research baseline ~0.1–0.3)")
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"\nNull counts:\n{nulls[nulls > 0]}")

    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from src.config import load as load_cfg
    run(load_cfg())
