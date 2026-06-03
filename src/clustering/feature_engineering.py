"""
Feature engineering: aggregates existing NLP features and computes derived features
to produce a ~20-feature company-quarter matrix for clustering.

Core features (from nlp_features.csv):
  - sentiment_overall, sentiment_divergence, lm_uncertainty, lm_litigious,
    lm_constraining, fls_ratio, fls_hedging_count, sentiment_std

Derived features:
  - Speaker-role sentiment variance (CEO/CFO/analyst gap)
  - Avg sentence length (linguistic complexity proxy)
  - Vocabulary diversity per call (unique word ratio)
  - QoQ change in sentiment, uncertainty, fls_ratio (linguistic momentum)
  - Intra-call uncertainty divergence (remarks vs Q&A uncertainty gap)
  - Strong-modal vs weak-modal ratio (linguistic confidence)
"""

from pathlib import Path
import numpy as np
import pandas as pd


SECTORS = {
    "tech":       ["AAPL", "MSFT", "META", "AMZN", "NVDA"],
    "finance":    ["JPM", "GS", "BAC"],
    "healthcare": ["JNJ", "PFE", "UNH"],
    "retail":     ["WMT", "TGT", "COST"],
    "energy":     ["XOM", "CVX"],
}
TICKER_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}


def _speaker_role_variance(sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each transcript, compute the std of mean-sentiment across speaker roles.
    High value → CEO and analyst tone diverge significantly.
    """
    role_means = (
        sent_df.groupby(["transcript_id", "role"])["sentiment_score"]
        .mean()
        .reset_index()
    )
    role_var = (
        role_means.groupby("transcript_id")["sentiment_score"]
        .std()
        .reset_index()
        .rename(columns={"sentiment_score": "role_sentiment_variance"})
    )
    return role_var


def _linguistic_complexity(sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per transcript:
      - avg_sentence_length: mean word count per sentence
      - vocab_diversity:     unique words / total words (type-token ratio proxy)
    """
    def _stats(group):
        words_per_sent = group["sentence"].str.split().str.len()
        all_words = " ".join(group["sentence"].str.lower()).split()
        return pd.Series({
            "avg_sentence_length": words_per_sent.mean(),
            "vocab_diversity":     len(set(all_words)) / max(len(all_words), 1),
        })

    return (
        sent_df.groupby("transcript_id")
        .apply(_stats)
        .reset_index()
    )


def _qoq_changes(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Quarter-over-quarter change for each feature, within each ticker."""
    df = df.sort_values(["ticker", "date"]).copy()
    delta_cols = {}
    for col in feature_cols:
        delta_cols[f"{col}_qoq"] = df.groupby("ticker")[col].diff()
    return df.assign(**delta_cols)


def run(feat_dir: Path, out_path: Path | None = None) -> pd.DataFrame:
    feat_dir = Path(feat_dir)

    base = pd.read_csv(feat_dir / "nlp_features.csv")
    base["date"] = pd.to_datetime(base["date"])
    base["sector"] = base["ticker"].map(TICKER_SECTOR)

    sent = pd.read_csv(feat_dir / "sentence_sentiments.csv")

    # ── speaker role variance ──────────────────────────────────────────────────
    role_var = _speaker_role_variance(sent)
    df = base.merge(role_var, on="transcript_id", how="left")

    # ── linguistic complexity ──────────────────────────────────────────────────
    complexity = _linguistic_complexity(sent)
    df = df.merge(complexity, on="transcript_id", how="left")

    # ── strong/weak modal ratio ────────────────────────────────────────────────
    # High ratio → more confident language ("will", "must") vs hedging ("may", "could")
    df["modal_confidence"] = df["lm_strong_modal"] / (df["lm_weak_modal"] + 1e-9)

    # ── uncertainty divergence (remarks vs Q&A gap) ────────────────────────────
    df["uncertainty_divergence"] = df["lm_uncertainty_remarks"] - df["lm_uncertainty_qa"]

    # ── QoQ changes for key dimensions ────────────────────────────────────────
    qoq_targets = [
        "sentiment_overall", "lm_uncertainty", "fls_ratio",
        "lm_constraining", "modal_confidence",
    ]
    df = _qoq_changes(df, qoq_targets)

    # ── select final feature set ───────────────────────────────────────────────
    # 8 core + 2 complexity + 1 role variance + 2 structural + 5 QoQ = 18 features
    feature_cols = [
        # core NLP dimensions
        "sentiment_overall",
        "sentiment_divergence",
        "sentiment_std",
        "lm_uncertainty",
        "lm_litigious",
        "lm_constraining",
        "fls_ratio",
        "fls_hedging_count",
        # structural
        "modal_confidence",
        "uncertainty_divergence",
        # speaker & linguistic complexity
        "role_sentiment_variance",
        "avg_sentence_length",
        "vocab_diversity",
        # momentum (QoQ change)
        "sentiment_overall_qoq",
        "lm_uncertainty_qoq",
        "fls_ratio_qoq",
        "lm_constraining_qoq",
        "modal_confidence_qoq",
    ]

    meta_cols = ["transcript_id", "ticker", "quarter", "date", "sector",
                 "return_1d", "return_3d", "return_7d"]

    out = df[meta_cols + feature_cols].copy()

    if out_path:
        out.to_csv(out_path, index=False)
        print(f"Saved cluster_features.csv  ({out.shape[0]} rows × {len(feature_cols)} features)")
        missing = out[feature_cols].isnull().sum()
        if missing.any():
            print("Null counts per feature:")
            print(missing[missing > 0].to_string())

    return out, feature_cols


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    ROOT = Path(__file__).parent.parent.parent
    feat_dir = ROOT / "data" / "features"
    run(feat_dir, feat_dir / "cluster_features.csv")
