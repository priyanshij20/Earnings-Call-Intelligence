"""
Correlation analysis: NLP features vs. post-earnings stock returns.

Outputs:
  data/features/correlation_results.csv  — full Spearman matrix with p-values
  data/features/findings_summary.json    — key findings for the dashboard / README
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import product

FEATURES_DIR = Path("data/features")
MARKET_DIR   = Path("data/market")

# Sectors map matches settings.yaml
SECTORS = {
    "tech":       ["AAPL", "MSFT", "META", "AMZN", "NVDA"],
    "finance":    ["JPM", "GS", "BAC"],
    "healthcare": ["JNJ", "PFE", "UNH"],
    "retail":     ["WMT", "TGT", "COST"],
    "energy":     ["XOM", "CVX"],
}

NLP_FEATURES = [
    "sentiment_overall", "sentiment_remarks", "sentiment_qa",
    "sentiment_divergence",
    "positive_ratio", "negative_ratio", "neutral_ratio", "sentiment_std",
    "lm_positive", "lm_negative", "lm_uncertainty", "lm_litigious",
    "lm_constraining", "lm_strong_modal", "lm_weak_modal",
    "lm_uncertainty_remarks", "lm_uncertainty_qa",
    "lm_negative_remarks", "lm_negative_qa",
    "fls_count", "fls_positive_count", "fls_hedging_count",
    "fls_ratio", "fls_sentiment_avg",
]
RETURN_WINDOWS = ["return_1d", "return_3d", "return_7d"]


# ── helpers ──────────────────────────────────────────────────────────────────

def spearman(x: pd.Series, y: pd.Series):
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def compute_volatility_5d(ticker: str, call_date: str) -> float | None:
    """Return std-dev of daily returns over the 5 trading days after the call."""
    csv = MARKET_DIR / f"{ticker}.csv"
    if not csv.exists():
        return None
    try:
        prices = pd.read_csv(csv, index_col=0, skiprows=[1, 2])
        prices.index = pd.to_datetime(prices.index, errors="coerce")
        prices = prices[prices.index.notna()].sort_index()
        prices["daily_ret"] = prices["Close"].pct_change()
        after = prices.loc[prices.index >= pd.Timestamp(call_date)]
        if len(after) < 5:
            return None
        return float(after["daily_ret"].iloc[1:6].std())
    except Exception:
        return None


# ── Part 1: Spearman correlation matrix ──────────────────────────────────────

def run_spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_tests = len(NLP_FEATURES) * len(RETURN_WINDOWS)
    bonferroni_alpha = 0.05 / n_tests

    for feat, ret in product(NLP_FEATURES, RETURN_WINDOWS):
        if feat not in df.columns or ret not in df.columns:
            continue
        rho, p = spearman(df[feat], df[ret])
        rows.append({
            "feature":         feat,
            "return_window":   ret,
            "spearman_rho":    rho,
            "p_value":         p,
            "n":               int(df[[feat, ret]].dropna().shape[0]),
            "significant_raw": p < 0.05 if not np.isnan(p) else False,
            "significant_bonferroni": p < bonferroni_alpha if not np.isnan(p) else False,
            "bonferroni_alpha": bonferroni_alpha,
        })

    result = pd.DataFrame(rows).sort_values("p_value")
    print(f"\n── Spearman matrix ({len(result)} tests, Bonferroni α={bonferroni_alpha:.5f}) ──")
    sig = result[result["significant_raw"]]
    print(f"  Significant at p<0.05 (uncorrected): {len(sig)}")
    print(f"  Significant after Bonferroni:         {result['significant_bonferroni'].sum()}")
    if len(sig):
        print("\n  Top 10 by |rho|:")
        top = sig.reindex(sig["spearman_rho"].abs().sort_values(ascending=False).index).head(10)
        for _, r in top.iterrows():
            print(f"    {r['feature']:35s} vs {r['return_window']}  "
                  f"ρ={r['spearman_rho']:+.3f}  p={r['p_value']:.4f}")
    return result


# ── Part 2: Divergence quartile test ─────────────────────────────────────────

def run_divergence_quartile(df: pd.DataFrame) -> dict:
    out = {}
    col = "sentiment_divergence"
    valid = df[df[col].notna()].copy()
    valid["divergence_quartile"] = pd.qcut(valid[col], 4, labels=["Q1","Q2","Q3","Q4"])

    print(f"\n── Divergence quartile test (n={len(valid)}) ──")
    print(f"  Q1 = least spinning (Q&A ≈ remarks)")
    print(f"  Q4 = most spinning  (remarks >> Q&A)")

    for ret in RETURN_WINDOWS:
        q1 = valid.loc[valid["divergence_quartile"] == "Q1", ret].dropna()
        q4 = valid.loc[valid["divergence_quartile"] == "Q4", ret].dropna()
        u_stat, p = stats.mannwhitneyu(q1, q4, alternative="two-sided")
        key = f"divergence_q1_vs_q4_{ret}"
        out[key] = {
            "q1_mean_return": float(q1.mean()),
            "q4_mean_return": float(q4.mean()),
            "q1_n": int(len(q1)),
            "q4_n": int(len(q4)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p),
            "direction": "Q4 worse" if q4.mean() < q1.mean() else "Q4 better",
        }
        pct_q4_negative = (q4 < 0).mean()
        print(f"\n  {ret}:")
        print(f"    Q1 (low divergence) mean = {q1.mean():+.4f}")
        print(f"    Q4 (high divergence) mean = {q4.mean():+.4f}  "
              f"({pct_q4_negative:.0%} negative)")
        print(f"    Mann-Whitney p = {p:.4f}  {'✓ significant' if p < 0.05 else '✗ not significant'}")

    # Uncertainty vs volatility
    print(f"\n── LM Uncertainty vs. post-earnings volatility ──")
    df = df.copy()
    df["volatility_5d"] = df.apply(
        lambda r: compute_volatility_5d(r["ticker"], r["date"]), axis=1
    )
    rho, p = spearman(df["lm_uncertainty"], df["volatility_5d"])
    out["uncertainty_vs_volatility"] = {"spearman_rho": rho, "p_value": p}
    print(f"  lm_uncertainty vs 5-day volatility:  ρ={rho:+.3f}  p={p:.4f}  "
          f"({'✓ sig' if p < 0.05 else '✗ n.s.'})")

    # Also per-section
    rho_r, p_r = spearman(df["lm_uncertainty_remarks"], df["volatility_5d"])
    rho_q, p_q = spearman(df["lm_uncertainty_qa"],     df["volatility_5d"])
    out["uncertainty_remarks_vs_volatility"] = {"spearman_rho": rho_r, "p_value": p_r}
    out["uncertainty_qa_vs_volatility"]      = {"spearman_rho": rho_q, "p_value": p_q}
    print(f"  lm_uncertainty_remarks vs volatility: ρ={rho_r:+.3f}  p={p_r:.4f}")
    print(f"  lm_uncertainty_qa      vs volatility: ρ={rho_q:+.3f}  p={p_q:.4f}")

    return out


# ── Part 3: Sector-level correlations ────────────────────────────────────────

def run_sector_correlations(df: pd.DataFrame) -> dict:
    ticker_to_sector = {t: s for s, tickers in SECTORS.items() for t in tickers}
    df = df.copy()
    df["sector"] = df["ticker"].map(ticker_to_sector)

    focus_features = ["sentiment_divergence", "lm_uncertainty", "fls_hedging_count",
                      "sentiment_overall"]
    out = {}

    print(f"\n── Sector-level Spearman (sentiment_divergence vs return_3d) ──")
    for sector in SECTORS:
        sub = df[df["sector"] == sector]
        results = {}
        for feat in focus_features:
            for ret in RETURN_WINDOWS:
                rho, p = spearman(sub[feat], sub[ret])
                results[f"{feat}_{ret}"] = {"rho": rho, "p": p, "n": int(sub[[feat,ret]].dropna().shape[0])}
        out[sector] = results
        rho_div = results.get("sentiment_divergence_return_3d", {})
        print(f"  {sector:12s} n={len(sub):2d}  "
              f"divergence→3d  ρ={rho_div.get('rho', float('nan')):+.3f}  "
              f"p={rho_div.get('p', float('nan')):.3f}")

    return out


# ── Part 4: Temporal split (pre/post 2022) ───────────────────────────────────

def run_temporal_split(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    pre  = df[df["year"] < 2022]
    post = df[df["year"] >= 2022]

    print(f"\n── Temporal split: pre-2022 (n={len(pre)}) vs post-2022 (n={len(post)}) ──")
    focus = ["sentiment_divergence", "lm_uncertainty", "sentiment_overall"]
    out = {"pre_2022": {}, "post_2022": {}}

    for feat in focus:
        for ret in ["return_1d", "return_3d"]:
            rho_pre,  p_pre  = spearman(pre[feat],  pre[ret])
            rho_post, p_post = spearman(post[feat], post[ret])
            out["pre_2022"][f"{feat}_{ret}"]  = {"rho": rho_pre,  "p": p_pre}
            out["post_2022"][f"{feat}_{ret}"] = {"rho": rho_post, "p": p_post}
        # Print divergence row
        rho_p, p_p   = out["pre_2022"].get(f"{feat}_return_3d",  {}).get("rho", float("nan")), \
                        out["pre_2022"].get(f"{feat}_return_3d",  {}).get("p",   float("nan"))
        rho_po, p_po = out["post_2022"].get(f"{feat}_return_3d", {}).get("rho", float("nan")), \
                        out["post_2022"].get(f"{feat}_return_3d", {}).get("p",   float("nan"))
        print(f"  {feat:30s}  pre ρ={rho_p:+.3f}  post ρ={rho_po:+.3f}")

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    df = pd.read_csv(FEATURES_DIR / "nlp_features.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    print(f"Loaded feature matrix: {df.shape[0]} rows × {df.shape[1]} cols")

    corr_df   = run_spearman_matrix(df)
    div_out   = run_divergence_quartile(df)
    sector_out = run_sector_correlations(df)
    temporal_out = run_temporal_split(df)

    # Save correlation matrix
    corr_df.to_csv(FEATURES_DIR / "correlation_results.csv", index=False)
    print(f"\nSaved correlation_results.csv ({len(corr_df)} rows)")

    # Assemble and save findings summary
    top_sig = corr_df[corr_df["significant_raw"]].head(10).to_dict("records")
    summary = {
        "n_transcripts":        int(df.shape[0]),
        "n_features_tested":    len(NLP_FEATURES),
        "n_hypotheses":         len(NLP_FEATURES) * len(RETURN_WINDOWS),
        "bonferroni_alpha":     round(0.05 / (len(NLP_FEATURES) * len(RETURN_WINDOWS)), 6),
        "top_correlations":     top_sig,
        "divergence_quartile":  div_out,
        "sector_correlations":  sector_out,
        "temporal_split":       temporal_out,
    }
    out_path = FEATURES_DIR / "findings_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved findings_summary.json")

    return corr_df, summary


if __name__ == "__main__":
    run()
