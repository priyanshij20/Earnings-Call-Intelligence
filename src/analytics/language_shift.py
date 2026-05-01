"""
Language shift detector: tracks quarter-over-quarter changes in NLP signals
per company and flags calls where the narrative changed significantly.

A "shift event" = any quarter where sentiment_overall, lm_uncertainty, or
fls_count moved more than 1.5 std-devs from that company's rolling mean.
Checks what happened to the stock in the window after each shift event.

Output: data/features/language_shifts.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

FEATURES_DIR = Path("data/features")

SHIFT_FEATURES = ["sentiment_overall", "lm_uncertainty", "fls_hedging_count"]
THRESHOLD_STD  = 1.5   # flag if delta > this many std-devs of the company's series


def compute_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each company, sort quarters chronologically and compute:
      - QoQ delta for each SHIFT_FEATURE
      - z-score of that delta (relative to company's own std)
      - whether any feature breached the threshold
    """
    records = []

    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").reset_index(drop=True)
        if len(g) < 3:
            continue

        # Per-company stats for z-scoring
        stats = {}
        for feat in SHIFT_FEATURES:
            if feat not in g.columns:
                continue
            deltas = g[feat].diff().dropna()
            stats[feat] = {
                "mean": deltas.mean(),
                "std":  deltas.std() if deltas.std() > 0 else np.nan,
            }

        for i in range(1, len(g)):
            row = g.iloc[i]
            prev = g.iloc[i - 1]

            shift_flags = {}
            max_z = 0.0

            for feat in SHIFT_FEATURES:
                if feat not in g.columns or pd.isna(row[feat]) or pd.isna(prev[feat]):
                    shift_flags[f"{feat}_delta"] = np.nan
                    shift_flags[f"{feat}_zscore"] = np.nan
                    continue
                delta = row[feat] - prev[feat]
                s = stats.get(feat, {})
                z = (delta - s.get("mean", 0)) / s.get("std", np.nan) \
                    if not np.isnan(s.get("std", np.nan)) else np.nan
                shift_flags[f"{feat}_delta"]  = round(delta, 6)
                shift_flags[f"{feat}_zscore"] = round(z, 4) if not np.isnan(z) else np.nan
                if not np.isnan(z):
                    max_z = max(max_z, abs(z))

            is_shift = max_z >= THRESHOLD_STD

            record = {
                "ticker":        ticker,
                "quarter":       row["quarter"],
                "date":          row["date"],
                "prev_quarter":  prev["quarter"],
                "is_shift_event": is_shift,
                "max_z_score":   round(max_z, 4),
                # Current values for context
                "sentiment_overall":  row.get("sentiment_overall"),
                "lm_uncertainty":     row.get("lm_uncertainty"),
                "fls_hedging_count":  row.get("fls_hedging_count"),
                # Stock returns after this call
                "return_1d":  row.get("return_1d"),
                "return_3d":  row.get("return_3d"),
                "return_7d":  row.get("return_7d"),
            }
            record.update(shift_flags)
            records.append(record)

    return pd.DataFrame(records)


def print_findings(shifts: pd.DataFrame):
    shift_events = shifts[shifts["is_shift_event"]].copy()
    non_events   = shifts[~shifts["is_shift_event"]].copy()

    print(f"\n── Language Shift Summary ──")
    print(f"  Total QoQ comparisons: {len(shifts)}")
    print(f"  Shift events flagged:  {len(shift_events)} "
          f"({len(shift_events)/len(shifts):.0%})")

    print(f"\n── Returns after shift events vs. normal quarters ──")
    for ret in ["return_1d", "return_3d", "return_7d"]:
        sv = shift_events[ret].dropna()
        nv = non_events[ret].dropna()
        from scipy import stats as sp
        if len(sv) >= 5 and len(nv) >= 5:
            u, p = sp.mannwhitneyu(sv, nv, alternative="two-sided")
            print(f"  {ret}:  shift mean={sv.mean():+.4f}  "
                  f"normal mean={nv.mean():+.4f}  "
                  f"p={p:.4f}  {'✓' if p < 0.05 else '✗'}")
        else:
            print(f"  {ret}:  too few shift events ({len(sv)}) for test")

    print(f"\n── Top 15 most dramatic narrative shifts ──")
    top = shift_events.nlargest(15, "max_z_score")[
        ["ticker", "quarter", "prev_quarter", "max_z_score",
         "sentiment_overall_delta", "lm_uncertainty_delta",
         "fls_hedging_count_delta", "return_3d"]
    ]
    for _, r in top.iterrows():
        ret_str = f"{r['return_3d']:+.1%}" if pd.notna(r["return_3d"]) else "  N/A"
        sent_d  = f"{r.get('sentiment_overall_delta', np.nan):+.3f}" \
                  if pd.notna(r.get("sentiment_overall_delta")) else "  N/A"
        unc_d   = f"{r.get('lm_uncertainty_delta', np.nan):+.4f}" \
                  if pd.notna(r.get("lm_uncertainty_delta")) else "  N/A"
        print(f"  {r['ticker']:5s} {r['prev_quarter']}→{r['quarter']}  "
              f"z={r['max_z_score']:.2f}  "
              f"Δsent={sent_d}  Δuncert={unc_d}  "
              f"3d_return={ret_str}")

    print(f"\n── Shift events by company ──")
    by_co = shift_events.groupby("ticker").agg(
        n_shifts=("is_shift_event", "sum"),
        avg_3d_return=("return_3d", "mean")
    ).sort_values("n_shifts", ascending=False)
    for ticker, row in by_co.iterrows():
        print(f"  {ticker:5s}  {int(row['n_shifts'])} shifts  "
              f"avg 3d return after shift: {row['avg_3d_return']:+.3f}")


def run():
    df = pd.read_csv(FEATURES_DIR / "nlp_features.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    shifts = compute_shifts(df)
    print_findings(shifts)

    shifts.to_csv(FEATURES_DIR / "language_shifts.csv", index=False)
    print(f"\nSaved language_shifts.csv ({len(shifts)} rows)")
    return shifts


if __name__ == "__main__":
    run()
