"""
Validation suite for the NLP pipeline.
Run: python src/validate.py
Prints results for all three validation levels so you can follow along manually.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

FEATURES   = Path("data/features")
PROCESSED  = Path("data/processed")

SEP = "=" * 65


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ─────────────────────────────────────────────────────────────
# LEVEL 1 — Statistical Sanity Checks
# ─────────────────────────────────────────────────────────────

def check_sentiment_distribution(df):
    section("1A · Sentiment Distribution")
    mean = df["sentiment_overall"].mean()
    std  = df["sentiment_overall"].std()
    mn   = df["sentiment_overall"].min()
    mx   = df["sentiment_overall"].max()
    print(f"  mean = {mean:.4f}  (expect 0.05–0.20 — calls are optimistic)")
    print(f"  std  = {std:.4f}")
    print(f"  min  = {mn:.4f}   max = {mx:.4f}")
    if 0.05 <= mean <= 0.30:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — mean outside expected range")

    print("\n  Per-company mean sentiment:")
    per_co = df.groupby("ticker")["sentiment_overall"].mean().sort_values(ascending=False)
    for t, v in per_co.items():
        bar = "█" * int(v * 80)
        print(f"    {t:5s}  {v:.3f}  {bar}")


def check_section_ratio(processed_dir):
    section("1B · Section Word-Count Ratio (Remarks vs Q&A)")
    ratios = []
    for f in sorted(processed_dir.glob("*.json")):
        with open(f) as fp:
            t = json.load(fp)
        rem_words = sum(len(b["text"].split()) for b in t["prepared_remarks"])
        qa_words  = sum(len(b["text"].split()) for b in t["qa_section"])
        total = rem_words + qa_words
        if total > 0:
            ratios.append(rem_words / total)

    arr = np.array(ratios)
    print(f"  Remarks % of total  mean={arr.mean():.1%}  std={arr.std():.1%}")
    print(f"  min={arr.min():.1%}   max={arr.max():.1%}")
    print(f"  Transcripts with ratio outside 20–70%: "
          f"{((arr < 0.20) | (arr > 0.70)).sum()} / {len(arr)}")
    if 0.30 <= arr.mean() <= 0.65:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — mean ratio outside 30–65% range")


def check_speaker_roles(processed_dir):
    section("1C · Speaker Role Coverage")
    total = has_ceo = has_cfo = has_both = 0
    for f in sorted(processed_dir.glob("*.json")):
        with open(f) as fp:
            t = json.load(fp)
        total += 1
        roles = set(b["role"] for b in t["prepared_remarks"] + t["qa_section"])
        ceo = "CEO" in roles
        cfo = "CFO" in roles
        has_ceo += ceo
        has_cfo += cfo
        has_both += (ceo and cfo)

    print(f"  CEO detected:      {has_ceo}/{total} = {has_ceo/total:.0%}")
    print(f"  CFO detected:      {has_cfo}/{total} = {has_cfo/total:.0%}")
    print(f"  Both CEO+CFO:      {has_both}/{total} = {has_both/total:.0%}  (expect 70%+)")
    if has_both / total >= 0.50:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — less than 50% of transcripts have both CEO and CFO")


def check_fls_distribution(df):
    section("1D · FLS Density Distribution")
    fls = df["fls_count"]
    print(f"  mean={fls.mean():.1f}  median={fls.median():.0f}  "
          f"std={fls.std():.1f}  min={fls.min()}  max={fls.max()}")
    print(f"  Transcripts with 0 FLS:    {(fls == 0).sum()}")
    print(f"  Transcripts with 50+ FLS:  {(fls >= 50).sum()}")
    bins = [0, 5, 10, 20, 30, 50, 200]
    labels = ["0–5", "6–10", "11–20", "21–30", "31–50", "50+"]
    for i, label in enumerate(labels):
        count = ((fls >= bins[i]) & (fls < bins[i+1])).sum()
        bar = "█" * count
        print(f"    {label:6s}  {count:3d}  {bar}")
    if 10 <= fls.mean() <= 60 and (fls == 0).sum() < 5:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — distribution looks off")


def check_return_distribution(df):
    section("1E · Stock Return Distribution")
    for w in [1, 3, 7]:
        col = f"return_{w}d"
        r = df[col].dropna()
        print(f"  {w}-day:  mean={r.mean():.4f}  std={r.std():.4f}  "
              f"min={r.min():.4f}  max={r.max():.4f}  n={len(r)}")
    if abs(df["return_1d"].mean()) < 0.02 and df["return_1d"].std() < 0.12:
        print("  ✓ PASS — centered near 0, reasonable spread")
    else:
        print("  ✗ FAIL — possible date alignment issue")


# ─────────────────────────────────────────────────────────────
# LEVEL 2 — Spot-Check Individual Transcripts
# ─────────────────────────────────────────────────────────────

def spot_check_transcript(ticker, quarter, processed_dir, sent_df, fls_df):
    section(f"2 · Spot Check: {ticker} {quarter}")

    json_file = processed_dir / f"{ticker}_{quarter}.json"
    if not json_file.exists():
        print(f"  File not found: {json_file}")
        return

    with open(json_file) as f:
        t = json.load(f)

    # --- Speaker blocks
    print(f"\n  PREPARED REMARKS — {len(t['prepared_remarks'])} speaker blocks")
    for b in t["prepared_remarks"][:4]:
        print(f"    [{b['role']:10s}] {b['speaker']}")
        print(f"               \"{b['text'][:120].strip()}...\"")

    print(f"\n  Q&A SECTION — {len(t['qa_section'])} speaker blocks")
    for b in t["qa_section"][:4]:
        print(f"    [{b['role']:10s}] {b['speaker']}")
        print(f"               \"{b['text'][:120].strip()}...\"")

    # --- Top/bottom sentences from FinBERT
    tid = f"{ticker}_{quarter}"
    sents = sent_df[sent_df["transcript_id"] == tid].copy()
    if not sents.empty:
        print(f"\n  FINBERT SCORES — {len(sents)} sentences scored")
        top3 = sents.nlargest(3, "sentiment_score")
        bot3 = sents.nsmallest(3, "sentiment_score")
        print("\n  Top 3 most POSITIVE sentences:")
        for _, row in top3.iterrows():
            print(f"    score={row['sentiment_score']:+.3f} [{row['section']}/{row['role']}]")
            print(f"    \"{row['sentence'][:130]}\"")
        print("\n  Top 3 most NEGATIVE sentences:")
        for _, row in bot3.iterrows():
            print(f"    score={row['sentiment_score']:+.3f} [{row['section']}/{row['role']}]")
            print(f"    \"{row['sentence'][:130]}\"")
    else:
        print("  (no sentence scores found for this transcript)")

    # --- FLS detections
    fls_rows = fls_df[fls_df["transcript_id"] == tid]
    if not fls_rows.empty:
        print(f"\n  FLS DETECTIONS — {len(fls_rows)} statements")
        for _, row in fls_rows.head(6).iterrows():
            score = f"{row['sentiment_score']:+.3f}" if pd.notna(row.get("sentiment_score")) else "  N/A"
            print(f"    [{row['fls_type']:8s} score={score}] \"{row['sentence'][:110]}\"")
    else:
        print("  (no FLS found for this transcript)")


# ─────────────────────────────────────────────────────────────
# LEVEL 3 — Reproducibility Check
# ─────────────────────────────────────────────────────────────

def check_reproducibility():
    section("3 · Reproducibility Check (10-transcript subset)")
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    import spacy

    TICKER, QUARTER = "AAPL", "2022-Q3"
    json_file = PROCESSED / f"{TICKER}_{QUARTER}.json"
    with open(json_file) as f:
        t = json.load(f)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    sentences = []
    for block in t["prepared_remarks"][:2]:
        doc = nlp(block["text"])
        sentences += [s.text.strip() for s in doc.sents if len(s.text.strip()) > 15]
    sentences = sentences[:20]

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()

    def score(sents):
        inputs = tokenizer(sents, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        with torch.no_grad():
            probs = F.softmax(model(**inputs).logits, dim=-1).numpy()
        return probs[:, 0] - probs[:, 1]

    run1 = score(sentences)
    run2 = score(sentences)
    max_diff = abs(run1 - run2).max()
    print(f"  Scored {len(sentences)} sentences twice.")
    print(f"  Max score difference between runs: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("  ✓ PASS — outputs are deterministic")
    else:
        print("  ✗ FAIL — non-deterministic output detected")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df      = pd.read_csv(FEATURES / "nlp_features.csv")
    sent_df = pd.read_csv(FEATURES / "sentence_sentiments.csv")
    fls_df  = pd.read_csv(FEATURES / "fls_sentences.csv")

    # Level 1
    check_sentiment_distribution(df)
    check_section_ratio(PROCESSED)
    check_speaker_roles(PROCESSED)
    check_fls_distribution(df)
    check_return_distribution(df)

    # Level 2 — spot-check 3 transcripts
    spot_check_transcript("AAPL", "2022-Q3", PROCESSED, sent_df, fls_df)
    spot_check_transcript("META", "2022-Q3", PROCESSED, sent_df, fls_df)  # "year of efficiency"
    spot_check_transcript("NVDA", "2023-Q1", PROCESSED, sent_df, fls_df)  # AI boom call

    # Level 3
    check_reproducibility()

    print(f"\n{SEP}")
    print("  Validation complete.")
    print(SEP)
