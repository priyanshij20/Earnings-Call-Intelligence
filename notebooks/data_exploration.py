"""
01_data_exploration.py — Quick-start data exploration
======================================================
Run this FIRST after downloading the Kaggle pickle file.
It tells you exactly what columns exist and what the data looks like,
so you can adjust the parser if the format is different.

Usage:
    python notebooks/01_data_exploration.py

Or paste these cells into a Jupyter notebook.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# ─── Step 1: Find and load the pickle file ───
pkl_files = list(RAW_DIR.glob("*.pkl")) + list(RAW_DIR.glob("*.pickle"))

if not pkl_files:
    print("❌ No pickle file found!")
    print(f"   Place the Kaggle download in: {RAW_DIR}/")
    print("   Download from: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts")
    sys.exit(1)

pkl_path = pkl_files[0]
print(f"📦 Loading: {pkl_path.name}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"   Type: {type(data)}")

# ─── Step 2: Convert to DataFrame ───
if isinstance(data, pd.DataFrame):
    df = data
elif isinstance(data, list):
    df = pd.DataFrame(data)
elif isinstance(data, dict):
    # Could be {ticker: [transcripts]} or {column: [values]}
    if all(isinstance(v, list) for v in data.values()):
        df = pd.DataFrame(data)
    else:
        print("   Dict structure:")
        for k, v in list(data.items())[:5]:
            print(f"     {k}: {type(v)}")
        sys.exit(0)
else:
    print(f"   Unexpected type: {type(data)}")
    sys.exit(1)

# ─── Step 3: Inspect the DataFrame ───
print(f"\n📊 DataFrame shape: {df.shape}")
print(f"\n📋 Columns ({len(df.columns)}):")
for col in df.columns:
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    sample = str(df[col].dropna().iloc[0])[:80] if non_null > 0 else "ALL NULL"
    print(f"   {col:30s} | {str(dtype):10s} | {non_null:6d} non-null | sample: {sample}")

# ─── Step 4: Look at a sample transcript ───
print("\n" + "=" * 80)
print("📝 SAMPLE TRANSCRIPT (first row)")
print("=" * 80)

row = df.iloc[0]
for col in df.columns:
    val = str(row[col])
    if len(val) > 200:
        print(f"\n--- {col} (first 500 chars) ---")
        print(val[:500])
        print(f"... [{len(val)} total chars]")
    else:
        print(f"   {col}: {val}")

# ─── Step 5: Check which target tickers exist ───
TARGET_TICKERS = ["AAPL", "MSFT", "META", "AMZN", "NVDA", "JPM", "GS", "BAC",
                  "JNJ", "PFE", "UNH", "WMT", "TGT", "COST", "XOM", "CVX"]

print("\n" + "=" * 80)
print("🎯 TARGET TICKER COVERAGE")
print("=" * 80)

# Try to find ticker column or extract from title
ticker_col = None
for col in ["ticker", "Ticker", "TICKER", "symbol", "Symbol"]:
    if col in df.columns:
        ticker_col = col
        break

if ticker_col:
    print(f"Found ticker column: '{ticker_col}'")
    for t in TARGET_TICKERS:
        count = (df[ticker_col].astype(str).str.upper() == t).sum()
        print(f"   {t}: {count} transcripts")
else:
    print("No direct ticker column found. Checking titles...")
    title_col = None
    for col in ["title", "Title", "heading", "Heading"]:
        if col in df.columns:
            title_col = col
            break

    if title_col:
        for t in TARGET_TICKERS:
            count = df[title_col].astype(str).str.contains(t, case=False, na=False).sum()
            print(f"   {t}: ~{count} transcripts (by title match)")
    else:
        print("   ⚠️  No title or ticker column found! Check columns above.")

# ─── Step 6: Check section markers in transcripts ───
print("\n" + "=" * 80)
print("🔍 SECTION MARKER DETECTION (sampling 20 transcripts)")
print("=" * 80)

text_col = None
for col in ["text", "content", "transcript", "body", "Text", "Content"]:
    if col in df.columns:
        text_col = col
        break

if text_col:
    sample = df[text_col].dropna().head(20)
    has_prepared = sum(1 for t in sample if "Prepared Remarks" in str(t) or "prepared remarks" in str(t))
    has_qa = sum(1 for t in sample if "Questions and Answers" in str(t) or "Question-and-Answer" in str(t))
    has_dashes = sum(1 for t in sample if " -- " in str(t) or " — " in str(t))

    print(f"   'Prepared Remarks' found in: {has_prepared}/20 sampled")
    print(f"   'Questions and Answers' found in: {has_qa}/20 sampled")
    print(f"   Speaker dashes ('--' or '—') found in: {has_dashes}/20 sampled")

    # Show the section headers found
    print(f"\n   Section headers found in first transcript:")
    first_text = str(sample.iloc[0])
    for line in first_text.split("\n"):
        stripped = line.strip()
        if any(kw in stripped.lower() for kw in
               ["prepared remarks", "questions and answers", "q&a", "question-and-answer",
                "call participants", "operator"]):
            print(f"      → '{stripped}'")
else:
    print("   ⚠️  No text column found! Check columns above.")

print("\n✅ Exploration complete! Use the column names above to adjust the parser if needed.")
print("   Next step: python -m src.ingestion.transcript_parser")
