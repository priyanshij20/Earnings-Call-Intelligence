"""
Loughran-McDonald Financial Dictionary scoring.
Downloads the master dictionary CSV and scores transcripts across 6 dimensions:
positive, negative, uncertainty, litigious, constraining, superfluous.
Scores are normalized word proportions, computed for overall, remarks, and Q&A.
"""

import re
import json
import yaml
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

LM_GDRIVE_ID = "1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF"
LM_LOCAL = "data/raw/lm_dictionary.csv"

# 2024 LM dictionary columns (Superfluous removed, Strong_Modal/Weak_Modal added)
CATEGORIES = {
    "lm_positive":     "Positive",
    "lm_negative":     "Negative",
    "lm_uncertainty":  "Uncertainty",
    "lm_litigious":    "Litigious",
    "lm_constraining": "Constraining",
    "lm_strong_modal": "Strong_Modal",
    "lm_weak_modal":   "Weak_Modal",
}

WORD_RE = re.compile(r"[a-z]+")


def load_lm_wordlists(csv_path: str) -> dict[str, set]:
    """Load LM dictionary; download if missing. Returns {category: set_of_words}."""
    path = Path(csv_path)
    if not path.exists():
        print("Downloading LM dictionary from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={LM_GDRIVE_ID}"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        print(f"Saved to {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    wordlists = {}
    for feat, col in CATEGORIES.items():
        if col in df.columns:
            # LM uses non-zero values to mark a word as belonging to that category
            words = df.loc[df[col] != 0, "Word"].str.lower().tolist()
            wordlists[feat] = set(words)
        else:
            print(f"  Warning: column '{col}' not found in LM dictionary")
            wordlists[feat] = set()

    total = sum(len(v) for v in wordlists.values())
    print(f"LM dictionary loaded: {total} word-category memberships")
    return wordlists


def score_text(text: str, wordlists: dict[str, set]) -> dict:
    """Score a text string against all LM categories. Returns proportions."""
    tokens = WORD_RE.findall(text.lower())
    total = len(tokens)
    if total == 0:
        return {feat: 0.0 for feat in wordlists}
    return {feat: sum(1 for t in tokens if t in words) / total
            for feat, words in wordlists.items()}


def blocks_to_text(blocks: list[dict]) -> str:
    return " ".join(b.get("text", "") for b in blocks)


def run(cfg: dict) -> pd.DataFrame:
    processed_dir = Path(cfg["paths"]["processed"])
    features_dir  = Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    wordlists = load_lm_wordlists(LM_LOCAL)

    records = []
    transcript_files = sorted(processed_dir.glob("*.json"))
    print(f"Scoring {len(transcript_files)} transcripts with LM dictionary...")

    for json_file in tqdm(transcript_files, desc="LM Dictionary"):
        with open(json_file) as f:
            t = json.load(f)

        remarks_text = blocks_to_text(t.get("prepared_remarks", []))
        qa_text      = blocks_to_text(t.get("qa_section", []))
        full_text    = remarks_text + " " + qa_text

        overall  = score_text(full_text, wordlists)
        remarks  = score_text(remarks_text, wordlists)
        qa       = score_text(qa_text, wordlists)

        row = {
            "transcript_id": json_file.stem,
            "ticker":  t["ticker"],
            "quarter": t["quarter"],
            "date":    t["date"],
        }
        for feat, val in overall.items():
            row[feat] = val
        for feat, val in remarks.items():
            row[f"{feat}_remarks"] = val
        for feat, val in qa.items():
            row[f"{feat}_qa"] = val

        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(features_dir / "lm_features.csv", index=False)
    print(f"Saved lm_features.csv ({len(df)} rows)")
    print(df[["ticker", "quarter", "lm_negative", "lm_uncertainty", "lm_positive"]].head(8))
    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from src.config import load as load_cfg
    run(load_cfg())
