"""
Forward-Looking Statement (FLS) detector.
Uses rule-based phrase matching (spaCy PhraseMatcher) to identify FLS sentences,
then classifies each as positive/cautious/hedging using pre-computed FinBERT scores.
"""

import re
import json
import yaml
import spacy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# SEC-style forward-looking phrases (positive/growth-oriented)
POSITIVE_FLS_PHRASES = [
    "we expect", "we anticipate", "we believe", "we project", "we forecast",
    "our outlook", "our guidance", "going forward", "looking ahead", "in the coming",
    "we plan to", "we intend to", "we are confident", "we remain confident",
    "we remain optimistic", "we're optimistic", "we see opportunities",
    "we are positioned", "we're positioned", "on track to", "we target",
    "we are committed to", "we continue to expect", "we will continue",
    "we are excited", "strong momentum", "we're on track", "we see strong",
    "we're well positioned", "we look forward", "we are pleased to", "we expect to",
    "expected to grow", "expected to increase", "expected to deliver",
]

# Cautious / hedging forward-looking phrases
HEDGING_FLS_PHRASES = [
    "subject to", "there can be no assurance", "may not", "might not",
    "risks include", "risk factors", "uncertainties", "we caution",
    "no guarantee", "could adversely", "may adversely", "could impact",
    "subject to change", "we cannot predict", "difficult to predict",
    "challenging environment", "headwinds", "uncertainty remains",
    "we are cautious", "cautiously optimistic", "remain cautious",
    "we're working through", "navigating challenges", "facing headwinds",
    "macro uncertainty", "market uncertainty",
]

ALL_FLS_PHRASES = POSITIVE_FLS_PHRASES + HEDGING_FLS_PHRASES
POSITIVE_SET = set(POSITIVE_FLS_PHRASES)
HEDGING_SET  = set(HEDGING_FLS_PHRASES)


def build_matcher(nlp):
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    pos_patterns = [nlp.make_doc(p) for p in POSITIVE_FLS_PHRASES]
    hed_patterns = [nlp.make_doc(p) for p in HEDGING_FLS_PHRASES]
    matcher.add("FLS_POSITIVE", pos_patterns)
    matcher.add("FLS_HEDGING",  hed_patterns)
    return matcher


# Safe harbor boilerplate phrases — identical legal text that appears in every call.
# These trigger FLS phrase matching but carry zero signal, so we filter them out.
# Keeping them would add a ~constant offset to every transcript's hedging count,
# which doesn't affect relative correlations but pollutes the fls_sentences detail file.
SAFE_HARBOR_RE = re.compile(
    r'(forward.looking\s+statements?|actual\s+results\s+(may|could|might)\s+differ|'
    r'safe\s+harbor|private\s+securities\s+litigation|risks?\s+and\s+uncertainties|'
    r'no\s+obligation\s+to\s+update|risk\s+factors)',
    re.IGNORECASE
)


def detect_fls_in_text(text: str, nlp, matcher) -> list[dict]:
    """Return list of sentences that contain at least one FLS phrase."""
    doc = nlp(text)
    fls_sents = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        # Skip legal boilerplate — it matches FLS phrases but is the same in every call
        if SAFE_HARBOR_RE.search(sentence):
            continue
        sent_doc = nlp.make_doc(sentence)
        matches = matcher(sent_doc)
        if matches:
            matched_labels = {nlp.vocab.strings[match_id] for match_id, _, _ in matches}
            is_hedging = "FLS_HEDGING" in matched_labels
            is_positive = "FLS_POSITIVE" in matched_labels
            fls_type = "hedging" if (is_hedging and not is_positive) else "positive"
            fls_sents.append({
                "sentence": sentence,
                "fls_type": fls_type,
            })
    return fls_sents


def merge_finbert_scores(fls_sentences: list[dict], sent_df: pd.DataFrame,
                         transcript_id: str, section: str) -> list[dict]:
    """Attach FinBERT scores to FLS sentences via fuzzy sentence matching."""
    if sent_df.empty:
        return fls_sentences

    subset = sent_df[
        (sent_df["transcript_id"] == transcript_id) &
        (sent_df["section"] == section)
    ][["sentence", "sentiment_score"]].set_index("sentence")

    for row in fls_sentences:
        s = row["sentence"]
        if s in subset.index:
            row["sentiment_score"] = subset.loc[s, "sentiment_score"]
        else:
            row["sentiment_score"] = None
    return fls_sentences


def run(cfg: dict) -> pd.DataFrame:
    processed_dir = Path(cfg["paths"]["processed"])
    features_dir  = Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    matcher = build_matcher(nlp)

    # Load pre-computed sentence sentiments for score merging
    sent_csv = features_dir / "sentence_sentiments.csv"
    if sent_csv.exists():
        sent_df = pd.read_csv(sent_csv)
    else:
        print("Warning: sentence_sentiments.csv not found — run finbert_sentiment first for score merging")
        sent_df = pd.DataFrame()

    records = []
    fls_detail_rows = []

    transcript_files = sorted(processed_dir.glob("*.json"))
    print(f"Detecting FLS in {len(transcript_files)} transcripts...")

    for json_file in tqdm(transcript_files, desc="FLS Detector"):
        with open(json_file) as f:
            t = json.load(f)

        transcript_id = json_file.stem

        def extract_section_text(blocks):
            return " ".join(b.get("text", "") for b in blocks)

        remarks_text = extract_section_text(t.get("prepared_remarks", []))
        qa_text      = extract_section_text(t.get("qa_section", []))

        remarks_fls = detect_fls_in_text(remarks_text, nlp, matcher)
        qa_fls      = detect_fls_in_text(qa_text, nlp, matcher)

        if not sent_df.empty:
            remarks_fls = merge_finbert_scores(remarks_fls, sent_df, transcript_id, "remarks")
            qa_fls      = merge_finbert_scores(qa_fls, sent_df, transcript_id, "qa")

        all_fls = remarks_fls + qa_fls
        total_sentences = (
            sum(1 for _ in nlp(remarks_text).sents) +
            sum(1 for _ in nlp(qa_text).sents)
        )

        scores = [r["sentiment_score"] for r in all_fls if r.get("sentiment_score") is not None]

        row = {
            "transcript_id":       transcript_id,
            "ticker":              t["ticker"],
            "quarter":             t["quarter"],
            "date":                t["date"],
            "fls_count":           len(all_fls),
            "fls_remarks_count":   len(remarks_fls),
            "fls_qa_count":        len(qa_fls),
            "fls_positive_count":  sum(1 for r in all_fls if r["fls_type"] == "positive"),
            "fls_hedging_count":   sum(1 for r in all_fls if r["fls_type"] == "hedging"),
            "fls_ratio":           len(all_fls) / total_sentences if total_sentences > 0 else 0,
            "fls_sentiment_avg":   sum(scores) / len(scores) if scores else None,
        }
        records.append(row)

        for section, fls_list in [("remarks", remarks_fls), ("qa", qa_fls)]:
            for item in fls_list:
                fls_detail_rows.append({
                    "transcript_id": transcript_id,
                    "ticker": t["ticker"],
                    "quarter": t["quarter"],
                    "section": section,
                    "fls_type": item["fls_type"],
                    "sentence": item["sentence"],
                    "sentiment_score": item.get("sentiment_score"),
                })

    df = pd.DataFrame(records)
    df.to_csv(features_dir / "fls_features.csv", index=False)
    print(f"Saved fls_features.csv ({len(df)} rows)")

    detail_df = pd.DataFrame(fls_detail_rows)
    detail_df.to_csv(features_dir / "fls_sentences.csv", index=False)
    print(f"Saved fls_sentences.csv ({len(detail_df)} rows)")

    print(df[["ticker", "quarter", "fls_count", "fls_positive_count",
              "fls_hedging_count", "fls_ratio"]].head(10))
    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from src.config import load as load_cfg
    run(load_cfg())
