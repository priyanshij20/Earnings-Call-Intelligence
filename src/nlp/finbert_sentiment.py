"""
Sentence-level FinBERT sentiment scoring for earnings call transcripts.
Outputs per-transcript aggregated features and a sentence-level detail CSV.
"""

import json
import yaml
import spacy
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


MAX_TOKENS = 512
BATCH_SIZE = 32


def load_finbert(model_name: str = "ProsusAI/finbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"FinBERT loaded on {device}")
    return tokenizer, model, device


def score_sentences(sentences: list[str], tokenizer, model, device) -> list[dict]:
    """Score a list of sentences in batches. Returns list of {pos, neg, neu, score}."""
    results = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        # ProsusAI/finbert label order: positive=0, negative=1, neutral=2
        for p in probs:
            results.append({
                "positive_prob": float(p[0]),
                "negative_prob": float(p[1]),
                "neutral_prob":  float(p[2]),
                "sentiment_score": float(p[0] - p[1])
            })
    return results


def extract_sentences(blocks: list[dict], nlp) -> list[dict]:
    """Run spaCy sentence splitter over each speaker block, return flat list."""
    rows = []
    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            continue
        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if len(s) > 15:  # skip very short fragments
                rows.append({
                    "speaker": block.get("speaker", ""),
                    "role": block.get("role", ""),
                    "sentence": s
                })
    return rows


def aggregate_scores(sentence_rows: list[dict]) -> dict:
    """Compute per-transcript aggregated FinBERT features."""
    if not sentence_rows:
        return {}

    scores = [r["sentiment_score"] for r in sentence_rows]
    remarks = [r["sentiment_score"] for r in sentence_rows if r.get("section") == "remarks"]
    qa      = [r["sentiment_score"] for r in sentence_rows if r.get("section") == "qa"]
    pos_probs = [r["positive_prob"] for r in sentence_rows]
    neg_probs = [r["negative_prob"] for r in sentence_rows]
    neu_probs = [r["neutral_prob"]  for r in sentence_rows]

    def safe_mean(lst): return sum(lst) / len(lst) if lst else None
    def safe_std(lst):
        if len(lst) < 2: return None
        m = safe_mean(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

    feats = {
        "sentiment_overall":    safe_mean(scores),
        "sentiment_remarks":    safe_mean(remarks),
        "sentiment_qa":         safe_mean(qa),
        "sentiment_divergence": (safe_mean(remarks) - safe_mean(qa))
                                if (remarks and qa) else None,
        "positive_ratio":       sum(1 for p in pos_probs if p > 0.5) / len(pos_probs),
        "negative_ratio":       sum(1 for p in neg_probs if p > 0.5) / len(neg_probs),
        "neutral_ratio":        sum(1 for p in neu_probs if p > 0.5) / len(neu_probs),
        "sentiment_std":        safe_std(scores),
        "sentence_count":       len(scores),
    }
    return feats


def run(cfg: dict):
    processed_dir = Path(cfg["paths"]["processed"])
    features_dir  = Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")

    tokenizer, model, device = load_finbert(cfg["models"].get("finbert_sentiment", "ProsusAI/finbert"))

    all_features = []
    all_sentences = []

    transcript_files = sorted(processed_dir.glob("*.json"))
    print(f"Scoring {len(transcript_files)} transcripts...")

    for json_file in tqdm(transcript_files, desc="FinBERT"):
        with open(json_file) as f:
            t = json.load(f)

        transcript_id = json_file.stem

        # Extract sentences with section labels
        remarks_sents = extract_sentences(t.get("prepared_remarks", []), nlp)
        for r in remarks_sents:
            r["section"] = "remarks"
        qa_sents = extract_sentences(t.get("qa_section", []), nlp)
        for r in qa_sents:
            r["section"] = "qa"

        all_sents = remarks_sents + qa_sents
        if not all_sents:
            continue

        texts = [r["sentence"] for r in all_sents]
        scores = score_sentences(texts, tokenizer, model, device)

        for row, score in zip(all_sents, scores):
            row.update(score)
            row["transcript_id"] = transcript_id
            row["ticker"] = t["ticker"]
            row["quarter"] = t["quarter"]
            all_sentences.append(row)

        feats = aggregate_scores(all_sents)
        feats["transcript_id"] = transcript_id
        feats["ticker"] = t["ticker"]
        feats["quarter"] = t["quarter"]
        feats["date"] = t["date"]
        all_features.append(feats)

    # Save sentence-level detail
    sent_df = pd.DataFrame(all_sentences)[[
        "transcript_id", "ticker", "quarter", "section", "speaker", "role",
        "sentence", "positive_prob", "negative_prob", "neutral_prob", "sentiment_score"
    ]]
    sent_df.to_csv(features_dir / "sentence_sentiments.csv", index=False)
    print(f"Saved sentence_sentiments.csv ({len(sent_df)} rows)")

    # Save transcript-level aggregates
    feat_df = pd.DataFrame(all_features)
    feat_df.to_csv(features_dir / "finbert_features.csv", index=False)
    print(f"Saved finbert_features.csv ({len(feat_df)} rows)")
    print(feat_df[["ticker", "quarter", "sentiment_overall", "sentiment_divergence"]].head(8))

    return feat_df


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from src.config import load as load_cfg
    run(load_cfg())
