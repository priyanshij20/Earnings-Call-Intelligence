"""
Parses raw Motley Fool earnings call transcripts into structured JSON.
Splits prepared remarks from Q&A, and identifies individual speakers with roles.
"""

import re
import json
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


REMARKS_HEADERS = re.compile(
    r'^(prepared\s+remarks?|opening\s+remarks?|presentation):?\s*$',
    re.IGNORECASE | re.MULTILINE
)
QA_HEADERS = re.compile(
    r'^(questions?\s*(?:&|and)\s*answers?|q\s*&\s*a|question[\-\s]and[\-\s]answer):?\s*$',
    re.IGNORECASE | re.MULTILINE
)

# Matches "Speaker Name -- Title" or "Speaker Name (Company)"
SPEAKER_LINE = re.compile(
    r'^([A-Z][A-Za-z\s\.\'\-]+?)\s*(?:--|—|–)\s*(.+)$|'
    r'^([A-Z][A-Za-z\s\.\'\-]+?)\s*\(([^)]+)\)\s*$'
)

# Strict C-suite matching: require explicit "Chief Executive" or standalone "CEO/CFO".
# "chairman", "president", "vice president" are deliberately excluded — too broad
# (IR directors, board members, and COOs frequently hold these titles).
CEO_TITLES = re.compile(r'\b(chief\s+executive\s+officer|chief\s+executive|\bceo\b)\b', re.IGNORECASE)
CFO_TITLES = re.compile(r'\b(chief\s+financial\s+officer|chief\s+financial|\bcfo\b)\b', re.IGNORECASE)
COO_TITLES = re.compile(r'\b(chief\s+operating\s+officer|chief\s+operating|\bcoo\b)\b', re.IGNORECASE)
IR_TITLES  = re.compile(r'\b(investor\s+relations|corporate\s+finance|ir\b)\b', re.IGNORECASE)
ANALYST_ROLES = re.compile(r'\b(analyst|research|capital|securities|partners|bank|llc|inc\.?)\b', re.IGNORECASE)
OPERATOR_ROLE = re.compile(r'^operator$', re.IGNORECASE)


def _classify_role(title: str) -> str:
    if not title:
        return "unknown"
    if OPERATOR_ROLE.match(title.strip()):
        return "operator"
    # IR must be checked before CEO/CFO — an IR director's title often contains
    # "finance" or other executive keywords that would trigger CFO matching.
    if IR_TITLES.search(title):
        return "IR"
    if CEO_TITLES.search(title):
        return "CEO"
    if CFO_TITLES.search(title):
        return "CFO"
    if COO_TITLES.search(title):
        return "COO"
    if ANALYST_ROLES.search(title):
        return "analyst"
    return "executive"


def _parse_speaker_blocks(text: str) -> list[dict]:
    """Split a section of transcript text into per-speaker blocks."""
    lines = text.strip().split('\n')
    blocks = []
    current_speaker = None
    current_role = None
    current_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = SPEAKER_LINE.match(line)
        if m:
            # Save previous block
            if current_speaker and current_lines:
                blocks.append({
                    "speaker": current_speaker,
                    "role": current_role,
                    "text": ' '.join(current_lines).strip()
                })
            # Start new block
            if m.group(1):  # "Name -- Title" format
                current_speaker = m.group(1).strip()
                current_role = _classify_role(m.group(2).strip())
            else:  # "Name (Company)" format
                current_speaker = m.group(3).strip()
                current_role = _classify_role(m.group(4).strip())
            current_lines = []
        else:
            current_lines.append(line)

    if current_speaker and current_lines:
        blocks.append({
            "speaker": current_speaker,
            "role": current_role,
            "text": ' '.join(current_lines).strip()
        })

    return blocks


def parse_transcript(row: dict) -> dict:
    """Parse one DataFrame row into a structured transcript dict."""
    text = row.get('transcript', '')
    ticker = row.get('ticker', '')
    date_str = row.get('date', '')
    quarter = row.get('q', '')
    exchange = row.get('exchange', '')

    # Parse date — strip timezone suffix ("ET", "p.m.", etc.), keep only "Month Day, Year"
    try:
        parts = [p.strip() for p in date_str.split(',')]
        date_str_short = f"{parts[0]}, {parts[1]}"
        for fmt in ('%b %d, %Y', '%B %d, %Y'):
            try:
                date_parsed = datetime.strptime(date_str_short, fmt).strftime('%Y-%m-%d')
                break
            except ValueError:
                continue
        else:
            date_parsed = date_str
    except Exception:
        date_parsed = date_str

    # Find section boundaries
    remarks_match = REMARKS_HEADERS.search(text)
    qa_match = QA_HEADERS.search(text)

    if remarks_match and qa_match:
        remarks_text = text[remarks_match.end():qa_match.start()]
        qa_text = text[qa_match.end():]
    elif remarks_match:
        remarks_text = text[remarks_match.end():]
        qa_text = ""
    elif qa_match:
        remarks_text = text[:qa_match.start()]
        qa_text = text[qa_match.end():]
    else:
        remarks_text = text
        qa_text = ""

    return {
        "ticker": ticker,
        "exchange": exchange,
        "date": date_parsed,
        "quarter": quarter,
        "prepared_remarks": _parse_speaker_blocks(remarks_text),
        "qa_section": _parse_speaker_blocks(qa_text),
        "raw_transcript": text
    }


def load_and_filter(pkl_path: str, tickers: list[str]) -> list[dict]:
    """Load the Kaggle pickle and return parsed transcripts for target tickers."""
    import pandas as pd

    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Total transcripts in dataset: {len(df)}")
    subset = df[df['ticker'].isin(tickers)].copy()
    print(f"Transcripts for target companies: {len(subset)}")

    parsed = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Parsing transcripts"):
        try:
            parsed.append(parse_transcript(row.to_dict()))
        except Exception as e:
            print(f"  Skipped {row.get('ticker')} {row.get('q')}: {e}")

    return parsed


def save_parsed(parsed: list[dict], output_dir: str):
    """Save each transcript as a JSON file named ticker_quarter.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for t in parsed:
        filename = f"{t['ticker']}_{t['quarter'].replace(' ', '_')}.json"
        with open(out / filename, 'w') as f:
            json.dump(t, f, indent=2)
    print(f"Saved {len(parsed)} transcripts to {output_dir}")


if __name__ == "__main__":
    import sys; sys.path.insert(0, '.')
    from src.config import load as load_cfg
    cfg = load_cfg()
    parsed = load_and_filter(cfg['paths']['raw_pkl'], cfg['tickers'])
    save_parsed(parsed, cfg['paths']['processed'])
