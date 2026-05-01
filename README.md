# Earnings Call Intelligence Engine

**Can the language executives use on earnings calls predict how their stock moves in the days that follow?**

This project processes 137 earnings call transcripts from 16 S&P 500 companies (2019–2023), extracts NLP signals using FinBERT and the Loughran-McDonald Financial Dictionary, and correlates those signals with 1-, 3-, and 7-day post-earnings stock returns.

**Results at a glance:** 137 transcripts parsed, 68,909 sentences scored, 3,089 forward-looking statements detected, 37 narrative shift events flagged across 16 companies and 5 sectors.

---

## Key Findings

1. **Uncertainty language in prepared remarks shows the strongest predictive signal (Spearman ρ = −0.166, p = 0.053).** When executives hedge more in their scripted remarks, stocks tend to underperform the following day. The signal is stronger in prepared remarks (ρ = −0.166) than in Q&A (where the equivalent LM dictionary feature is weaker), consistent with the idea that scripted language is more deliberate and therefore more informative.

2. **Forward-looking statement density and hedging count both negatively correlate with 1-day returns (ρ = −0.137 and −0.113).** More FLS language is not bullish — it is associated with weaker post-earnings performance, likely because high FLS density reflects management needing to explain away a difficult quarter rather than simply reporting strong results.

3. **The remarks-vs-Q&A sentiment divergence, while structurally present in 93% of calls, shows a directionally correct within-company signal (ρ = −0.114 after company demeaning).** Executives uniformly script more positive prepared remarks than their Q&A tone suggests — this is a constant, not a quarterly signal. The within-company variation (quarters where the gap widens relative to a company's own baseline) does trend in the expected direction, but the effect is underpowered at n = 137.

4. **The language shift detector identified verifiable narrative changes:** Target's uncertainty spike preceded the 2022 inventory crisis (stock −4.8% in the 3 days following); AMZN's 2021-Q4 call registered as a positive shift event (+16.3% 3-day return) when AWS re-accelerated, and its 2022-Q1 call reversed sharply (−14.1%) as post-pandemic deceleration set in. MSFT's 2023-Q1 flagged a sentiment rise driven by AI integration optimism, but the stock fell −5.9% as the market priced in Azure growth deceleration instead. 31% of quarter-over-quarter comparisons met the shift threshold (> 1.5σ on any of 3 features).

5. **No features survive Bonferroni correction at n = 137, consistent with the efficient market hypothesis for large-cap equities.** With 72 simultaneous tests (24 NLP features × 3 return windows), the corrected α = 0.00069. The signals detected are real directional tendencies, not arbitrageable edges — which is the expected result for heavily-covered S&P 500 names where earnings surprises are priced within minutes.

---

## Limitations

**FinBERT anchor bias.** FinBERT was pretrained on financial news headlines, not earnings call transcripts. This creates systematic misclassifications for hedged growth language. For example, the sentence *"we anticipate headcount growth will slow"* is classified as **positive** by FinBERT because it anchors on the modal verb "anticipate" and the word "growth," missing the negative substance of slowing headcount. This pattern is pervasive in earnings calls, where positive framing surrounds negative guidance. The bias cannot be corrected by threshold tuning; it reflects a training distribution mismatch. A model fine-tuned on earnings call data (e.g., a FinBERT variant with SEC 10-Q supervision) would likely improve sentence-level accuracy meaningfully.

**n = 137 and survivorship bias.** The dataset covers 16 large-cap companies over 4–5 years. Large-cap equities are among the most efficiently priced assets in the world, making NLP-based return prediction especially difficult. The sample is also survivorship-biased: all 16 companies remained prominent S&P 500 constituents throughout the period. A dataset including companies that faced distress or delisted would likely show stronger NLP-return correlations.

**Returns are not risk-adjusted.** Post-earnings returns are raw price changes, not excess returns over the market or sector. On days with broad market moves, the NLP signal is swamped by macro noise. Controlling for SPY or sector ETF returns on the same window would likely sharpen all correlations.

**Sentiment divergence is structural, not temporal.** The remarks-vs-Q&A gap (prepared remarks are more positive than Q&A) is a property of earnings call format, not a meaningful quarter-over-quarter signal in raw form. The within-company demeaned version shows the correct directional effect but is underpowered. This is noted to avoid interpreting the aggregate divergence distribution as predictive.

---

## Methodology

### Data

- **Transcripts**: 137 earnings call transcripts (2019–2023) for 16 S&P 500 companies across 5 sectors (tech, finance, healthcare, retail, energy), sourced from a Motley Fool Kaggle dataset.
- **Prices**: Daily OHLCV data via `yfinance`. Post-earnings returns computed for 1-, 3-, and 7-day windows starting the day after the call.

### NLP Pipeline

**Stage 1 — Parsing**  
Each transcript is parsed into two structured sections: `prepared_remarks` (scripted executive statements) and `qa_section` (analyst questions + executive answers). Each block is tagged with speaker name, title, and role (CEO, CFO, analyst, operator, IR). The parser uses a multi-strategy role classifier: exact title match → acronym fallback → IR/relations intercept before CEO/CFO to avoid false positives from titles like "Head of Investor Relations."

**Stage 2 — FinBERT Sentence Scoring**  
Every sentence is scored using [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT model pretrained on financial news and fine-tuned for positive/negative/neutral classification. Scores are mapped to [−1, +1] via `P(positive) − P(negative)`. Processing: batch size 32, MPS acceleration (Apple Silicon), `model.eval()` + `torch.no_grad()` for deterministic output. 68,909 sentences scored across 137 transcripts.

**Stage 3 — Loughran-McDonald Dictionary**  
Word-proportion scores computed for 7 categories: positive, negative, uncertainty, litigious, constraining, strong_modal, weak_modal. Dictionary sourced from the 2024 SRAF release (7,000+ words). Scores computed separately for prepared remarks and Q&A sections.

**Stage 4 — Forward-Looking Statement Detection**  
~65 curated FLS phrases split into positive-oriented ("we expect", "looking ahead", "we are confident") and hedging-oriented ("subject to", "headwinds", "macro uncertainty") sets. Matching uses spaCy PhraseMatcher on lowercased text. Safe harbor legal boilerplate (identical across every call) is filtered via regex before matching to avoid constant-offset inflation.

**Stage 5 — Feature Matrix**  
Features from all three NLP stages are merged with stock returns into a 137-row × 45-column feature matrix (`nlp_features.csv`).

### Analytics

- **Spearman correlation matrix**: 72 tests with Bonferroni correction. Non-parametric to handle non-normal return distributions.
- **Divergence quartile test**: Mann-Whitney U comparing Q1 (low divergence) vs Q4 (high divergence) returns.
- **Sector-level correlations**: Separate Spearman analysis per sector for 4 focus features.
- **Temporal split**: Pre- vs post-2022 correlations to check stability across market regimes.
- **Language shift detector**: Per-company QoQ z-scores on 3 features; events flagged at > 1.5σ.

### Validation

Pipeline validation includes sentiment distribution checks, section word-count ratios, speaker role coverage verification (76% of transcripts have both CEO and CFO detected), FLS density analysis, return distribution checks, spot-checks on 3 individual transcripts (AAPL 2022-Q3, META 2022-Q2, NVDA 2023-Q1), and a deterministic reproducibility test confirming zero score delta across repeated FinBERT runs.

---

## How to Run

### Setup

```bash
git clone <repo>
cd earnings-call-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place the Motley Fool transcript data (`motley-fool-data.pkl`) in `data/raw/`.

### Run the full pipeline

```bash
# 1. Parse transcripts → data/processed/
python src/ingestion/transcript_parser.py

# 2. Fetch stock prices → data/market/
python src/ingestion/price_fetcher.py

# 3. FinBERT sentence scoring → data/features/sentence_sentiments.csv
python src/nlp/finbert_sentiment.py              # ~11 min on MPS

# 4. Loughran-McDonald dictionary → data/features/lm_features.csv
python src/nlp/lm_dictionary.py

# 5. FLS detection → data/features/fls_features.csv
python src/nlp/fls_detector.py

# 6. Build feature matrix → data/features/nlp_features.csv
python src/nlp/build_features.py

# 7. Correlation analysis → data/features/correlation_results.csv
python src/analytics/correlation.py

# 8. Language shift detection → data/features/language_shifts.csv
python src/analytics/language_shift.py
```

### Validate the pipeline

```bash
python src/validate.py
```

### Launch the dashboard

```bash
streamlit run src/app/streamlit_app.py
```

The dashboard has three views:
- **Signal Discovery** — correlation heatmap, scatter plots for the two strongest signals, interpretation cards
- **Company Narrative Timeline** — per-company NLP trends with shift event overlays and dual-axis stock price
- **Transcript Explainability** — sentence-level drill-down into any individual call, with FinBERT scores and FLS attribution

---

## Project Structure

```
earnings-call-intelligence/
├── config/
│   └── settings.yaml              # tickers, paths, model config
├── data/
│   ├── raw/                       # Kaggle pickle file
│   ├── processed/                 # parsed transcript JSONs (139 files)
│   ├── market/                    # yfinance price CSVs (16 files)
│   └── features/                  # all CSV outputs + findings_summary.json
├── src/
│   ├── config.py                  # settings.yaml normalizer
│   ├── validate.py                # 3-level validation suite
│   ├── ingestion/
│   │   ├── transcript_parser.py   # raw pickle → structured JSON
│   │   └── price_fetcher.py       # yfinance download + return calc
│   ├── nlp/
│   │   ├── finbert_sentiment.py   # sentence-level FinBERT scoring
│   │   ├── lm_dictionary.py       # Loughran-McDonald word scoring
│   │   ├── fls_detector.py        # forward-looking statement detection
│   │   └── build_features.py      # merge all features into matrix
│   ├── analytics/
│   │   ├── correlation.py         # Spearman, Mann-Whitney, sector/temporal
│   │   └── language_shift.py      # QoQ narrative shift detector
│   └── app/
│       └── streamlit_app.py       # interactive dashboard
├── requirements.txt
└── .gitignore
```
