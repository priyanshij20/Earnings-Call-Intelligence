# Earnings Call Intelligence Engine

**Can the language executives use on earnings calls predict how their stock moves — and can we cluster companies by their linguistic fingerprint?**

This project processes 137 earnings call transcripts from 16 S&P 500 companies (2019–2023), extracts NLP signals using FinBERT and the Loughran-McDonald Financial Dictionary, correlates those signals with post-earnings stock returns, and applies unsupervised clustering to discover communication archetypes across companies and quarters.

**Results at a glance:** 137 transcripts · 68,909 sentences scored · 3,089 forward-looking statements · 37 narrative shift events · 4 linguistic archetypes discovered

---

## Architecture

```
Raw Transcripts (137 calls, 16 S&P 500 companies, 5 sectors)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  NLP PIPELINE                                           │
│                                                         │
│  1. Transcript Parser   → prepared_remarks / qa_section │
│  2. FinBERT Scoring     → 68,909 sentence-level scores  │
│  3. LM Dictionary       → 7 linguistic dimensions       │
│  4. FLS Detector        → 3,089 forward-looking stmts   │
│  5. Feature Matrix      → 137 × 45 feature table        │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
┌──────────────────┐  ┌────────────────────────────────────┐
│  ANALYTICS       │  │  CLUSTERING PIPELINE               │
│                  │  │                                    │
│  Spearman ρ      │  │  Feature Engineering (18 features) │
│  Bonferroni      │  │  → QoQ momentum, role variance,    │
│  Language shifts │  │    complexity, modal confidence    │
│  Sector splits   │  │                                    │
└──────────────────┘  │  Preprocessing                     │
                      │  → StandardScaler, correlation     │
                      │    check, PCA variance report      │
                      │                                    │
                      │  Dimensionality Reduction          │
                      │  → PCA (2D/3D) · t-SNE (2D)       │
                      │                                    │
                      │  Clustering (3 methods compared)   │
                      │  → K-Means · DBSCAN · Hierarchical │
                      │                                    │
                      │  Interpretation                    │
                      │  → Archetypes · Temporal migration │
                      │    Kruskal-Wallis validation       │
                      └────────────────┬───────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────┐
                         │  STREAMLIT DASHBOARD    │
                         │                         │
                         │  Signal Discovery       │
                         │  Narrative Timeline     │
                         │  Transcript Explainer   │
                         │  Cluster Explorer  ◄NEW │
                         └─────────────────────────┘
```

---

## Key Findings

### NLP → Return Signals

| Feature | Window | Spearman ρ | p-value | Interpretation |
|---------|--------|-----------|---------|----------------|
| `lm_uncertainty_remarks` | 1d | −0.166 | 0.053 | More hedging in scripted remarks → underperformance |
| `fls_hedging_count` | 1d | −0.113 | 0.190 | More hedging FLS → weaker returns |
| `fls_ratio` | 1d | −0.137 | 0.113 | FLS density is not bullish signal |
| `sentiment_divergence` | 1d | −0.114 | 0.340 | Structural, not a quarterly signal |

**No features survive Bonferroni correction (corrected α = 0.00069 for 72 tests) — consistent with the efficient market hypothesis for large-cap equities.**

### Linguistic Clustering

| Method | Optimal k | Silhouette | Notes |
|--------|-----------|-----------|-------|
| K-Means | 3–4 | ~0.15 | Best balance of separation and interpretability |
| Hierarchical (Ward) | 3–4 | ~0.12 | Consistent with K-Means groupings |
| DBSCAN | — | — | ~2% outlier calls; one dominant cluster |

**The low silhouette scores are themselves a finding:** large-cap earnings calls are linguistically homogeneous. S&P 500 executives converge on similar communication styles. The interesting signal is *which companies and quarters deviate from archetype*, not the tightness of the clusters.

### Archetype Profiles

| Archetype | Signature | Example Companies |
|-----------|-----------|-------------------|
| **Confident Optimists** | High sentiment, strong modal language, low uncertainty | NVDA AI quarters, AAPL product cycles |
| **Cautious Hedgers** | High uncertainty + hedging FLS, low modal confidence | TGT 2022 inventory crisis, PFE post-vaccine |
| **Measured Reporters** | Near-average on all dimensions; consistent, flat tone | JPM, BAC routine quarters |
| **Forward-Looking Visionaries** | High FLS ratio, positive-oriented forward guidance | MSFT AI pivot quarters, AMZN AWS quarters |

### Notable Case Studies

- **AMZN 2021-Q4 → 2022-Q1**: Language shifted from "Confident Optimist" to "Cautious Hedger" exactly when AWS re-acceleration reversed. 3-day returns: +16.3% → −14.1%.
- **TGT 2022-Q2**: Uncertainty spike in prepared remarks preceded inventory write-down. Stock −4.8% in 3 days.
- **META 2022-Q2**: FinBERT anchor bias visible — "headcount growth will slow" scored positive due to anchoring on "growth."
- **NVDA 2023**: Consistent "Confident Optimist" archetype throughout AI boom quarters.

---

## Limitations

**FinBERT anchor bias.** FinBERT was pretrained on financial news headlines, not earnings call transcripts. Sentences like *"we anticipate headcount growth will slow"* are classified positive (anchors on "growth"), missing the negative substance. A model fine-tuned on 10-Q/earnings data would materially improve sentence-level accuracy.

**n = 137 and survivorship bias.** Covers 16 large-cap companies over 4–5 years — all of which remained prominent S&P 500 constituents. Distressed or delisted companies would likely show stronger NLP-return correlations.

**Returns not risk-adjusted.** Raw price changes, not excess returns over market/sector. Controlling for SPY or sector ETF returns on the same window would sharpen correlations.

**Weak cluster structure.** Silhouette scores of 0.10–0.15 reflect genuine linguistic homogeneity among large-cap executives, not a modeling failure. The archetypes are best interpreted as communication tendency distributions, not hard categories.

---

## Methodology

### Data

- **Transcripts**: 137 earnings call transcripts (2019–2023) for 16 S&P 500 companies across 5 sectors (tech, finance, healthcare, retail, energy), sourced from the Motley Fool Kaggle dataset.
- **Prices**: Daily OHLCV data via `yfinance`. Post-earnings returns computed for 1-, 3-, and 7-day windows starting the day after the call.

### NLP Pipeline

**Stage 1 — Parsing**
Each transcript is parsed into `prepared_remarks` and `qa_section`, with each block tagged with speaker name, title, and role (CEO, CFO, analyst, operator, IR). Role classifier uses exact title match → acronym fallback → IR intercept before CEO/CFO to avoid false positives.

**Stage 2 — FinBERT Sentence Scoring**
Every sentence scored via [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert). Score = `P(positive) − P(negative)` ∈ [−1, +1]. Batch size 32, MPS acceleration (Apple Silicon). 68,909 sentences scored.

**Stage 3 — Loughran-McDonald Dictionary**
Word-proportion scores for 7 categories: positive, negative, uncertainty, litigious, constraining, strong_modal, weak_modal. Computed separately for remarks and Q&A. Source: 2024 SRAF release (7,000+ words).

**Stage 4 — Forward-Looking Statement Detection**
~65 curated FLS phrases split into positive-oriented and hedging-oriented sets. spaCy PhraseMatcher on lowercased text. Safe-harbor boilerplate filtered via regex before matching.

**Stage 5 — Feature Matrix**
All NLP features merged with stock returns into a 137 × 45 feature matrix (`nlp_features.csv`).

### Clustering Pipeline

**Feature Engineering (18 features)**

| Category | Features |
|----------|---------|
| Core NLP | `sentiment_overall`, `sentiment_divergence`, `sentiment_std`, `lm_uncertainty`, `lm_litigious`, `lm_constraining`, `fls_ratio`, `fls_hedging_count` |
| Structural | `modal_confidence` (strong/weak modal ratio), `uncertainty_divergence` (remarks vs Q&A uncertainty gap) |
| Linguistic complexity | `role_sentiment_variance` (CEO vs CFO vs analyst gap), `avg_sentence_length`, `vocab_diversity` |
| Momentum (QoQ Δ) | `sentiment_overall_qoq`, `lm_uncertainty_qoq`, `fls_ratio_qoq`, `lm_constraining_qoq`, `modal_confidence_qoq` |

**Preprocessing**: StandardScaler (zero mean, unit variance) · outlier clipping at ±4σ · correlation check (no pairs > 0.85)

**Dimensionality Reduction**: PCA (2D + 3D, scree plot) · t-SNE (perplexity 25, 1000 iterations)

**Clustering**: K-Means (k = 2–6, elbow + silhouette + gap statistic) · DBSCAN (k-distance elbow for ε) · Hierarchical Ward (dendrogram)

**Validation**: Kruskal-Wallis test on 3-day returns by cluster

### Analytics

- **Spearman correlation matrix**: 72 tests with Bonferroni correction. Non-parametric to handle non-normal return distributions.
- **Language shift detector**: Per-company QoQ z-scores on 3 features; events flagged at > 1.5σ.
- **Sector-level correlations**: Separate Spearman analysis per sector for 4 focus features.

---

## How to Run

### Setup

```bash
git clone <repo>
cd earnings-call-intelligence
uv venv --python 3.11 .venv && source .venv/bin/activate
# Pin numpy to 1.x to avoid macOS Sequoia/Accelerate hang
uv pip install pandas "numpy==1.26.4" scipy scikit-learn plotly streamlit statsmodels
```

Place the Motley Fool transcript data (`motley-fool-data.pkl`) in `data/raw/`.

### Run the NLP pipeline (one-time, ~15 min)

```bash
python src/ingestion/transcript_parser.py
python src/ingestion/price_fetcher.py
python src/nlp/finbert_sentiment.py        # ~11 min on MPS
python src/nlp/lm_dictionary.py
python src/nlp/fls_detector.py
python src/nlp/build_features.py
python src/analytics/correlation.py
python src/analytics/language_shift.py
```

### Run the clustering pipeline (~2 min)

```bash
python run_cluster.py
```

Produces:
- `data/features/cluster_features.csv` — 18-feature company-quarter matrix
- `data/features/cluster_projections.csv` — PCA/t-SNE coordinates + cluster labels
- `data/features/cluster_results.json` — profiles, diagnostics, archetypes

### Launch the dashboard

```bash
streamlit run src/app/streamlit_app.py
```

The dashboard has four views:
- **Signal Discovery** — correlation heatmap, scatter plots for the two strongest signals
- **Company Narrative Timeline** — per-company NLP trends with shift event overlays and stock price
- **Transcript Explainability** — sentence-level drill-down with FinBERT scores and FLS attribution
- **Cluster Explorer** — PCA/t-SNE scatter, radar charts, temporal migration heatmap, returns by archetype

---

## Project Structure

```
earnings-call-intelligence/
├── config/
│   └── settings.yaml                  # tickers, paths, model config
├── data/
│   ├── raw/                           # Kaggle pickle (not committed)
│   ├── processed/                     # parsed transcript JSONs (137 files)
│   ├── market/                        # yfinance price CSVs (16 tickers)
│   └── features/                      # all CSV/JSON outputs
│       ├── nlp_features.csv           # 137 × 45 NLP feature matrix
│       ├── sentence_sentiments.csv    # 68,909 sentence-level scores
│       ├── cluster_features.csv       # 137 × 18 clustering feature matrix
│       ├── cluster_projections.csv    # PCA/t-SNE coords + cluster labels
│       └── cluster_results.json      # profiles, diagnostics, archetypes
├── src/
│   ├── ingestion/
│   │   ├── transcript_parser.py       # raw pickle → structured JSON
│   │   └── price_fetcher.py           # yfinance download + return calc
│   ├── nlp/
│   │   ├── finbert_sentiment.py       # sentence-level FinBERT scoring
│   │   ├── lm_dictionary.py           # Loughran-McDonald word scoring
│   │   ├── fls_detector.py            # forward-looking statement detection
│   │   └── build_features.py          # merge features into matrix
│   ├── analytics/
│   │   ├── correlation.py             # Spearman, Mann-Whitney, sector splits
│   │   └── language_shift.py          # QoQ narrative shift detector
│   ├── clustering/
│   │   ├── feature_engineering.py     # 18-feature engineering from NLP matrix
│   │   ├── preprocessing.py           # StandardScaler, correlation check
│   │   ├── dimensionality.py          # PCA + t-SNE projections
│   │   ├── clustering.py              # K-Means, DBSCAN, Hierarchical
│   │   ├── interpretation.py          # archetypes, KW test, JSON export
│   │   └── pipeline.py                # end-to-end orchestration
│   └── app/
│       └── streamlit_app.py           # 4-tab interactive dashboard
├── run_cluster.py                     # standalone clustering runner
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| NLP | ProsusAI/FinBERT · Loughran-McDonald Dict · spaCy |
| ML | scikit-learn (K-Means, DBSCAN, Agglomerative, PCA, t-SNE) |
| Stats | scipy (Spearman, Kruskal-Wallis, Mann-Whitney) |
| Data | pandas · numpy |
| Viz | Plotly · Streamlit |
| Infra | PyTorch (MPS) · yfinance · uv |
