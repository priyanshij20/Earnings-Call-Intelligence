"""
Earnings Call Intelligence Engine — Streamlit Dashboard
Run: streamlit run src/app/streamlit_app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as sp

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent.parent
FEAT_DIR = ROOT / "data" / "features"
MKT_DIR  = ROOT / "data" / "market"

SECTORS = {
    "tech":       ["AAPL", "MSFT", "META", "AMZN", "NVDA"],
    "finance":    ["JPM", "GS", "BAC"],
    "healthcare": ["JNJ", "PFE", "UNH"],
    "retail":     ["WMT", "TGT", "COST"],
    "energy":     ["XOM", "CVX"],
}
TICKER_TO_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}

SECTOR_COLORS = {
    "tech": "#3498db", "finance": "#e67e22", "healthcare": "#2ecc71",
    "retail": "#9b59b6", "energy": "#e74c3c",
}
POS_COL   = "#2ecc71"
NEG_COL   = "#e74c3c"
NEU_COL   = "#7f8c8d"
SHIFT_COL = "#f39c12"
DARK_BG   = "#0e1117"
CARD_BG   = "#1a1d23"

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    df = pd.read_csv(FEAT_DIR / "nlp_features.csv")
    df["date"]   = pd.to_datetime(df["date"])
    df["sector"] = df["ticker"].map(TICKER_TO_SECTOR)
    return df

@st.cache_data
def load_sentences():
    return pd.read_csv(FEAT_DIR / "sentence_sentiments.csv")

@st.cache_data
def load_shifts():
    df = pd.read_csv(FEAT_DIR / "language_shifts.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_corr():
    return pd.read_csv(FEAT_DIR / "correlation_results.csv")

@st.cache_data
def load_findings():
    with open(FEAT_DIR / "findings_summary.json") as f:
        return json.load(f)

@st.cache_data
def load_fls():
    return pd.read_csv(FEAT_DIR / "fls_sentences.csv")

@st.cache_data
def load_prices(ticker: str) -> pd.DataFrame:
    csv = MKT_DIR / f"{ticker}.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[df.index.notna()].sort_index()

# ── helpers ────────────────────────────────────────────────────────────────────
def dark_layout(fig, height=420, margin=None):
    fig.update_layout(
        height=height,
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
        font=dict(color="white", size=12),
        margin=margin or dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#1e2530", zeroline=False),
        yaxis=dict(gridcolor="#1e2530", zeroline=False),
        hovermode="closest",
    )
    return fig

def stat_card(label, value, note="", color=None):
    col = color or "white"
    st.markdown(
        f"<div style='background:{CARD_BG};border-radius:8px;padding:14px 18px;"
        f"margin:4px 0;border-left:4px solid {col}'>"
        f"<div style='font-size:11px;color:#888;text-transform:uppercase'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{col}'>{value}</div>"
        f"<div style='font-size:11px;color:#aaa;margin-top:2px'>{note}</div></div>",
        unsafe_allow_html=True,
    )

def sentence_card(text, score, section, role, width=220):
    col = POS_COL if score > 0.1 else (NEG_COL if score < -0.1 else NEU_COL)
    st.markdown(
        f"<div style='border-left:3px solid {col};padding:7px 12px;margin:4px 0;"
        f"background:{CARD_BG};border-radius:0 6px 6px 0;font-size:12px'>"
        f"<span style='color:{col};font-weight:700'>{score:+.3f}</span> "
        f"<span style='color:#666;font-size:11px'>[{section}/{role}]</span><br>"
        f"<span style='color:#ddd'>{text[:width]}</span></div>",
        unsafe_allow_html=True,
    )

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Call Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

df       = load_features()
sent_df  = load_sentences()
shifts   = load_shifts()
corr_df  = load_corr()
findings = load_findings()
fls_df   = load_fls()

ALL_TICKERS = sorted(df["ticker"].unique())

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Earnings Call Intelligence")
    st.markdown("*NLP signals from 137 earnings calls across 16 S&P 500 companies*")
    st.markdown("---")
    page = st.radio(
        "View",
        ["Signal Discovery", "Company Narrative Timeline", "Transcript Explainability"],
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#666'>"
        "Models: ProsusAI/FinBERT · Loughran-McDonald Dict<br>"
        "Data: Motley Fool Kaggle · yfinance<br>"
        "Stats: Spearman ρ · Bonferroni correction"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 1 — Signal Discovery Panel
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Signal Discovery":
    st.title("Signal Discovery Panel")
    st.caption(
        "Which NLP features from earnings calls predict post-earnings stock returns? "
        "Spearman ρ across 137 transcripts · 16 S&P 500 companies · Bonferroni α = 0.00069"
    )

    # ── top stat row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Transcripts",      "137",     "16 companies · 5 sectors")
    with c2: stat_card("Sentences scored", "68,909",  "FinBERT sentence-level")
    with c3: stat_card("Strongest signal", "ρ = −0.17", "lm_uncertainty_remarks → 1d", NEG_COL)
    with c4: stat_card("Bonferroni tests", "72",      "0 survive correction (expected)")

    st.markdown("---")

    # ── lead scatter: lm_uncertainty_remarks vs return_1d (full width) ────────
    st.subheader("Strongest Signal: Uncertainty Language → 1-Day Return")
    st.caption(
        "When executives hedge more in scripted remarks, stocks underperform the next day. "
        "AMZN's two consecutive outlier quarters (2021-Q4 acceleration, 2022-Q1 crash) are labelled."
    )

    feat_lead  = "lm_uncertainty_remarks"
    scatter_lead = df[[feat_lead, "return_1d", "ticker", "quarter", "sector"]].dropna().copy()
    fig_lead = px.scatter(
        scatter_lead, x=feat_lead, y="return_1d",
        color="sector",
        color_discrete_map=SECTOR_COLORS,
        hover_data=["ticker", "quarter"],
        trendline="ols",
        labels={feat_lead: "LM Uncertainty (Prepared Remarks)", "return_1d": "1-Day Return"},
    )
    fig_lead.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text="ρ = −0.166   p = 0.053",
        showarrow=False,
        font=dict(size=14, color=NEG_COL),
        bgcolor=CARD_BG, bordercolor="#333", borderwidth=1, borderpad=6, align="left",
    )
    # Annotate AMZN 2021-Q4 and 2022-Q1 outliers
    amzn_labels = {"2021-Q4": "AMZN 2021-Q4\n(+16.3%)", "2022-Q1": "AMZN 2022-Q1\n(−14.1%)"}
    for _, r in scatter_lead[scatter_lead["ticker"] == "AMZN"].iterrows():
        if r["quarter"] in amzn_labels:
            fig_lead.add_annotation(
                x=r[feat_lead], y=r["return_1d"],
                text=amzn_labels[r["quarter"]],
                showarrow=True, arrowhead=2, arrowcolor=SHIFT_COL,
                font=dict(size=10, color=SHIFT_COL),
                bgcolor=DARK_BG, borderpad=3, ax=30, ay=-35,
            )
    fig_lead.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.2)
    fig_lead.update_traces(marker=dict(size=8, opacity=0.78), selector=dict(mode="markers"))
    dark_layout(fig_lead, height=400)
    fig_lead.update_layout(
        yaxis=dict(tickformat=".1%", gridcolor="#1e2530"),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
    )
    st.plotly_chart(fig_lead, use_container_width=True)

    st.markdown("---")

    # ── correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Spearman ρ: NLP Features × Return Windows")
    st.caption("1-day column is the primary signal window — uncertainty effects decay across 3 and 7 days.")

    DISPLAY_FEATURES = [
        "lm_uncertainty_remarks", "fls_ratio", "fls_hedging_count",
        "lm_negative_remarks",    "lm_uncertainty_qa",
        "lm_positive",            "fls_positive_count",
        "sentiment_divergence",   "sentiment_overall",
        "lm_uncertainty",         "lm_strong_modal",
        "lm_weak_modal",          "lm_constraining",
    ]
    pivot = corr_df[corr_df["feature"].isin(DISPLAY_FEATURES)].pivot(
        index="feature", columns="return_window", values="spearman_rho"
    ).reindex(columns=["return_1d", "return_3d", "return_7d"])
    pivot = pivot.reindex(pivot["return_1d"].abs().sort_values(ascending=False).index)

    piv_p = corr_df[corr_df["feature"].isin(DISPLAY_FEATURES)].pivot(
        index="feature", columns="return_window", values="p_value"
    ).reindex(index=pivot.index, columns=["return_1d", "return_3d", "return_7d"])
    text_vals = pivot.round(3).astype(str).where(
        piv_p >= 0.05, pivot.round(3).astype(str) + "*"
    )

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=["1-Day Return ◀", "3-Day Return", "7-Day Return"],
        y=pivot.index.tolist(),
        text=text_vals.values,
        texttemplate="%{text}",
        colorscale="RdBu", zmid=0, zmin=-0.25, zmax=0.25,
        colorbar=dict(title="ρ", thickness=12, len=0.8),
        hoverongaps=False,
    ))
    # highlight 1-day column with a bounding rect
    n_rows = len(pivot)
    fig_heat.add_shape(
        type="rect",
        x0=-0.5, x1=0.5, y0=-0.5, y1=n_rows - 0.5,
        line=dict(color=SHIFT_COL, width=2),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )
    dark_layout(fig_heat, height=440, margin=dict(l=200, r=20, t=20, b=20))
    fig_heat.update_layout(yaxis=dict(tickfont=dict(size=11)))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("\\* p < 0.05 (uncorrected) · ◀ 1-day column bordered in amber. No features survive Bonferroni correction — consistent with EMH for large-cap equities at n=137.")

    st.markdown("---")

    # ── secondary scatter: fls_hedging_count (half-width) ─────────────────────
    st.subheader("Secondary Signal: FLS Hedging Count → 1-Day Return")
    col_sc, col_interp = st.columns([3, 2])
    with col_sc:
        feat2 = "fls_hedging_count"
        scatter2 = df[[feat2, "return_1d", "ticker", "quarter", "sector"]].dropna()
        fig_sc2 = px.scatter(
            scatter2, x=feat2, y="return_1d",
            color="sector", color_discrete_map=SECTOR_COLORS,
            hover_data=["ticker", "quarter"],
            trendline="ols",
            labels={feat2: "FLS Hedging Count", "return_1d": "1-Day Return"},
        )
        fig_sc2.add_annotation(
            x=0.02, y=0.97, xref="paper", yref="paper",
            text="ρ = −0.113   p = 0.190",
            showarrow=False,
            font=dict(size=13, color=NEG_COL),
            bgcolor=CARD_BG, bordercolor="#333", borderwidth=1, borderpad=6,
        )
        fig_sc2.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.2)
        fig_sc2.update_traces(marker=dict(size=7, opacity=0.75), selector=dict(mode="markers"))
        dark_layout(fig_sc2, height=340)
        fig_sc2.update_layout(
            yaxis=dict(tickformat=".1%", gridcolor="#1e2530"),
            legend=dict(orientation="h", y=-0.22, font=dict(size=10)),
        )
        st.plotly_chart(fig_sc2, use_container_width=True)
    with col_interp:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:{CARD_BG};border-left:4px solid {NEG_COL};"
            f"border-radius:4px;padding:14px 16px;font-size:13px;line-height:1.6'>"
            f"More FLS language is <b>not bullish</b>. High hedging FLS density reflects "
            f"management needing to qualify guidance rather than simply reporting strong results. "
            f"The direction is consistent with the uncertainty signal, though not individually "
            f"significant (p = 0.190).</div>",
            unsafe_allow_html=True,
        )

    # ── interpretation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Interpretation")
    KEY_FINDINGS = [
        {
            "finding": "Uncertainty language in prepared remarks is the strongest predictor",
            "rho": -0.166, "p": 0.053,
            "detail": "When executives hedge more in scripted remarks, stocks underperform the next day. "
                      "The effect is stronger in remarks (ρ=−0.166) than Q&A (ρ=−0.113), consistent with "
                      "prepared language being more deliberate and therefore more informative.",
        },
        {
            "finding": "FLS hedging count and FLS ratio both negatively correlate with 1-day returns",
            "rho": -0.137, "p": 0.113,
            "detail": "More forward-looking statement density is not bullish — it reflects management "
                      "needing to qualify guidance rather than simply reporting strong results.",
        },
        {
            "finding": "Remarks–Q&A divergence is structural, not a quarterly signal",
            "rho": -0.114, "p": 0.340,
            "detail": "93% of calls have positive divergence (scripted remarks more positive than Q&A). "
                      "The within-company demeaned version trends in the right direction but is underpowered at n=137.",
        },
        {
            "finding": "No features survive Bonferroni correction — consistent with EMH",
            "rho": 0.0, "p": 1.0,
            "detail": "With 72 simultaneous tests, corrected α = 0.00069. Large-cap S&P 500 names are "
                      "heavily covered; earnings surprises are priced within minutes. These are real "
                      "directional tendencies, not arbitrageable edges.",
        },
    ]
    for f in KEY_FINDINGS:
        border_col = NEG_COL if f["rho"] < 0 else NEU_COL
        rho_str = f"ρ = {f['rho']:+.3f}" if f["rho"] != 0.0 else "0 / 72 survive"
        p_str   = f"p = {f['p']:.3f}"    if f["p"] < 1.0    else "Bonferroni α = 0.00069"
        st.markdown(
            f"<div style='border-left:4px solid {border_col};padding:10px 16px;"
            f"margin:8px 0;background:{CARD_BG};border-radius:4px'>"
            f"<b>{f['finding']}</b>"
            f"<span style='color:{border_col};margin-left:12px'>{rho_str}</span>"
            f"<span style='color:#888;margin-left:8px'>{p_str}</span><br>"
            f"<span style='font-size:12px;color:#aaa;margin-top:4px;display:block'>"
            f"{f['detail']}</span></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 2 — Company Narrative Timeline
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Company Narrative Timeline":
    st.title("Company Narrative Timeline")
    st.caption("How executive language evolved quarter-by-quarter — and whether the market agreed.")

    # ── controls ──────────────────────────────────────────────────────────────
    col_t, col_m = st.columns([2, 2])
    with col_t:
        ticker = st.selectbox(
            "Company",
            ALL_TICKERS,
            index=ALL_TICKERS.index("AMZN"),
            help="AMZN and TGT show the clearest narrative-to-return stories in this dataset.",
        )
    with col_m:
        nlp_metric = st.selectbox(
            "NLP signal",
            ["sentiment_overall", "lm_uncertainty", "fls_hedging_count", "sentiment_divergence"],
            format_func=lambda x: {
                "sentiment_overall":    "Overall Sentiment (FinBERT)",
                "lm_uncertainty":       "Uncertainty Language (LM Dict)",
                "fls_hedging_count":    "Hedging FLS Count",
                "sentiment_divergence": "Remarks–Q&A Divergence",
            }[x],
        )

    co_df    = df[df["ticker"] == ticker].sort_values("date").copy()
    prices   = load_prices(ticker)
    co_shift = shifts[(shifts["ticker"] == ticker) & shifts["is_shift_event"]].copy()

    if co_df.empty:
        st.warning(f"No transcripts found for {ticker}.")
        st.stop()

    # ── dual-axis chart ───────────────────────────────────────────────────────
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # stock price
    if not prices.empty:
        lo = co_df["date"].min() - pd.Timedelta(days=45)
        hi = co_df["date"].max() + pd.Timedelta(days=45)
        px_trim = prices.loc[(prices.index >= lo) & (prices.index <= hi)]
        fig.add_trace(
            go.Scatter(
                x=px_trim.index, y=px_trim["Close"],
                name="Stock Price", line=dict(color="#4a6fa5", width=1.5),
                hovertemplate="%{x|%b %Y}  $%{y:.1f}<extra>Stock</extra>",
            ),
            secondary_y=True,
        )

    # NLP metric bars
    bar_colors = [
        (POS_COL if v >= 0 else NEG_COL)
        for v in co_df[nlp_metric].fillna(0)
    ]
    metric_label = {
        "sentiment_overall":    "Sentiment",
        "lm_uncertainty":       "Uncertainty",
        "fls_hedging_count":    "Hedging FLS",
        "sentiment_divergence": "Divergence",
    }[nlp_metric]

    fig.add_trace(
        go.Bar(
            x=co_df["date"], y=co_df[nlp_metric],
            name=metric_label,
            marker_color=bar_colors, opacity=0.85,
            hovertemplate="%{x|%b %Y}  %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Named labels for the two landmark AMZN shift events
    AMZN_EVENT_LABELS = {
        "2021-Q4": "⚡ 2021-Q4\nAWS re-acceleration\n+16.3%",
        "2022-Q1": "⚡ 2022-Q1\nMacro reversal\n−14.1%",
    }

    # shift event overlays
    for _, s in co_shift.iterrows():
        fig.add_vrect(
            x0=s["date"] - pd.Timedelta(days=20),
            x1=s["date"] + pd.Timedelta(days=20),
            fillcolor=SHIFT_COL, opacity=0.10, line_width=0,
        )
        ret3 = s.get("return_3d")
        ret_str = f"{ret3:+.1%}" if pd.notna(ret3) else ""
        # Use named label for landmark AMZN events; generic label otherwise
        if ticker == "AMZN" and s["quarter"] in AMZN_EVENT_LABELS:
            label_text = AMZN_EVENT_LABELS[s["quarter"]]
        else:
            label_text = f"⚡ {s['quarter']} {ret_str}"
        fig.add_annotation(
            x=s["date"], y=1.04, yref="paper",
            text=label_text,
            showarrow=False,
            font=dict(size=10, color=SHIFT_COL),
            bgcolor=DARK_BG, borderpad=3,
        )

    dark_layout(fig, height=440, margin=dict(l=10, r=10, t=55, b=10))
    fig.update_layout(
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis_title="", hovermode="x unified",
        yaxis_title=metric_label,
        yaxis2_title="Stock Price ($)",
        yaxis2=dict(gridcolor="#1e2530"),
        barmode="relative",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚡ = language shift event (>1.5 σ QoQ change). Shaded band = ±20 days around shift.")

    # ── quarter metrics table ─────────────────────────────────────────────────
    show_cols = ["quarter", "sentiment_overall", "sentiment_divergence",
                 "lm_uncertainty", "fls_hedging_count", "return_1d", "return_3d"]
    show_cols = [c for c in show_cols if c in co_df.columns]

    def color_return(val):
        if pd.isna(val): return ""
        return f"color: {POS_COL}" if val > 0 else f"color: {NEG_COL}"

    styled = (
        co_df[show_cols].set_index("quarter")
        .style
        .format({
            "sentiment_overall":    "{:.3f}",
            "sentiment_divergence": "{:.3f}",
            "lm_uncertainty":       "{:.5f}",
            "fls_hedging_count":    "{:.0f}",
            "return_1d":            "{:+.2%}",
            "return_3d":            "{:+.2%}",
        })
        .map(color_return, subset=["return_1d", "return_3d"])
    )
    with st.expander("Quarterly data table"):
        st.dataframe(styled, use_container_width=True)

    # ── shift event detail ────────────────────────────────────────────────────
    if not co_shift.empty:
        st.subheader(f"Narrative shift events — {ticker}")
        for _, s in co_shift.sort_values("max_z_score", ascending=False).iterrows():
            ret3   = s.get("return_3d")
            ret_col = POS_COL if pd.notna(ret3) and ret3 > 0 else NEG_COL
            sent_d  = s.get("sentiment_overall_delta", np.nan)
            unc_d   = s.get("lm_uncertainty_delta", np.nan)
            sent_str = f"&nbsp;&nbsp;Δsentiment = {sent_d:+.3f}" if pd.notna(sent_d) else ""
            unc_str  = f"&nbsp;&nbsp;Δuncertainty = {unc_d:+.5f}" if pd.notna(unc_d) else ""
            ret_str  = (f"&nbsp;&nbsp;<span style='color:{ret_col}'>3d return: {ret3:+.1%}</span>"
                        if pd.notna(ret3) else "")
            st.markdown(
                f"<div style='border-left:4px solid {SHIFT_COL};padding:10px 14px;"
                f"margin:6px 0;background:{CARD_BG};border-radius:4px'>"
                f"<b>{s['prev_quarter']} → {s['quarter']}</b>"
                f"&nbsp;&nbsp;z = {s['max_z_score']:.2f}"
                f"{sent_str}{unc_str}{ret_str}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info(f"No narrative shift events flagged for {ticker}.")


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW 3 — Transcript Explainability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Transcript Explainability":
    st.title("Transcript Explainability")
    st.caption("Drill into any single earnings call — sentence-level attribution, FLS detection, and remarks-vs-Q&A breakdown.")

    col_t, col_q = st.columns(2)
    with col_t:
        ticker = st.selectbox(
            "Company", ALL_TICKERS, index=ALL_TICKERS.index("META"),
            help="META 2022-Q2 shows the clearest remarks-vs-Q&A divergence and the FinBERT anchor bias case.",
        )
    quarters_avail = sorted(df[df["ticker"] == ticker]["quarter"].unique(), reverse=True)
    default_q_idx  = quarters_avail.index("2022-Q2") if "2022-Q2" in quarters_avail else 0
    with col_q:
        quarter = st.selectbox("Quarter", quarters_avail, index=default_q_idx)

    tid  = f"{ticker}_{quarter}"
    row  = df[(df["ticker"] == ticker) & (df["quarter"] == quarter)]
    sents = sent_df[sent_df["transcript_id"] == tid].copy()

    if row.empty or sents.empty:
        st.warning("No sentence-level data for this transcript.")
        st.stop()

    row = row.iloc[0]

    # ── header metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: stat_card("Overall Sentiment",  f"{row['sentiment_overall']:.3f}")
    with m2: stat_card("Remarks Sentiment",  f"{row['sentiment_remarks']:.3f}" if pd.notna(row.get('sentiment_remarks')) else "—")
    with m3: stat_card("Q&A Sentiment",      f"{row['sentiment_qa']:.3f}"      if pd.notna(row.get('sentiment_qa'))      else "—")
    div = row.get("sentiment_divergence")
    with m4: stat_card(
        "Divergence (R−Q)",
        f"{div:.3f}" if pd.notna(div) else "—",
        "remarks more positive than Q&A" if (pd.notna(div) and div > 0) else "",
        NEG_COL if (pd.notna(div) and div > 0.2) else None,
    )
    ret3 = row.get("return_3d")
    with m5: stat_card(
        "3-Day Return",
        f"{ret3:+.2%}" if pd.notna(ret3) else "—",
        color=POS_COL if (pd.notna(ret3) and ret3 > 0) else NEG_COL,
    )

    # ── FinBERT misclassification callout (META 2022-Q2) ─────────────────────
    if ticker == "META" and quarter == "2022-Q2":
        _rem_s  = f"{row.get('sentiment_remarks', 0):.3f}"
        _qa_s   = f"{row.get('sentiment_qa', 0):.3f}"
        _div_s  = f"{div:.3f}" if pd.notna(div) else "—"
        st.markdown(
            f"<div style='background:#1a1200;border:1px solid {SHIFT_COL};"
            f"border-radius:8px;padding:14px 18px;margin:10px 0'>"
            f"<b style='color:{SHIFT_COL}'>⚠ FinBERT anchor bias — visible in this call</b><br>"
            f"<span style='font-size:13px;color:#ddd'>"
            f"This call (2022-Q2) is where Zuckerberg announced headcount freezes and cost cuts "
            f"after the DAU decline. Sentences like <i>\"we anticipate headcount growth will slow\"</i> "
            f"are classified <b style='color:{POS_COL}'>positive</b> by FinBERT because the model "
            f"anchors on the word \"growth\" — despite the sentence expressing a slowdown. "
            f"This inflates the remarks sentiment ({_rem_s}) "
            f"relative to what Q&A tone ({_qa_s}) reveals. "
            f"The divergence here ({_div_s}) reflects genuine executive spin, "
            f"but FinBERT over-credits the scripted language.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── remarks vs Q&A breakdown ──────────────────────────────────────────────
    st.subheader("Remarks vs. Q&A Sentiment Breakdown")
    col_rem, col_qa = st.columns(2)

    for col, section_label, section_key in [
        (col_rem, "Prepared Remarks", "remarks"),
        (col_qa,  "Q&A",             "qa"),
    ]:
        with col:
            sec = sents[sents["section"] == section_key]
            if sec.empty:
                st.info(f"No {section_label} data.")
                continue

            mean_s = sec["sentiment_score"].mean()
            pos_n  = (sec["sentiment_score"] > 0.1).sum()
            neg_n  = (sec["sentiment_score"] < -0.1).sum()
            neu_n  = len(sec) - pos_n - neg_n

            st.markdown(
                f"<div style='background:{CARD_BG};border-radius:8px;padding:12px 16px;"
                f"margin-bottom:8px'>"
                f"<b>{section_label}</b> &nbsp; "
                f"<span style='color:{'#2ecc71' if mean_s > 0 else '#e74c3c'}'>"
                f"mean = {mean_s:+.3f}</span> &nbsp; "
                f"<span style='font-size:12px;color:#888'>{len(sec)} sentences</span><br>"
                f"<span style='color:{POS_COL}'>▲ {pos_n} positive</span> &nbsp; "
                f"<span style='color:{NEG_COL}'>▼ {neg_n} negative</span> &nbsp; "
                f"<span style='color:{NEU_COL}'>— {neu_n} neutral</span></div>",
                unsafe_allow_html=True,
            )

            # role breakdown
            role_grp = sec.groupby("role")["sentiment_score"].mean().sort_values()
            if len(role_grp) > 1:
                fig_role = px.bar(
                    role_grp.reset_index(), x="sentiment_score", y="role",
                    orientation="h", color="sentiment_score",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    range_color=[-0.5, 0.5],
                    labels={"sentiment_score": "Mean Sentiment", "role": "Speaker Role"},
                )
                fig_role.update_layout(
                    height=160, showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG,
                    font=dict(color="white", size=11),
                    xaxis=dict(gridcolor="#1e2530"),
                )
                st.plotly_chart(fig_role, use_container_width=True)

    st.markdown("---")

    # ── section toggle + top sentences ────────────────────────────────────────
    st.subheader("Sentence-Level Attribution")
    sec_filter = st.radio("Section filter", ["All", "Remarks only", "Q&A only"], horizontal=True)
    display_sents = sents.copy()
    if sec_filter == "Remarks only":
        display_sents = sents[sents["section"] == "remarks"]
    elif sec_filter == "Q&A only":
        display_sents = sents[sents["section"] == "qa"]

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**Most positive sentences**")
        for _, r in display_sents.nlargest(5, "sentiment_score").iterrows():
            sentence_card(r["sentence"], r["sentiment_score"], r["section"], r["role"])
    with col_neg:
        st.markdown("**Most negative sentences**")
        for _, r in display_sents.nsmallest(5, "sentiment_score").iterrows():
            sentence_card(r["sentence"], r["sentiment_score"], r["section"], r["role"])

    # ── full heatmap ──────────────────────────────────────────────────────────
    with st.expander(f"Full sentence heatmap ({len(display_sents)} sentences)"):
        n_show = st.slider("Show first N sentences", 20, min(300, len(display_sents)), 80, 10)
        for _, r in display_sents.head(n_show).iterrows():
            sentence_card(r["sentence"], r["sentiment_score"], r["section"], r["role"], width=260)

    # ── FLS panel ─────────────────────────────────────────────────────────────
    st.markdown("---")
    fls_t = fls_df[fls_df["transcript_id"] == tid]
    st.subheader(f"Forward-Looking Statements ({len(fls_t)} detected)")

    if fls_t.empty:
        st.info("No FLS detected for this transcript.")
    else:
        fls_filter = st.radio("Type", ["All", "Positive", "Hedging"], horizontal=True)
        show_fls = fls_t if fls_filter == "All" else fls_t[fls_t["fls_type"] == fls_filter.lower()]

        pos_count = (fls_t["fls_type"] == "positive").sum()
        hed_count = (fls_t["fls_type"] == "hedging").sum()
        st.markdown(
            f"<div style='background:{CARD_BG};border-radius:6px;padding:10px 14px;margin-bottom:10px'>"
            f"<span style='color:{POS_COL}'>▲ {pos_count} positive FLS</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:{SHIFT_COL}'>⚠ {hed_count} hedging FLS</span>"
            f"&nbsp;&nbsp;"
            f"<span style='color:#888;font-size:12px'>"
            f"FLS ratio = {row.get('fls_ratio', 0):.1%} of all sentences</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        for _, r in show_fls.iterrows():
            fls_col   = POS_COL if r["fls_type"] == "positive" else SHIFT_COL
            score_str = f"{r['sentiment_score']:+.3f}" if pd.notna(r.get("sentiment_score")) else "—"
            st.markdown(
                f"<div style='border-left:3px solid {fls_col};padding:7px 12px;margin:4px 0;"
                f"background:{CARD_BG};border-radius:0 6px 6px 0;font-size:12px'>"
                f"<span style='color:{fls_col};font-weight:700'>[{r['fls_type']}]</span>"
                f"&nbsp;<span style='color:#888'>FinBERT: {score_str}</span>"
                f"&nbsp;<span style='color:#666;font-size:11px'>[{r.get('section','?')}]</span><br>"
                f"<span style='color:#ddd'>{r['sentence'][:240]}</span></div>",
                unsafe_allow_html=True,
            )
