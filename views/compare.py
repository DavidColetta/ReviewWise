"""
Compare Businesses view — add companies, PCA landscape, and comparison charts.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from pipeline import THEME_KEYWORDS, load_data, extract_signals
from views.components import COLORS
from utils.charts import base_layout, topic_group_bars, sentiment_bar


# ─────────────────────────────────────────────
# Cached scraping
# ─────────────────────────────────────────────

@st.cache_data
def scrape_and_process(url: str, max_pages: int) -> dict:
    """Scrape a Trustpilot URL and return an aggregated company feature dict."""
    from scraper import scrape_trustpilot
    df, name = scrape_trustpilot(url, max_pages=max_pages)
    df = load_data(df)
    df = extract_signals(df)

    theme_cols = list(THEME_KEYWORDS.keys())
    feature = {
        "name":          name,
        "review_count":  len(df),
        "avg_sentiment": round(float(df["sentiment_score"].mean()), 3),
        "avg_rating":    round(float(df["rating"].mean()), 2) if "rating" in df.columns else None,
        "pct_positive":  round(float((df["sentiment_label"] == "Positive").mean()) * 100, 1),
        "pct_negative":  round(float((df["sentiment_label"] == "Negative").mean()) * 100, 1),
        "sample_reviews": (
            df.nlargest(3, "sentiment_score")["review_text"].tolist() +
            df.nsmallest(2, "sentiment_score")["review_text"].tolist()
        ),
    }
    for tc in theme_cols:
        feature[tc] = round(float(df[tc].mean()), 4)
        # Sentiment among reviews that meaningfully mention this topic (score > 0.1)
        mentioning = df[df[tc] > 0]  # any keyword hit counts as a mention
        if len(mentioning) > 0:
            feature[f"{tc}_sentiment"] = round(float(mentioning["sentiment_score"].mean()), 3)
        else:
            feature[f"{tc}_sentiment"] = 0.0

    return feature


# ─────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────

def render(add_clicked: bool, compare_url: str, compare_pages: int):
    """Entry point called from app.py."""

    st.markdown("# ⚖️ Compare Businesses")
    st.caption("Add companies via Trustpilot URLs. Each company becomes a bubble on the map.")

    # ── Handle add button ──
    if add_clicked and compare_url:
        with st.spinner(f"Scraping and processing {compare_url}..."):
            try:
                feature = scrape_and_process(compare_url, compare_pages)
                if "compare_companies" not in st.session_state:
                    st.session_state.compare_companies = []

                existing = [c["name"] for c in st.session_state.compare_companies]
                if feature["name"] in existing:
                    st.warning(f"**{feature['name']}** is already added.")
                else:
                    st.session_state.compare_companies.append(feature)
                    st.success(f"Added **{feature['name']}** ({feature['review_count']} reviews)")
            except Exception as e:
                st.error(f"Failed to scrape {compare_url}: {e}")

    companies = st.session_state.get("compare_companies", [])

    if len(companies) < 2:
        remaining = 2 - len(companies)
        noun = "company" if remaining == 1 else "companies"
        st.info(f"👈 Add at least {remaining} more {noun} using the sidebar to start comparing.")
        return

    # ── Build aggregated dataframe ──
    theme_cols = list(THEME_KEYWORDS.keys())
    co_df = pd.DataFrame(companies)
    co_df = _add_pca_coords(co_df, theme_cols)
    variance = co_df.attrs.get("variance", [0.0, 0.0])

    # ── Tabs ──
    ctab1, ctab2, ctab3, ctab4 = st.tabs([
        "🗺️ Company Map", "📊 Topic Breakdown",
        "📈 Sentiment", "🏆 Rankings"
    ])

    with ctab1:
        _tab_map(co_df, variance)

    with ctab2:
        _tab_topics(co_df, theme_cols)

    with ctab3:
        _tab_sentiment(co_df)

    with ctab4:
        _tab_rankings(co_df)


# ─────────────────────────────────────────────
# PCA helper
# ─────────────────────────────────────────────

def _add_pca_coords(co_df: pd.DataFrame, theme_cols: list) -> pd.DataFrame:
    feature_cols = ["avg_sentiment", "avg_rating"] + theme_cols
    feature_cols = [c for c in feature_cols if c in co_df.columns and co_df[c].notna().all()]

    X = co_df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    n_components = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    co_df = co_df.copy()
    co_df["pca_x"] = coords[:, 0]
    co_df["pca_y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
    co_df.attrs["variance"] = pca.explained_variance_ratio_.tolist()
    return co_df


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

def _tab_map(co_df: pd.DataFrame, variance: list):
    st.subheader("Company Landscape")
    st.caption(
        f"Position reflects review profile similarity. Size = review count. "
        f"Chart captures {sum(variance):.0%} of variance."
    )

    fig = go.Figure()
    max_count = co_df["review_count"].max()

    for i, row in co_df.iterrows():
        reviews_text = "<br>".join(
            f'• {r[:90]}{"…" if len(r) > 90 else ""}'
            for r in row["sample_reviews"][:3]
        )
        rating_str = f"{row['avg_rating']} ★  ·  " if row.get("avg_rating") else ""
        hover = (
            f"<b>{row['name']}</b><br>"
            f"{row['review_count']} reviews  ·  {rating_str}"
            f"sentiment {row['avg_sentiment']:+.2f}<br><br>"
            f"{reviews_text}"
        )
        fig.add_trace(go.Scatter(
            x=[row["pca_x"]], y=[row["pca_y"]],
            mode="markers+text",
            marker=dict(
                size=row["review_count"],
                sizemode="area",
                sizeref=2.0 * max_count / (80 ** 2),
                color=COLORS[i % len(COLORS)],
                opacity=0.75,
                line=dict(width=2, color="white"),
            ),
            text=row["name"],
            textposition="top center",
            textfont=dict(size=12, color="black", family="Source Sans 3"),
            hovertemplate=hover + "<extra></extra>",
            name=row["name"],
        ))

    x_title = f"PC1 ({variance[0]:.0%} variance)" if variance else ""
    y_title = f"PC2 ({variance[1]:.0%} variance)" if len(variance) > 1 else ""

    fig.update_layout(**base_layout(
        height=560,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text=x_title, font=dict(size=11, color="#888"))),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text=y_title, font=dict(size=11, color="#888"))),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#ddd"),
        margin=dict(t=20, b=40, l=20, r=20),
    ))
    st.plotly_chart(fig, use_container_width=True)


def _tab_topics(co_df: pd.DataFrame, theme_cols: list):
    st.subheader("Topic Mentions vs Sentiment")
    st.caption(
        "Each bubble is one company × topic. "
        "**X axis** = how often the topic is mentioned. "
        "**Y axis** = sentiment among those mentions. "
        "Bubbles above the line are praised; below are criticised."
    )

    import plotly.graph_objects as go

    fig = go.Figure()

    # Reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="#ccc", line_width=1)
    fig.add_hrect(y0=0, y1=1.1, fillcolor="#27ae60", opacity=0.03, line_width=0)
    fig.add_hrect(y0=-1.1, y1=0, fillcolor="#c0392b", opacity=0.03, line_width=0)

    for i, row in co_df.iterrows():
        x_vals, y_vals, labels, hovers = [], [], [], []
        for tc in theme_cols:
            mention_rate = row.get(tc, 0.0)
            sentiment    = row.get(f"{tc}_sentiment", 0.0)
            x_vals.append(mention_rate)
            y_vals.append(sentiment)
            labels.append(tc)
            hovers.append(
                f"<b>{row['name']}</b><br>"
                f"Topic: {tc}<br>"
                f"Mention rate: {mention_rate:.3f}<br>"
                f"Sentiment: {sentiment:+.3f}"
            )

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+text",
            name=row["name"],
            marker=dict(
                size=18,
                color=COLORS[i % len(COLORS)],
                opacity=0.8,
                line=dict(width=1.5, color="white"),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=10, color="#333"),
            hovertemplate=[h + "<extra></extra>" for h in hovers],
        ))

    fig.update_layout(**base_layout(
        height=520,
        xaxis=dict(
            title="Mention rate (higher = topic comes up more often)",
            tickformat=".3f",
            showgrid=True, gridcolor="#eee",
        ),
        yaxis=dict(
            title="Avg sentiment among mentions",
            range=[-1.1, 1.1],
            zeroline=False,
            showgrid=True, gridcolor="#eee",
        ),
        legend=dict(orientation="h", y=-0.18),
        margin=dict(t=20, b=60, l=20, r=20),
        annotations=[
            dict(x=0, y=0.55, xref="paper", yref="paper",
                 text="← Praised", showarrow=False,
                 font=dict(color="#27ae60", size=11)),
            dict(x=0, y=0.42, xref="paper", yref="paper",
                 text="← Criticised", showarrow=False,
                 font=dict(color="#c0392b", size=11)),
        ],
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Topics near the right edge are frequently discussed. "
        "Topics near the top are spoken about positively. "
        "Bottom-right is the danger zone — talked about a lot, negatively."
    )


def _tab_sentiment(co_df: pd.DataFrame):
    st.subheader("Sentiment Comparison")
    scol1, scol2 = st.columns(2)

    with scol1:
        fig = sentiment_bar(
            co_df["name"].tolist(),
            co_df["avg_sentiment"].tolist(),
        )
        fig.update_layout(title="Average Sentiment Score")
        st.plotly_chart(fig, use_container_width=True)

    with scol2:
        pct_colors = [COLORS[i % len(COLORS)] for i in range(len(co_df))]
        fig_pos = go.Figure(go.Bar(
            x=co_df["name"],
            y=co_df["pct_positive"],
            marker_color=pct_colors,
            text=[f"{v:.0f}%" for v in co_df["pct_positive"]],
            textposition="outside",
        ))
        fig_pos.update_layout(**base_layout(
            title="% Positive Reviews",
            height=360,
            yaxis=dict(range=[0, 110]),
            margin=dict(t=40, b=10),
        ))
        st.plotly_chart(fig_pos, use_container_width=True)


def _tab_rankings(co_df: pd.DataFrame):
    st.subheader("Company Rankings")

    rank_df = co_df[["name", "avg_sentiment", "avg_rating", "pct_positive", "review_count"]].copy()
    rank_df.columns = ["Company", "Avg Sentiment", "Avg Rating", "% Positive", "Reviews"]
    rank_df["Avg Rating"]   = rank_df["Avg Rating"].apply(lambda x: f"{x} ★" if x else "—")
    rank_df["% Positive"]   = rank_df["% Positive"].apply(lambda x: f"{x:.0f}%")
    rank_df["Avg Sentiment"] = rank_df["Avg Sentiment"].apply(lambda x: f"{x:+.3f}")
    rank_df = rank_df.sort_values("% Positive", ascending=False).reset_index(drop=True)
    rank_df.index += 1

    st.dataframe(rank_df, use_container_width=True)

    csv = co_df.drop(columns=["sample_reviews", "pca_x", "pca_y"], errors="ignore").to_csv(index=False)
    st.download_button(
        "⬇️ Download comparison CSV",
        data=csv,
        file_name="company_comparison.csv",
        mime="text/csv",
    )