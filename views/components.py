"""
Shared UI components used across single business and compare views.
"""
import streamlit as st

COLORS = [
    "#c0392b", "#2980b9", "#27ae60", "#e67e22",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50",
    "#f39c12", "#1abc9c"
]

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
    .main { background-color: #faf8f5; }
    .block-container { padding-top: 2rem; }

    .cluster-card {
        background: white;
        border-left: 5px solid #c0392b;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .cluster-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.3rem;
    }
    .cluster-meta { font-size: 0.85rem; color: #888; margin-bottom: 0.6rem; }
    .tag {
        display: inline-block;
        background: #f0ebe3;
        color: #555;
        font-size: 0.75rem;
        padding: 2px 10px;
        border-radius: 20px;
        margin: 2px 3px 2px 0;
    }
    .review-quote {
        font-style: italic;
        color: #444;
        font-size: 0.9rem;
        border-left: 3px solid #e8e0d5;
        padding-left: 0.8rem;
        margin: 0.4rem 0;
    }
    .sentiment-pos { color: #27ae60; font-weight: 600; }
    .sentiment-neg { color: #c0392b; font-weight: 600; }
    .sentiment-neu { color: #888;    font-weight: 600; }

    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
    }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def sentiment_color(score: float) -> tuple[str, str]:
    """Return (css_class, label) for a sentiment score."""
    if score >= 0.05:
        return "sentiment-pos", "● Positive"
    elif score <= -0.05:
        return "sentiment-neg", "● Negative"
    return "sentiment-neu", "● Neutral"


def cluster_card(s: dict) -> str:
    """Build HTML for a single cluster card."""
    sent_class, sent_text = sentiment_color(s["avg_sentiment"])
    rating_str = f" · {s['avg_rating']} ★" if s.get("avg_rating") else ""

    tag = s.get("sentiment_tag", "Mixed")
    border_color = {"Praise": "#27ae60", "Complaints": "#c0392b"}.get(tag, "#f39c12")

    tags_html = "".join(f'<span class="tag">{w}</span>' for w in s["top_words"])
    quotes_html = "".join(
        f'<div class="review-quote">"{r[:160]}{"..." if len(r) > 160 else ""}"</div>'
        for r in s["sample_reviews"][:3]
    )

    return f"""
    <div class="cluster-card" style="border-left-color: {border_color}">
        <div class="cluster-title">{s['name']}</div>
        <div class="cluster-meta">
            {s['review_count']} reviews
            <span class="{sent_class}"> · {sent_text}</span>
            {rating_str}
        </div>
        <div style="margin-bottom:0.7rem">{tags_html}</div>
        {quotes_html}
    </div>
    """