"""
Reusable Plotly figure builders shared between views.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

FONT = "Source Sans 3"
BG   = "#faf8f5"


def base_layout(**kwargs) -> dict:
    """Common layout defaults for all charts."""
    return dict(
        font_family=FONT,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        **kwargs,
    )


def bubble_chart(
    df: pd.DataFrame,
    colors: list,
    hover_texts: list,
    pca_meta: dict = None,
    height: int = 620,
) -> go.Figure:
    """
    Cluster bubble chart — one trace per row in df.
    df must have: pca_x, pca_y, review_count, theme (or name) columns.
    """
    fig = go.Figure()
    label_col = "theme" if "theme" in df.columns else "name"

    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["pca_x"]], y=[row["pca_y"]],
            mode="markers+text",
            marker=dict(
                size=row["review_count"] * 6,
                sizemode="area",
                sizeref=2.0 * df["review_count"].max() / (80 ** 2),
                color=colors[i % len(colors)],
                opacity=0.7,
                line=dict(width=2, color="white"),
            ),
            text=str(row[label_col]).split(" — ")[0],
            textposition="middle center",
            textfont=dict(size=13, color="black", family=FONT),
            hovertemplate=hover_texts[i] + "<extra></extra>",
            name=str(row[label_col]),
        ))

    x_title = pca_meta["axis_x_label"] if pca_meta else ""
    y_title = pca_meta["axis_y_label"] if pca_meta else ""

    fig.update_layout(**base_layout(
        height=height,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text=x_title, font=dict(size=11, color="#888"))),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   title=dict(text=y_title, font=dict(size=11, color="#888"))),
        hoverlabel=dict(bgcolor="white", font_size=13, bordercolor="#ddd"),
        margin=dict(t=20, b=60, l=20, r=20),
    ))
    return fig


def sentiment_donut(sentiment_counts: dict, height: int = 320) -> go.Figure:
    """Donut chart of positive/neutral/negative counts."""
    colors = {"Positive": "#27ae60", "Neutral": "#bdc3c7", "Negative": "#c0392b"}
    labels = list(sentiment_counts.keys())
    values = list(sentiment_counts.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=[colors.get(l, "#aaa") for l in labels],
        textfont_size=13,
    )])
    fig.update_layout(**base_layout(
        height=height,
        showlegend=True,
        margin=dict(t=10, b=10),
    ))
    return fig


def sentiment_bar(names: list, scores: list, height: int = 360) -> go.Figure:
    """Horizontal or vertical bar chart of sentiment scores per group."""
    bar_colors = ["#27ae60" if v >= 0.05 else "#c0392b" if v <= -0.05 else "#bdc3c7"
                  for v in scores]
    fig = go.Figure(go.Bar(
        x=names,
        y=scores,
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in scores],
        textposition="outside",
    ))
    fig.update_layout(**base_layout(
        height=height,
        yaxis=dict(range=[-1.1, 1.1], zeroline=True, zerolinecolor="#ddd"),
        margin=dict(t=40, b=10),
    ))
    return fig


def elbow_curve(k_values: list, inertias: list, elbow_k: int, height: int = 320) -> go.Figure:
    """Inertia vs K elbow curve with elbow point highlighted."""
    elbow_idx = k_values.index(elbow_k)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values, y=inertias,
        mode="lines+markers",
        line=dict(color="#2980b9", width=2),
        marker=dict(size=8, color="#2980b9"),
        name="Inertia",
    ))
    fig.add_trace(go.Scatter(
        x=[elbow_k], y=[inertias[elbow_idx]],
        mode="markers",
        marker=dict(size=14, color="#c0392b", symbol="star"),
        name=f"Elbow (K={elbow_k})",
    ))
    fig.update_layout(**base_layout(
        height=height,
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=10),
    ))
    return fig


def silhouette_bars(k_values: list, silhouettes: list, highlight_k: int, height: int = 320) -> go.Figure:
    """Bar chart of silhouette scores with best K highlighted."""
    colors = ["#27ae60" if k == highlight_k else "#bdc3c7" for k in k_values]
    fig = go.Figure(go.Bar(
        x=k_values,
        y=silhouettes,
        marker_color=colors,
        text=[f"{s:.3f}" for s in silhouettes],
        textposition="outside",
    ))
    fig.update_layout(**base_layout(
        height=height,
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Silhouette Score",
        yaxis=dict(range=[0, max(silhouettes) * 1.2]),
        margin=dict(t=10),
    ))
    return fig


def rating_histogram(ratings: pd.Series, height: int = 320) -> go.Figure:
    """Bar chart of star rating distribution."""
    counts = ratings.value_counts().sort_index()
    rating_colors = ["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=rating_colors[:len(counts)],
        text=counts.values,
        textposition="outside",
    ))
    fig.update_layout(**base_layout(
        height=height,
        xaxis=dict(title="Stars", tickmode="linear"),
        yaxis_title="Reviews",
        margin=dict(t=10),
    ))
    return fig


def topic_group_bars(df: pd.DataFrame, theme_cols: list, colors: list, height: int = 400) -> go.Figure:
    """Grouped bar chart — one group per company, one bar per topic."""
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row["name"],
            x=theme_cols,
            y=[row[tc] for tc in theme_cols],
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(**base_layout(
        barmode="group",
        height=height,
        legend=dict(orientation="h", y=-0.2),
        yaxis_title="Mention rate",
        margin=dict(t=10),
    ))
    return fig