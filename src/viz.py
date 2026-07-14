"""Chart and map builders for the Streamlit app.

Charts use Plotly (interactive, native to Streamlit). The map uses pydeck, which
ships with Streamlit and needs no extra JS or API key. All builders take the
scored DataFrame from :mod:`src.scoring` and return a figure/deck object; they do
no Streamlit rendering themselves so they stay testable.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from .loading import QUALITY_COLS, Dataset
from .scoring import CRITERIA, travel_matrix

# One shared colour scale across every view: red (worst) -> green (best).
SCORE_SCALE = "RdYlGn"

QUALITY_LABELS = {
    "q_scenery": "Scenery",
    "q_space": "Space",
    "q_facilities": "Facilities",
    "q_climbing": "Climbing",
    "q_family": "Family",
}
CRITERION_LABELS = {
    "quality": "Quality",
    "walk_time": "Walk time",
    "drive_time": "Drive time",
    "transit_time": "Transit time",
    "parking": "Parking",
    "caz": "CAZ charge",
}


def _score_to_rgb(score: float, lo: float, hi: float) -> list[int]:
    """Map a composite score to a red->green RGB triple for the map markers."""
    frac = 0.5 if hi == lo else (score - lo) / (hi - lo)
    red = int(220 * (1 - frac))
    green = int(180 * frac)
    return [red, green, 60]


# --------------------------------------------------------------------------- #
# Rank view
# --------------------------------------------------------------------------- #
def ranked_bar(scored: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of parks by composite score, best at the top."""
    df = scored.sort_values("score")
    fig = px.bar(
        df,
        x="score",
        y="name",
        orientation="h",
        color="score",
        color_continuous_scale=SCORE_SCALE,
        range_color=(0, 100),
        text="score",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        xaxis_title="Composite score",
        yaxis_title=None,
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(260, 44 * len(df)),
    )
    fig.update_xaxes(range=[0, 105])
    return fig


def score_breakdown(scored: pd.DataFrame, weights: dict[str, float]) -> go.Figure:
    """Stacked bar showing each criterion's weighted contribution per park."""
    df = scored.sort_values("score", ascending=False)
    fig = go.Figure()
    for c in CRITERIA:
        contribution = df[f"norm_{c}"] * weights.get(c, 0.0) * 100
        fig.add_bar(name=CRITERION_LABELS[c], x=df["name"], y=contribution)
    fig.update_layout(
        barmode="stack",
        yaxis_title="Weighted contribution",
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Criterion",
        height=420,
    )
    return fig


# --------------------------------------------------------------------------- #
# Compare view
# --------------------------------------------------------------------------- #
def travel_heatmap(dataset: Dataset, mode: str) -> go.Figure:
    """Origin (rows) x park (cols) travel-time heatmap for one mode."""
    mat = travel_matrix(dataset, mode)
    fig = px.imshow(
        mat,
        color_continuous_scale="RdYlGn_r",  # reversed: low time = green = good
        aspect="auto",
        text_auto=".0f",
        labels=dict(color="Minutes"),
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
    )
    fig.update_xaxes(tickangle=-40)
    return fig


def quality_radar(dataset: Dataset, park_ids: list[str]) -> go.Figure:
    """Overlay the five quality sub-scores for the selected parks."""
    parks = dataset.parks.set_index("park_id")
    axes = [QUALITY_LABELS[c] for c in QUALITY_COLS]
    fig = go.Figure()
    for pid in park_ids:
        if pid not in parks.index:
            continue
        row = parks.loc[pid]
        values = [row[c] for c in QUALITY_COLS]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=axes + [axes[0]],
                fill="toself",
                name=row["name"],
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        margin=dict(l=30, r=30, t=30, b=30),
        height=420,
    )
    return fig


# --------------------------------------------------------------------------- #
# Map view
# --------------------------------------------------------------------------- #
def park_map(scored: pd.DataFrame, dataset: Dataset) -> pdk.Deck:
    """Interactive map: parks coloured/sized by score, origins as markers."""
    parks = dataset.parks.set_index("park_id")
    df = scored.join(parks[["lat", "lon", "in_caz"]])
    lo, hi = df["score"].min(), df["score"].max()
    df = df.assign(
        color=df["score"].map(lambda s: _score_to_rgb(s, lo, hi)),
        radius=df["score"].map(lambda s: 120 + 3.2 * s),
        caz_flag=df["in_caz"].map({True: "  ⚠ in CAZ", False: ""}),
    )

    parks_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.85,
        stroked=True,
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
    )
    origins_layer = pdk.Layer(
        "ScatterplotLayer",
        data=dataset.origins,
        get_position="[lon, lat]",
        get_fill_color=[30, 90, 200],
        get_radius=180,
        pickable=True,
        opacity=0.9,
    )

    view = pdk.ViewState(
        latitude=float(pd.concat([df["lat"], dataset.origins["lat"]]).mean()),
        longitude=float(pd.concat([df["lon"], dataset.origins["lon"]]).mean()),
        zoom=11,
    )
    tooltip = {
        "html": "<b>{name}</b>{caz_flag}<br/>Score: {score}",
        "style": {"backgroundColor": "#222", "color": "white"},
    }
    return pdk.Deck(
        layers=[parks_layer, origins_layer],
        initial_view_state=view,
        map_style=None,  # no external tile provider / API key needed
        tooltip=tooltip,
    )
