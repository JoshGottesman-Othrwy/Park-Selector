"""Park Meetup Selector - a Streamlit MCDA tool for choosing a Bristol park.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src import viz
from src.loading import DataValidationError, QUALITY_COLS, load_dataset
from src.scoring import CRITERIA, DEFAULT_CAZ_CHARGE, normalise_weights, score
from src.viz import CRITERION_LABELS, QUALITY_LABELS

st.set_page_config(page_title="Park Meetup Selector", page_icon="🌳", layout="wide")

# Default criterion + quality sub-weights, and the persona presets that overwrite
# them. Presets set both the six criterion weights and the five quality sub-weights.
DEFAULT_CRITERION_WEIGHTS = {
    "quality": 0.30, "walk_time": 0.10, "drive_time": 0.20,
    "transit_time": 0.15, "parking": 0.15, "caz": 0.10,
}
DEFAULT_QUALITY_WEIGHTS = {c: 1.0 for c in QUALITY_COLS}

PRESETS: dict[str, dict] = {
    "Driver": {
        "criterion": {"quality": 0.25, "walk_time": 0.0, "drive_time": 0.30,
                       "transit_time": 0.0, "parking": 0.25, "caz": 0.20},
        "quality": {"q_scenery": 1, "q_space": 1, "q_facilities": 1, "q_climbing": 1, "q_family": 1},
    },
    "Public transport": {
        "criterion": {"quality": 0.30, "walk_time": 0.10, "drive_time": 0.0,
                       "transit_time": 0.45, "parking": 0.0, "caz": 0.15},
        "quality": {"q_scenery": 1, "q_space": 1, "q_facilities": 1, "q_climbing": 1, "q_family": 1},
    },
    "Family": {
        "criterion": {"quality": 0.40, "walk_time": 0.10, "drive_time": 0.20,
                       "transit_time": 0.10, "parking": 0.15, "caz": 0.05},
        "quality": {"q_scenery": 1, "q_space": 2, "q_facilities": 2, "q_climbing": 0, "q_family": 3},
    },
    "Climber": {
        "criterion": {"quality": 0.45, "walk_time": 0.05, "drive_time": 0.25,
                       "transit_time": 0.10, "parking": 0.15, "caz": 0.0},
        "quality": {"q_scenery": 1, "q_space": 1, "q_facilities": 1, "q_climbing": 4, "q_family": 0},
    },
}


def _init_state(default_weights: dict[str, float]) -> None:
    """Seed session_state with default weights on first load."""
    base = {**DEFAULT_CRITERION_WEIGHTS, **default_weights}
    for c in CRITERIA:
        st.session_state.setdefault(f"w_{c}", float(base.get(c, 0.1)))
    for c in QUALITY_COLS:
        st.session_state.setdefault(f"qw_{c}", float(DEFAULT_QUALITY_WEIGHTS[c]))


def _apply_preset(name: str) -> None:
    preset = PRESETS[name]
    for c in CRITERIA:
        st.session_state[f"w_{c}"] = float(preset["criterion"][c])
    for c in QUALITY_COLS:
        st.session_state[f"qw_{c}"] = float(preset["quality"][c])


def sidebar() -> tuple[dict[str, float], dict[str, float], str, float]:
    """Render sidebar controls; return (weights, quality_weights, heatmap_mode, caz_charge)."""
    st.sidebar.title("🌳 Park Selector")
    st.sidebar.caption("Weight what matters to you — the ranking updates live.")

    st.sidebar.subheader("Presets")
    cols = st.sidebar.columns(2)
    for i, name in enumerate(PRESETS):
        if cols[i % 2].button(name, width="stretch"):
            _apply_preset(name)
            st.rerun()

    st.sidebar.subheader("Criterion weights")
    weights = {}
    for c in CRITERIA:
        weights[c] = st.sidebar.slider(
            CRITERION_LABELS[c], 0.0, 1.0, key=f"w_{c}", step=0.05
        )

    with st.sidebar.expander("Quality sub-weights"):
        quality_weights = {}
        for c in QUALITY_COLS:
            quality_weights[c] = st.slider(
                QUALITY_LABELS[c], 0.0, 4.0, key=f"qw_{c}", step=1.0
            )

    with st.sidebar.expander("Assumptions"):
        caz_charge = st.number_input(
            "CAZ daily charge (£)", min_value=0.0, value=DEFAULT_CAZ_CHARGE, step=1.0
        )
        st.caption("Bristol Clean Air Zone D — non-compliant private car.")

    heatmap_mode = st.session_state.get("heatmap_mode", "drive")
    return weights, quality_weights, heatmap_mode, caz_charge


def main() -> None:
    try:
        dataset = load_dataset()
    except (FileNotFoundError, DataValidationError) as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    _init_state(dataset.default_weights)
    weights, quality_weights, _, caz_charge = sidebar()

    scored = score(dataset, weights, quality_weights, caz_charge)
    norm_weights = normalise_weights(weights)
    top = scored.iloc[0]

    st.title("Park Meetup Selector — Bristol")
    st.markdown(
        f"**Current top pick: {top['name']}** "
        f"(score {top['score']:.0f}/100) · "
        f"drive ~{top['drive_time']:.0f} min · transit ~{top['transit_time']:.0f} min"
    )

    tab_map, tab_compare, tab_rank = st.tabs(["🗺️ Map", "📊 Compare", "🏆 Rank"])

    with tab_map:
        st.subheader("Where the parks are")
        st.caption("Green = higher composite score. Blue markers are your origin suburbs.")
        st.pydeck_chart(viz.park_map(scored, dataset))
        if top["name"]:
            st.info(
                f"Top pick **{top['name']}** — {dataset.parks.set_index('name').loc[top['name'], 'notes']}"
            )

    with tab_compare:
        st.subheader("Travel-time heatmap")
        mode = st.radio(
            "Mode", ["walk", "drive", "transit"], horizontal=True, key="heatmap_mode",
            format_func=str.title,
        )
        st.plotly_chart(viz.travel_heatmap(dataset, mode), width="stretch")

        st.subheader("Park quality profiles")
        default_sel = list(scored.index[:3])
        chosen = st.multiselect(
            "Compare parks",
            options=list(dataset.parks["park_id"]),
            default=default_sel,
            format_func=lambda pid: dataset.parks.set_index("park_id").loc[pid, "name"],
        )
        if chosen:
            st.plotly_chart(viz.quality_radar(dataset, chosen), width="stretch")

        st.subheader("All metrics")
        st.caption("Raw values plus normalised 0–1 scores (1 = best). Sortable.")
        st.dataframe(_metrics_table(scored), width="stretch", hide_index=True)

    with tab_rank:
        st.subheader("Ranking")
        st.plotly_chart(viz.ranked_bar(scored), width="stretch")
        st.subheader("Why — weighted contribution by criterion")
        st.caption("How much each criterion adds to every park's score, given your weights.")
        st.plotly_chart(
            viz.score_breakdown(scored, norm_weights), width="stretch"
        )


def _metrics_table(scored):
    """Tidy the scored frame into a readable, sortable table."""
    cols = {
        "rank": "Rank", "name": "Park", "score": "Score",
        "quality": "Quality", "walk_time": "Walk (min)", "drive_time": "Drive (min)",
        "transit_time": "Transit (min)", "parking": "Parking penalty", "caz": "CAZ (£)",
    }
    df = scored.reset_index()[list(cols)].rename(columns=cols)
    return df.round(1).sort_values("Rank")


if __name__ == "__main__":
    main()
