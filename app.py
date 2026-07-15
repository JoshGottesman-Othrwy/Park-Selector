"""Park Meetup Selector - a Streamlit MCDA tool for choosing a Bristol park.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import viz
from src.loading import DataValidationError, MODES, QUALITY_COLS, load_dataset
from src.scoring import CRITERIA, normalise_weights, score
from src.viz import CRITERION_LABELS, MODE_LABELS, QUALITY_LABELS

st.set_page_config(page_title="Park Meetup Selector", page_icon="🌳", layout="wide")

# Default criterion + quality sub-weights, and the persona presets that overwrite
# them. Presets set both the six criterion weights and the five quality sub-weights.
# "drive_no_caz" is driving while avoiding the Clean Air Zone (CAZ parks are N/A).
DEFAULT_CRITERION_WEIGHTS = {
    "quality": 0.30, "walk_time": 0.10, "drive_time": 0.15,
    "drive_no_caz": 0.10, "transit_time": 0.15, "parking": 0.20,
}
DEFAULT_QUALITY_WEIGHTS = {c: 1.0 for c in QUALITY_COLS}

_EQUAL_QUALITY = {"q_scenery": 1, "q_space": 1, "q_facilities": 1, "q_tree_cover": 1, "q_flatness": 1}

PRESETS: dict[str, dict] = {
    "Driver": {
        # Happy to drive into the CAZ; weights normal drive time + parking.
        "criterion": {"quality": 0.25, "walk_time": 0.0, "drive_time": 0.40,
                       "drive_no_caz": 0.0, "transit_time": 0.0, "parking": 0.35},
        "quality": _EQUAL_QUALITY,
    },
    "Driver · avoid CAZ": {
        # Won't enter the CAZ; uses drive_no_caz, so CAZ parks drop out (N/A).
        "criterion": {"quality": 0.25, "walk_time": 0.0, "drive_time": 0.0,
                       "drive_no_caz": 0.40, "transit_time": 0.0, "parking": 0.35},
        "quality": _EQUAL_QUALITY,
    },
    "Public transport": {
        "criterion": {"quality": 0.35, "walk_time": 0.10, "drive_time": 0.0,
                       "drive_no_caz": 0.0, "transit_time": 0.55, "parking": 0.0},
        "quality": _EQUAL_QUALITY,
    },
    "Acrobat": {
        # Cares most about park quality, and avoids the CAZ (weights
        # drive_no_caz instead of plain drive time, so CAZ parks drop out).
        "criterion": {"quality": 0.50, "walk_time": 0.05, "drive_time": 0.0,
                       "drive_no_caz": 0.20, "transit_time": 0.10, "parking": 0.15},
        # Flat, spacious, shaded ground for acro / tumbling.
        "quality": {"q_scenery": 1, "q_space": 2, "q_facilities": 1, "q_tree_cover": 2, "q_flatness": 3},
    },
    "Shade & scenery": {
        "criterion": {"quality": 0.50, "walk_time": 0.05, "drive_time": 0.25,
                       "drive_no_caz": 0.0, "transit_time": 0.10, "parking": 0.10},
        # Leafy, good-looking spots for a picnic in the shade.
        "quality": {"q_scenery": 3, "q_space": 1, "q_facilities": 1, "q_tree_cover": 3, "q_flatness": 1},
    },
    "Overall transport": {
        # A compromise across every way of getting there, with quality de-emphasised.
        "criterion": {"quality": 0.10, "walk_time": 0.15, "drive_time": 0.20,
                       "drive_no_caz": 0.15, "transit_time": 0.20, "parking": 0.20},
        "quality": _EQUAL_QUALITY,
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


def sidebar() -> tuple[dict[str, float], dict[str, float]]:
    """Render sidebar controls; return (weights, quality_weights)."""
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
    st.sidebar.caption(
        "“Drive (no CAZ)” avoids the Clean Air Zone, so parks inside the CAZ "
        "are treated as unreachable (N/A) for that criterion."
    )

    with st.sidebar.expander("Quality sub-weights"):
        quality_weights = {}
        for c in QUALITY_COLS:
            quality_weights[c] = st.slider(
                QUALITY_LABELS[c], 0.0, 4.0, key=f"qw_{c}", step=1.0
            )

    return weights, quality_weights


def main() -> None:
    try:
        dataset = load_dataset()
    except (FileNotFoundError, DataValidationError) as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    _init_state(dataset.default_weights)
    weights, quality_weights = sidebar()

    scored = score(dataset, weights, quality_weights)
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
            "Mode", MODES, horizontal=True, key="heatmap_mode",
            format_func=lambda m: MODE_LABELS[m],
        )
        st.plotly_chart(viz.travel_heatmap(dataset, mode), width="stretch")
        if mode == "drive_no_caz":
            st.caption("Blank cells = parks inside the CAZ (N/A when avoiding the zone).")

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
        "drive_no_caz": "Drive no-CAZ (min)", "transit_time": "Transit (min)",
        "parking": "Parking penalty",
    }
    df = scored.reset_index()[list(cols)].rename(columns=cols).round(1).sort_values("Rank")
    # Show CAZ parks (NaN drive_no_caz) explicitly as N/A.
    df["Drive no-CAZ (min)"] = df["Drive no-CAZ (min)"].map(
        lambda v: "N/A" if pd.isna(v) else f"{v:.1f}"
    )
    return df


if __name__ == "__main__":
    main()
