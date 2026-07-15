"""Correctness tests for the MCDA scoring pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loading import Dataset, load_dataset
from src.scoring import (
    CRITERIA,
    build_criteria_table,
    normalise,
    normalise_weights,
    parking_penalty,
    quality_score,
    score,
)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return load_dataset()


# --- normalise ------------------------------------------------------------- #
def test_normalise_benefit_direction():
    s = pd.Series([1.0, 3.0, 5.0])
    n = normalise(s, benefit=True)
    assert n.iloc[0] == 0.0 and n.iloc[-1] == 1.0
    assert n.is_monotonic_increasing


def test_normalise_cost_direction():
    s = pd.Series([10.0, 20.0, 30.0])  # lower is better
    n = normalise(s, benefit=False)
    assert n.iloc[0] == 1.0 and n.iloc[-1] == 0.0


def test_normalise_bounds():
    s = pd.Series([2.0, 7.0, 4.0, 9.0])
    for benefit in (True, False):
        n = normalise(s, benefit=benefit)
        assert n.min() >= 0.0 and n.max() <= 1.0


def test_normalise_all_equal_scores_one():
    # A criterion that can't discriminate must not penalise anyone.
    s = pd.Series([5.0, 5.0, 5.0])
    assert (normalise(s, benefit=True) == 1.0).all()
    assert (normalise(s, benefit=False) == 1.0).all()


def test_normalise_nan_scores_zero():
    # N/A (unreachable) scores worst; reachable values still span 0..1.
    s = pd.Series([10.0, np.nan, 20.0])
    n = normalise(s, benefit=False)  # cost: lower is better
    assert n.iloc[1] == 0.0          # NaN -> worst
    assert n.iloc[0] == 1.0          # 10 min is the best reachable
    assert n.iloc[2] == 0.0


def test_normalise_nan_scores_zero_even_when_others_equal():
    s = pd.Series([7.0, 7.0, np.nan])
    n = normalise(s, benefit=True)
    assert (n.iloc[:2] == 1.0).all()
    assert n.iloc[2] == 0.0


# --- weights --------------------------------------------------------------- #
def test_weights_renormalise_to_one():
    w = normalise_weights({"quality": 2, "drive_time": 2})
    assert pytest.approx(sum(w.values())) == 1.0
    assert pytest.approx(w["quality"]) == 0.5


def test_all_zero_weights_fall_back_to_equal():
    w = normalise_weights({c: 0 for c in CRITERIA})
    assert pytest.approx(sum(w.values())) == 1.0
    assert all(pytest.approx(v) == 1 / len(CRITERIA) for v in w.values())


def test_negative_weights_clamped():
    w = normalise_weights({"quality": -5, "parking": 1})
    assert w["quality"] == 0.0
    assert pytest.approx(w["parking"]) == 1.0


# --- parking penalty ------------------------------------------------------- #
def test_parking_penalty_free_plenty_is_best():
    parks = pd.DataFrame(
        {
            "parking_type": ["free", "none"],
            "parking_spaces": ["plenty", "scarce"],
            "parking_cost_per_day": [0.0, 0.0],
        }
    )
    pen = parking_penalty(parks)
    assert pen.iloc[0] == 0.0            # free + plenty + £0 = no friction
    assert pen.iloc[1] > pen.iloc[0]    # none + scarce is clearly worse
    assert (pen >= 0).all() and (pen <= 1).all()


def test_parking_penalty_reaches_max_when_expensive():
    parks = pd.DataFrame(
        {
            "parking_type": ["none"],
            "parking_spaces": ["scarce"],
            "parking_cost_per_day": [10.0],
        }
    )
    assert parking_penalty(parks).iloc[0] == 1.0


# --- quality --------------------------------------------------------------- #
def test_quality_weight_shifts_ranking():
    parks = pd.DataFrame(
        {
            "q_scenery": [5, 1],
            "q_space": [1, 1],
            "q_facilities": [1, 1],
            "q_tree_cover": [1, 5],
            "q_flatness": [1, 1],
        }
    )
    scenic = quality_score(parks, {"q_scenery": 1})
    assert scenic.iloc[0] > scenic.iloc[1]
    shady = quality_score(parks, {"q_tree_cover": 1})
    assert shady.iloc[1] > shady.iloc[0]


# --- composite ------------------------------------------------------------- #
def test_score_equals_weighted_sum_of_norms(dataset):
    weights = dataset.default_weights
    df = score(dataset, weights)
    w = normalise_weights(weights)
    expected = sum(df[f"norm_{c}"] * w[c] for c in CRITERIA) * 100
    pd.testing.assert_series_equal(df["score"], expected.round(1), check_names=False)


def test_score_bounds_and_rank(dataset):
    df = score(dataset, dataset.default_weights)
    assert df["score"].between(0, 100).all()
    assert df["rank"].min() == 1
    # sorted best-first
    assert df["score"].is_monotonic_decreasing


def test_drive_no_caz_is_na_for_caz_parks(dataset):
    """CAZ parks are unreachable when avoiding the zone -> NaN drive_no_caz."""
    table = build_criteria_table(dataset)
    caz_ids = list(dataset.parks[dataset.parks["in_caz"]]["park_id"])
    non_caz_ids = list(dataset.parks[~dataset.parks["in_caz"]]["park_id"])
    assert caz_ids and non_caz_ids, "fixture needs both CAZ and non-CAZ parks"
    assert table.loc[caz_ids, "drive_no_caz"].isna().all()
    assert table.loc[non_caz_ids, "drive_no_caz"].notna().all()


def test_drive_no_caz_weight_excludes_caz_parks(dataset):
    """When only drive_no_caz matters, CAZ parks lose out to reachable ones."""
    caz_ids = set(dataset.parks[dataset.parks["in_caz"]]["park_id"])
    non_caz_ids = set(dataset.parks["park_id"]) - caz_ids

    scored = score(dataset, {"drive_no_caz": 1.0})
    # CAZ parks get the worst possible normalised score on this criterion...
    assert (scored.loc[list(caz_ids), "norm_drive_no_caz"] == 0.0).all()
    # ...so none can be the top pick. They sit in the bottom group: no CAZ park
    # beats the worst reachable park (under min-max the slowest reachable park
    # also scores 0, so a tie there is expected), and the best reachable park
    # clearly outranks every CAZ park.
    assert scored.index[0] not in caz_ids
    worst_non_caz_rank = scored.loc[list(non_caz_ids), "rank"].max()
    best_caz_rank = scored.loc[list(caz_ids), "rank"].min()
    best_non_caz_rank = scored.loc[list(non_caz_ids), "rank"].min()
    assert best_caz_rank >= worst_non_caz_rank
    assert best_non_caz_rank < best_caz_rank


def test_criteria_table_has_all_columns(dataset):
    table = build_criteria_table(dataset)
    for c in CRITERIA:
        assert c in table.columns
    assert len(table) == len(dataset.parks)
