"""Correctness tests for the MCDA scoring pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

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
            "q_climbing": [1, 5],
            "q_family": [1, 1],
        }
    )
    scenic = quality_score(parks, {"q_scenery": 1})
    assert scenic.iloc[0] > scenic.iloc[1]
    climb = quality_score(parks, {"q_climbing": 1})
    assert climb.iloc[1] > climb.iloc[0]


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


def test_caz_weight_penalises_caz_park(dataset):
    """Cranking the CAZ weight should not improve a CAZ park's rank."""
    caz_park = dataset.parks[dataset.parks["in_caz"]]["park_id"].iloc[0]
    low = score(dataset, {"caz": 0.0, "quality": 1.0})
    high = score(dataset, {"caz": 1.0})
    assert high.loc[caz_park, "rank"] >= low.loc[caz_park, "rank"]
    # A CAZ park can never be top when CAZ is the only thing that matters.
    assert high.loc[caz_park, "rank"] > 1


def test_criteria_table_has_all_columns(dataset):
    table = build_criteria_table(dataset)
    for c in CRITERIA:
        assert c in table.columns
    assert len(table) == len(dataset.parks)
