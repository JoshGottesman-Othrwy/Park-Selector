"""MCDA scoring: derived matrices, normalisation and the composite ranking.

Everything here is pure (DataFrame in -> DataFrame out) so it is easy to unit
test and cheap to recompute on every slider change in the app.

Six criteria per park drive the ranking:

    quality       - weighted mean of the 1-5 subjective sub-scores (benefit)
    walk_time     - origin-weighted mean walking minutes            (cost)
    drive_time    - origin-weighted mean driving minutes            (cost)
    drive_no_caz  - driving while avoiding the Clean Air Zone;      (cost)
                    N/A for parks inside the CAZ (unreachable
                    without paying), so those parks score worst
    transit_time  - origin-weighted mean public-transport minutes   (cost)
    parking       - 0-1 parking-friction penalty                    (cost)

"benefit" criteria score higher when the raw value is higher; "cost" criteria
score higher when the raw value is lower. Normalisation maps every criterion to
0-1 where 1 is always best. N/A (NaN) values score 0 (worst / unreachable).
"""
from __future__ import annotations

import pandas as pd

from .loading import MODES, QUALITY_COLS, Dataset

CRITERIA = ["quality", "walk_time", "drive_time", "drive_no_caz", "transit_time", "parking"]
# Everything except quality is a cost criterion (lower is better).
COST_CRITERIA = {"walk_time", "drive_time", "drive_no_caz", "transit_time", "parking"}

MODE_TO_CRITERION = {
    "walk": "walk_time",
    "drive": "drive_time",
    "transit": "transit_time",
    "drive_no_caz": "drive_no_caz",
}

# Interpretable parking-friction components (each 0 = best, 1 = worst).
_PARKING_TYPE_PENALTY = {"free": 0.0, "street": 0.3, "paid": 0.5, "none": 1.0}
_PARKING_SPACE_PENALTY = {"plenty": 0.0, "limited": 0.5, "scarce": 1.0}
_PARKING_COST_CAP = 10.0  # GBP/day treated as maximum friction


# --------------------------------------------------------------------------- #
# Derived matrices
# --------------------------------------------------------------------------- #
def travel_matrix(dataset: Dataset, mode: str) -> pd.DataFrame:
    """Return an origin (rows) x park (cols) matrix of durations for one mode.

    Used directly by the heatmap. Column/row labels use display names.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    t = dataset.travel[dataset.travel["mode"] == mode]
    mat = t.pivot(index="origin_id", columns="park_id", values="duration_min")
    mat = mat.rename(index=dataset.origins.set_index("origin_id")["name"])
    mat = mat.rename(columns=dataset.parks.set_index("park_id")["name"])
    return mat


def aggregate_travel(dataset: Dataset) -> pd.DataFrame:
    """Origin-weighted mean travel time per park and mode.

    Returns a DataFrame indexed by ``park_id`` with a column per mode criterion
    (``walk_time``/``drive_time``/``drive_no_caz``/``transit_time``), in minutes.
    A park with no reachable value for a mode (e.g. ``drive_no_caz`` for a CAZ
    park) comes back as NaN.
    """
    travel = dataset.travel.merge(
        dataset.origins[["origin_id", "weight"]], on="origin_id", how="left"
    )

    def _wmean(group: pd.DataFrame) -> float:
        g = group.dropna(subset=["duration_min"])
        if g.empty:  # nothing reachable -> N/A
            return float("nan")
        w = g["weight"].sum()
        if w == 0:
            return g["duration_min"].mean()
        return (g["duration_min"] * g["weight"]).sum() / w

    agg = (
        travel.groupby(["park_id", "mode"], group_keys=False)
        .apply(_wmean, include_groups=False)
        .rename("t")
        .reset_index()
    )
    wide = agg.pivot(index="park_id", columns="mode", values="t")
    wide = wide.rename(columns=MODE_TO_CRITERION)
    return wide.reindex(columns=list(MODE_TO_CRITERION.values()))


def parking_penalty(parks: pd.DataFrame) -> pd.Series:
    """0-1 parking-friction penalty per park (0 = easy free parking, 1 = none)."""
    type_pen = parks["parking_type"].map(_PARKING_TYPE_PENALTY).fillna(0.5)
    space_pen = parks["parking_spaces"].map(_PARKING_SPACE_PENALTY).fillna(0.5)
    cost_pen = (parks["parking_cost_per_day"] / _PARKING_COST_CAP).clip(0, 1)
    penalty = 0.5 * type_pen + 0.3 * space_pen + 0.2 * cost_pen
    return penalty.clip(0, 1).rename("parking")


def quality_score(parks: pd.DataFrame, quality_weights: dict[str, float] | None = None) -> pd.Series:
    """Weighted mean of the five 1-5 quality sub-scores, per park.

    ``quality_weights`` maps sub-score column -> weight; missing/empty means an
    equal weighting. Weights need not sum to 1 (they are renormalised).
    """
    if not quality_weights:
        quality_weights = {c: 1.0 for c in QUALITY_COLS}
    weights = pd.Series({c: quality_weights.get(c, 0.0) for c in QUALITY_COLS})
    total = weights.sum()
    if total == 0:
        weights = pd.Series({c: 1.0 for c in QUALITY_COLS})
        total = weights.sum()
    return (parks[QUALITY_COLS] * weights).sum(axis=1) / total


# --------------------------------------------------------------------------- #
# Criteria table + normalisation + composite
# --------------------------------------------------------------------------- #
def build_criteria_table(
    dataset: Dataset,
    quality_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assemble the raw per-park criteria table (index = park_id).

    Columns: park name + the raw criteria values. ``drive_no_caz`` is NaN for
    parks inside the CAZ.
    """
    parks = dataset.parks.set_index("park_id")
    table = pd.DataFrame(index=parks.index)
    table["name"] = parks["name"]
    table["quality"] = quality_score(parks, quality_weights)

    travel = aggregate_travel(dataset)
    table = table.join(travel)

    table["parking"] = parking_penalty(parks)
    return table


def normalise(series: pd.Series, benefit: bool) -> pd.Series:
    """Map a criterion to 0-1 where 1 is best.

    ``benefit=True``  -> higher raw value is better.
    ``benefit=False`` -> lower raw value is better (cost).
    When all (non-NaN) values are equal, everything scores 1 (the criterion
    can't discriminate, so it shouldn't penalise anyone). NaN values are treated
    as unreachable / worst and score 0.
    """
    result = pd.Series(0.0, index=series.index)  # NaN -> 0 (worst / unreachable)
    valid = series.dropna()
    if valid.empty:
        return result
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        result.loc[valid.index] = 1.0
    elif benefit:
        result.loc[valid.index] = (valid - lo) / (hi - lo)
    else:
        result.loc[valid.index] = (hi - valid) / (hi - lo)
    return result


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Renormalise criterion weights to sum to 1 (all-zero -> equal weights)."""
    picked = {c: max(0.0, float(weights.get(c, 0.0))) for c in CRITERIA}
    total = sum(picked.values())
    if total == 0:
        return {c: 1.0 / len(CRITERIA) for c in CRITERIA}
    return {c: v / total for c, v in picked.items()}


def score(
    dataset: Dataset,
    weights: dict[str, float],
    quality_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Full scoring pipeline -> tidy DataFrame every view consumes.

    The returned frame (index = park_id) contains, per park:
      * ``name`` and the raw criteria values
      * a ``norm_<criterion>`` column (0-1, 1 = best) for each criterion
      * ``score`` (0-100 composite) and integer ``rank`` (1 = best)
    """
    table = build_criteria_table(dataset, quality_weights)
    w = normalise_weights(weights)

    composite = pd.Series(0.0, index=table.index)
    for c in CRITERIA:
        norm = normalise(table[c], benefit=(c not in COST_CRITERIA))
        table[f"norm_{c}"] = norm
        composite += norm * w[c]

    table["score"] = (composite * 100).round(1)
    table["rank"] = table["score"].rank(ascending=False, method="min").astype(int)
    return table.sort_values("score", ascending=False)
