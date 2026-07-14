"""MCDA scoring: derived matrices, normalisation and the composite ranking.

Everything here is pure (DataFrame in -> DataFrame out) so it is easy to unit
test and cheap to recompute on every slider change in the app.

Six criteria per park drive the ranking:

    quality       - weighted mean of the 1-5 subjective sub-scores (benefit)
    walk_time     - origin-weighted mean walking minutes            (cost)
    drive_time    - origin-weighted mean driving minutes            (cost)
    transit_time  - origin-weighted mean public-transport minutes   (cost)
    parking       - 0-1 parking-friction penalty                    (cost)
    caz           - Clean Air Zone charge in GBP (drive only)        (cost)

"benefit" criteria score higher when the raw value is higher; "cost" criteria
score higher when the raw value is lower. Normalisation maps every criterion to
0-1 where 1 is always best.
"""
from __future__ import annotations

import pandas as pd

from .loading import MODES, QUALITY_COLS, Dataset

CRITERIA = ["quality", "walk_time", "drive_time", "transit_time", "parking", "caz"]
# Everything except quality is a cost criterion (lower is better).
COST_CRITERIA = {"walk_time", "drive_time", "transit_time", "parking", "caz"}

MODE_TO_CRITERION = {"walk": "walk_time", "drive": "drive_time", "transit": "transit_time"}

DEFAULT_CAZ_CHARGE = 9.0  # Bristol CAZ D, non-compliant private car, GBP/day

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

    Returns a DataFrame indexed by ``park_id`` with columns
    ``walk_time``/``drive_time``/``transit_time`` (minutes).
    """
    travel = dataset.travel.merge(
        dataset.origins[["origin_id", "weight"]], on="origin_id", how="left"
    )

    def _wmean(group: pd.DataFrame) -> float:
        w = group["weight"].sum()
        if w == 0:
            return group["duration_min"].mean()
        return (group["duration_min"] * group["weight"]).sum() / w

    agg = (
        travel.groupby(["park_id", "mode"], group_keys=False)
        .apply(_wmean, include_groups=False)
        .rename("t")
        .reset_index()
    )
    wide = agg.pivot(index="park_id", columns="mode", values="t")
    return wide.rename(columns=MODE_TO_CRITERION)[list(MODE_TO_CRITERION.values())]


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
    caz_charge: float = DEFAULT_CAZ_CHARGE,
) -> pd.DataFrame:
    """Assemble the raw per-park criteria table (index = park_id).

    Columns: park name + the six raw criteria values.
    """
    parks = dataset.parks.set_index("park_id")
    table = pd.DataFrame(index=parks.index)
    table["name"] = parks["name"]
    table["quality"] = quality_score(dataset.parks.set_index("park_id"), quality_weights)

    travel = aggregate_travel(dataset)
    table = table.join(travel)

    table["parking"] = parking_penalty(parks)
    table["caz"] = parks["in_caz"].astype(float) * caz_charge
    return table


def normalise(series: pd.Series, benefit: bool) -> pd.Series:
    """Map a criterion to 0-1 where 1 is best.

    ``benefit=True``  -> higher raw value is better.
    ``benefit=False`` -> lower raw value is better (cost).
    When all values are equal, everything scores 1 (the criterion can't
    discriminate, so it shouldn't penalise anyone).
    """
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    if benefit:
        return (series - lo) / (hi - lo)
    return (hi - series) / (hi - lo)


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
    caz_charge: float = DEFAULT_CAZ_CHARGE,
) -> pd.DataFrame:
    """Full scoring pipeline -> tidy DataFrame every view consumes.

    The returned frame (index = park_id) contains, per park:
      * ``name`` and the six raw criteria values
      * a ``norm_<criterion>`` column (0-1, 1 = best) for each criterion
      * ``score`` (0-100 composite) and integer ``rank`` (1 = best)
    """
    table = build_criteria_table(dataset, quality_weights, caz_charge)
    w = normalise_weights(weights)

    composite = pd.Series(0.0, index=table.index)
    for c in CRITERIA:
        norm = normalise(table[c], benefit=(c not in COST_CRITERIA))
        table[f"norm_{c}"] = norm
        composite += norm * w[c]

    table["score"] = (composite * 100).round(1)
    table["rank"] = table["score"].rank(ascending=False, method="min").astype(int)
    return table.sort_values("score", ascending=False)
