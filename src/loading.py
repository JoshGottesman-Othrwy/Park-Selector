"""Load and validate the park-selector CSV data.

Everything the app needs comes from three CSVs under ``data/`` plus a defaults
file. Keeping loading + validation here means the rest of the app can assume the
DataFrames are well-formed, and it's the single place to swap CSV for SQLite /
Parquet later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MODES = ["walk", "drive", "transit"]
QUALITY_COLS = ["q_scenery", "q_space", "q_facilities", "q_climbing", "q_family"]
PARKING_TYPES = {"free", "paid", "street", "none"}
PARKING_SPACES = {"plenty", "limited", "scarce"}

PARK_COLS = [
    "park_id", "name", "lat", "lon", *QUALITY_COLS,
    "parking_type", "parking_cost_per_day", "parking_spaces", "in_caz", "notes",
]
ORIGIN_COLS = ["origin_id", "name", "lat", "lon", "weight"]
TRAVEL_COLS = ["origin_id", "park_id", "mode", "duration_min"]


@dataclass
class Dataset:
    """The validated, ready-to-use data bundle."""
    parks: pd.DataFrame
    origins: pd.DataFrame
    travel: pd.DataFrame
    default_weights: dict[str, float]


class DataValidationError(ValueError):
    """Raised when a CSV is missing columns or contains inconsistent data."""


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name} is missing columns: {missing}")


def _coerce_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    )


def load_dataset(data_dir: Path | str = DATA_DIR) -> Dataset:
    """Read the CSVs, validate them, and return a :class:`Dataset`.

    Raises :class:`DataValidationError` on any structural problem so the app can
    surface a clear message instead of failing deep inside a chart.
    """
    data_dir = Path(data_dir)

    parks = pd.read_csv(data_dir / "parks.csv")
    origins = pd.read_csv(data_dir / "origins.csv")
    travel = pd.read_csv(data_dir / "travel_times.csv")

    _require_columns(parks, PARK_COLS, "parks.csv")
    _require_columns(origins, ORIGIN_COLS, "origins.csv")
    _require_columns(travel, TRAVEL_COLS, "travel_times.csv")

    # --- types -------------------------------------------------------------
    parks["in_caz"] = _coerce_bool(parks["in_caz"])
    for col in QUALITY_COLS:
        parks[col] = pd.to_numeric(parks[col])
    parks["parking_cost_per_day"] = pd.to_numeric(parks["parking_cost_per_day"])
    origins["weight"] = pd.to_numeric(origins["weight"])
    travel["duration_min"] = pd.to_numeric(travel["duration_min"])

    # --- value checks ------------------------------------------------------
    if parks["park_id"].duplicated().any():
        raise DataValidationError("Duplicate park_id in parks.csv")
    if origins["origin_id"].duplicated().any():
        raise DataValidationError("Duplicate origin_id in origins.csv")

    for col in QUALITY_COLS:
        bad = parks[(parks[col] < 1) | (parks[col] > 5)]
        if not bad.empty:
            raise DataValidationError(f"{col} must be 1-5 (parks: {list(bad['park_id'])})")

    bad_type = set(parks["parking_type"]) - PARKING_TYPES
    if bad_type:
        raise DataValidationError(f"Unknown parking_type values: {bad_type}")
    bad_spaces = set(parks["parking_spaces"]) - PARKING_SPACES
    if bad_spaces:
        raise DataValidationError(f"Unknown parking_spaces values: {bad_spaces}")
    bad_mode = set(travel["mode"]) - set(MODES)
    if bad_mode:
        raise DataValidationError(f"Unknown travel mode values: {bad_mode}")

    # --- referential integrity + completeness ------------------------------
    park_ids = set(parks["park_id"])
    origin_ids = set(origins["origin_id"])
    if not set(travel["park_id"]).issubset(park_ids):
        raise DataValidationError("travel_times.csv references unknown park_id")
    if not set(travel["origin_id"]).issubset(origin_ids):
        raise DataValidationError("travel_times.csv references unknown origin_id")

    expected = len(park_ids) * len(origin_ids) * len(MODES)
    actual = len(travel.drop_duplicates(["origin_id", "park_id", "mode"]))
    if actual != expected:
        raise DataValidationError(
            f"travel_times.csv should have {expected} origin×park×mode rows, found {actual}. "
            "Every park needs walk/drive/transit times from every origin."
        )

    default_weights = _load_default_weights(data_dir)
    return Dataset(parks=parks, origins=origins, travel=travel, default_weights=default_weights)


def _load_default_weights(data_dir: Path) -> dict[str, float]:
    path = data_dir / "weights_default.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["criterion"], pd.to_numeric(df["weight"])))
