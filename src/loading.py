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

# Core modes are reachable for every park and must be fully populated.
CORE_MODES = ["walk", "drive", "transit"]
# "drive_no_caz" is driving while avoiding the Clean Air Zone: it is N/A (blank)
# for parks inside the CAZ, so it is validated separately from the core modes.
MODES = [*CORE_MODES, "drive_no_caz"]
QUALITY_COLS = ["q_scenery", "q_space", "q_facilities", "q_tree_cover", "q_flatness"]
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

    # skipinitialspace tolerates spaces after commas in hand-edited CSVs.
    parks = pd.read_csv(data_dir / "parks.csv", skipinitialspace=True)
    origins = pd.read_csv(data_dir / "origins.csv", skipinitialspace=True)
    travel = pd.read_csv(data_dir / "travel_times.csv", skipinitialspace=True)

    # Trim any stray surrounding whitespace from string key/label columns.
    for df, cols in [
        (parks, ["park_id", "name"]),
        (origins, ["origin_id", "name"]),
        (travel, ["origin_id", "park_id", "mode"]),
    ]:
        for col in cols:
            df[col] = df[col].astype(str).str.strip()

    _require_columns(parks, PARK_COLS, "parks.csv")
    _require_columns(origins, ORIGIN_COLS, "origins.csv")
    _require_columns(travel, TRAVEL_COLS, "travel_times.csv")

    # --- types -------------------------------------------------------------
    parks["in_caz"] = _coerce_bool(parks["in_caz"])
    for col in QUALITY_COLS:
        parks[col] = pd.to_numeric(parks[col])
    parks["parking_cost_per_day"] = pd.to_numeric(parks["parking_cost_per_day"])
    origins["weight"] = pd.to_numeric(origins["weight"])
    # Blank durations (N/A, e.g. drive_no_caz for CAZ parks) become NaN.
    travel["duration_min"] = pd.to_numeric(travel["duration_min"], errors="coerce")

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
            "Every park needs a row for each of "
            f"{MODES} from every origin (drive_no_caz may be blank for CAZ parks)."
        )

    # Core modes must be reachable (filled) for every origin×park.
    core = travel[travel["mode"].isin(CORE_MODES)]
    if core["duration_min"].isna().any():
        raise DataValidationError(
            "walk/drive/transit durations must be filled for every origin×park."
        )

    # drive_no_caz must be N/A (blank) for parks inside the CAZ, and filled for
    # parks outside it (those are the parks you can drive to without paying).
    caz_ids = set(parks[parks["in_caz"]]["park_id"])
    dnc = travel[travel["mode"] == "drive_no_caz"]
    caz_filled = dnc[dnc["park_id"].isin(caz_ids) & dnc["duration_min"].notna()]
    if not caz_filled.empty:
        raise DataValidationError(
            "drive_no_caz must be blank (N/A) for parks inside the CAZ: "
            f"{sorted(set(caz_filled['park_id']))}"
        )
    non_caz_blank = dnc[~dnc["park_id"].isin(caz_ids) & dnc["duration_min"].isna()]
    if not non_caz_blank.empty:
        raise DataValidationError(
            "drive_no_caz is missing for non-CAZ parks: "
            f"{sorted(set(non_caz_blank['park_id']))}"
        )

    default_weights = _load_default_weights(data_dir)
    return Dataset(parks=parks, origins=origins, travel=travel, default_weights=default_weights)


def _load_default_weights(data_dir: Path) -> dict[str, float]:
    path = data_dir / "weights_default.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["criterion"], pd.to_numeric(df["weight"])))
