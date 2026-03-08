import os
import pandas as pd


"""
Build a model-ready driver-race table for the logistic regression milestone.

Design choices encoded here:
- Row unit: one driver-race observation identified by raceId + driverId
- Sample window: 2011-2022 inclusive, because pit-stop records are available there
- Outcome: win = 1 when finishing position is 1, else 0
- Predictor 1: pole = 1 when starting grid is 1, else 0
- Predictor 2:
    pit_stop_count = number of pit-stop records for the driver in that race
    pit_stop_binary = 1 when pit_stop_count >= 3, else 0

The pit-stop threshold is set to 3 so the binary feature represents a clearly
above-typical strategy. In this sample, 1-2 stops are most common, while 3+
stops still provides a sizable positive class for modeling.
"""


START_YEAR = 2011
END_YEAR = 2022
PIT_STOP_BINARY_THRESHOLD = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MERGED_PATH = os.path.join(PROJECT_ROOT, "f1_merged.csv")
RACES_PATH = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
PIT_STOPS_PATH = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_pit_stops.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "f1_model_ready_2011_2022.csv")


def build_model_ready_table() -> pd.DataFrame:
    merged = pd.read_csv(MERGED_PATH)
    races = pd.read_csv(RACES_PATH, usecols=["raceId", "year"])
    pit_stops = pd.read_csv(PIT_STOPS_PATH, usecols=["raceId", "driverId"])

    valid_races = races[races["year"].between(START_YEAR, END_YEAR)].copy()
    model = merged.merge(valid_races, on="raceId", how="inner")

    pit_stop_counts = (
        pit_stops[pit_stops["raceId"].isin(valid_races["raceId"])]
        .groupby(["raceId", "driverId"])
        .size()
        .reset_index(name="pit_stop_count")
    )

    model = model.merge(pit_stop_counts, on=["raceId", "driverId"], how="left")
    model["pit_stop_count"] = model["pit_stop_count"].fillna(0).astype(int)
    model["win"] = (model["position"] == 1).astype(int)
    model["pole"] = (model["grid"] == 1).astype(int)
    model["pit_stop_binary"] = (model["pit_stop_count"] >= PIT_STOP_BINARY_THRESHOLD).astype(int)

    model = model[
        [
            "raceId",
            "driverId",
            "year",
            "constructorId",
            "grid",
            "position",
            "win",
            "pole",
            "pit_stop_count",
            "pit_stop_binary",
        ]
    ].sort_values(["raceId", "driverId"]).reset_index(drop=True)

    return model


def validate_model_ready_table(model: pd.DataFrame) -> None:
    required_columns = {
        "raceId",
        "driverId",
        "year",
        "constructorId",
        "grid",
        "position",
        "win",
        "pole",
        "pit_stop_count",
        "pit_stop_binary",
    }

    missing_columns = required_columns.difference(model.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if model[["raceId", "driverId"]].isnull().any().any():
        raise ValueError("raceId and driverId must be non-null")

    duplicate_count = model.duplicated(subset=["raceId", "driverId"]).sum()
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate raceId+driverId rows")

    if model["year"].min() != START_YEAR or model["year"].max() != END_YEAR:
        raise ValueError(
            f"Expected years {START_YEAR}-{END_YEAR}, got {model['year'].min()}-{model['year'].max()}"
        )

    if model[["year", "constructorId", "grid", "position", "pit_stop_count"]].isnull().any().any():
        raise ValueError("Model table contains unexpected nulls")

    for column in ["win", "pole", "pit_stop_binary"]:
        values = set(model[column].unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} must be binary, got values {sorted(values)}")

    if not model["win"].equals((model["position"] == 1).astype(int)):
        raise ValueError("win is not coded consistently with position == 1")

    if not model["pole"].equals((model["grid"] == 1).astype(int)):
        raise ValueError("pole is not coded consistently with grid == 1")

    if (model["pit_stop_count"] < 0).any():
        raise ValueError("pit_stop_count cannot be negative")

    expected_binary = (model["pit_stop_count"] >= PIT_STOP_BINARY_THRESHOLD).astype(int)
    if not model["pit_stop_binary"].equals(expected_binary):
        raise ValueError(
            f"pit_stop_binary must equal 1 when pit_stop_count >= {PIT_STOP_BINARY_THRESHOLD}"
        )

    if model["pole"].nunique() != 2:
        raise ValueError("pole must contain both 0 and 1 values")

    if model["pit_stop_binary"].nunique() != 2:
        raise ValueError("pit_stop_binary must contain both 0 and 1 values")

    yearly_binary = model.groupby("year")["pit_stop_binary"].agg(["sum", "count"])
    if not ((yearly_binary["sum"] > 0) & (yearly_binary["sum"] < yearly_binary["count"])).all():
        raise ValueError("pit_stop_binary must contain both classes in every modeled year")


def main() -> None:
    model = build_model_ready_table()
    validate_model_ready_table(model)
    model.to_csv(OUTPUT_PATH, index=False)

    pit_binary_rate = model["pit_stop_binary"].mean()
    print(f"Saved {OUTPUT_PATH}")
    print(f"Rows: {len(model)}")
    print(f"Unique races: {model['raceId'].nunique()}")
    print(f"Years: {model['year'].min()}-{model['year'].max()}")
    print(f"Win rate: {model['win'].mean():.4f}")
    print(f"Pole rate: {model['pole'].mean():.4f}")
    print(
        f"Pit-stop binary rate (>= {PIT_STOP_BINARY_THRESHOLD} stops): {pit_binary_rate:.4f}"
    )


if __name__ == "__main__":
    main()
