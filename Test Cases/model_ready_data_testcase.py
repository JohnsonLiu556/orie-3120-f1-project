import os
import pandas as pd


MODEL_PATH = "f1_model_ready_2011_2022.csv"
PIT_BINARY_THRESHOLD = 3


assert os.path.exists(MODEL_PATH), f"FAIL: model-ready file not found at {MODEL_PATH}"

df = pd.read_csv(MODEL_PATH)
races = pd.read_csv("Formula 1 Dataset Cleaned/cleaned_races.csv")[["raceId", "year"]]
merged = pd.read_csv("f1_merged.csv")
pit = pd.read_csv("Formula 1 Dataset Cleaned/cleaned_pit_stops.csv")

base = merged.merge(races[races["year"].between(2011, 2022)], on="raceId", how="inner")
pit_counts = pit[pit["raceId"].isin(base["raceId"])].groupby(["raceId", "driverId"]).size()

expected_columns = {
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

# Test 1: Exact schema is present
assert set(df.columns) == expected_columns, f"FAIL: unexpected columns {list(df.columns)}"

# Test 2: Table has the expected row count from the 2011-2022 base sample
assert len(df) == len(base), f"FAIL: {len(df)} rows, expected {len(base)}"

# Test 3: No duplicate driver-race rows
dupes = df.duplicated(subset=["raceId", "driverId"]).sum()
assert dupes == 0, f"FAIL: found {dupes} duplicate raceId+driverId rows"

# Test 4: Year window is exactly 2011-2022
assert df["year"].min() == 2011, f"FAIL: min year is {df['year'].min()}, expected 2011"
assert df["year"].max() == 2022, f"FAIL: max year is {df['year'].max()}, expected 2022"

# Test 5: No unexpected nulls in model columns
assert df.isnull().sum().sum() == 0, f"FAIL: nulls found:\n{df.isnull().sum()}"

# Test 6: win and pole are correctly encoded
assert df["win"].equals((df["position"] == 1).astype(int)), "FAIL: win coding is incorrect"
assert df["pole"].equals((df["grid"] == 1).astype(int)), "FAIL: pole coding is incorrect"

# Test 7: pit_stop_count matches the raw pit-stop records
for _, row in df.sample(n=min(250, len(df)), random_state=3120).iterrows():
    expected_count = pit_counts.get((row["raceId"], row["driverId"]), 0)
    assert row["pit_stop_count"] == expected_count, (
        "FAIL: pit_stop_count mismatch for "
        f"raceId={row['raceId']}, driverId={row['driverId']}"
    )

# Test 8: pit_stop_binary uses the >=3 rule exactly
assert df["pit_stop_binary"].equals((df["pit_stop_count"] >= PIT_BINARY_THRESHOLD).astype(int)), (
    "FAIL: pit_stop_binary is not coded as 1 when pit_stop_count >= 3"
)

# Test 9: pit_stop_binary has both classes overall and within each year
assert df["pit_stop_binary"].nunique() == 2, "FAIL: pit_stop_binary is constant"
yearly = df.groupby("year")["pit_stop_binary"].agg(["sum", "count"])
assert ((yearly["sum"] > 0) & (yearly["sum"] < yearly["count"])).all(), (
    "FAIL: at least one modeled year does not contain both pit_stop_binary classes"
)

# Test 10: Sample size and race coverage are stable
assert df["raceId"].nunique() == 240, f"FAIL: {df['raceId'].nunique()} unique races, expected 240"
assert len(df) == 5037, f"FAIL: {len(df)} rows, expected 5037"

print("ALL model ready dataset tests PASSED")
