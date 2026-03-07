import pandas as pd
import os


# 2023 is also excluded because raceIds 1111-1120 have no qualifying
# entries in cleaned_qualifying.csv. analysis covers 2003-2022 only.

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
base = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned")
files = {
    "results": "cleaned_results.csv",
    "qualifying": "cleaned_qualifying.csv",
    "races": "cleaned_races.csv",
    "circuits": "cleaned_circuits.csv",
    "constructors": "cleaned_constructors.csv",
    "constructor_standings": "cleaned_constructor_standings.csv",
    "drivers": "cleaned_drivers.csv",
    "pit_stops": "cleaned_pit_stops.csv",
    "lap_times": "cleaned_lap_times.csv",
    "constructor_results": "cleaned_constructor_results.csv",
    "driver_standings": "cleaned_driver_standings.csv",
    "sprint_results": "cleaned_sprint_results.csv",
    "status": "cleaned_status.csv",
    "seasons": "cleaned_seasons.csv",
}
dfs = {name: pd.read_csv(os.path.join(base, path)) for name, path in files.items()}

for name, df in dfs.items():
    print(f"=== {name} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("null counts:")
    print(df.isnull().sum().to_string())
    print()

qual = dfs["qualifying"][["raceId", "driverId", "constructorId"]].copy()
qual["grid"] = dfs["qualifying"]["position"]
res = dfs["results"][["raceId", "driverId", "position", "constructorId"]].copy()
merged = qual.merge(
    res[["raceId", "driverId", "position"]],
    on=["raceId", "driverId"],
    how="inner",
)
merged = merged[["raceId", "driverId", "grid", "position", "constructorId"]]

nulls = merged.isnull().any(axis=1)
if nulls.any():
    print("Rows with nulls after join:")
    print(merged[nulls])
    print("Count:", nulls.sum())
else:
    print("No rows with nulls after join.")

dup_key = merged.duplicated(subset=["raceId", "driverId"], keep=False)
if dup_key.any():
    print("Rows with duplicate raceId+driverId:")
    print(merged[dup_key].sort_values(["raceId", "driverId"]))
    print("Count:", dup_key.sum())
else:
    print("No duplicate raceId+driverId.")

merged = merged.drop_duplicates(subset=["raceId", "driverId"], keep="first")

races = dfs["races"]
races_1983 = races.loc[races["year"] >= 1983, "raceId"]
merged = merged[merged["raceId"].isin(races_1983)]

races_2003_2022 = races.loc[(races["year"] >= 2003) & (races["year"] <= 2022), "raceId"]
merged = merged[merged["raceId"].isin(races_2003_2022)]

out_path = os.path.join(PROJECT_ROOT, "f1_merged.csv")
merged.to_csv(out_path, index=False)
print("Saved f1_merged.csv, shape:", merged.shape)

merged_with_year = merged.merge(races[["raceId", "year"]], on="raceId", how="left")
merged_counts = merged_with_year.groupby("year")["raceId"].nunique()
race_counts = races.groupby("year")["raceId"].nunique()

print("\nSeason completeness check (merged races vs. races table):")
for year in sorted(race_counts.index):
    if year < 2003 or year > 2022:
        continue
    total = race_counts.loc[year]
    merged_total = merged_counts.get(year, 0)
    completeness = merged_total / total if total else 0
    flag = " <-- WARNING: <90% complete" if completeness < 0.9 else ""
    print(f"{year}: merged {merged_total} / races {total} ({completeness:.1%}){flag}")

