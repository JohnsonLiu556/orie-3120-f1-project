import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(PROJECT_ROOT, "f1_merged.csv"))
races = pd.read_csv(os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv"))
df = df.merge(races[["raceId", "year"]], on="raceId")
df = df[(df["year"] >= 2003) & (df["year"] <= 2022)]

df["win"] = (df["position"] == 1).astype(int)
df["grid_bucket"] = df["grid"].apply(lambda g: g if g <= 10 else 11)

grid_stats = df.groupby("grid_bucket").agg(
    total=("win", "count"),
    wins=("win", "sum"),
).reset_index()
grid_stats["win_rate"] = 100 * grid_stats["wins"] / grid_stats["total"]

# Test 1: Pole position has the highest win rate
pole_rate = grid_stats[grid_stats["grid_bucket"] == 1]["win_rate"].values[0]
p2_rate = grid_stats[grid_stats["grid_bucket"] == 2]["win_rate"].values[0]
assert pole_rate > p2_rate, f"FAIL: P1 win rate ({pole_rate:.1f}%) should exceed P2 ({p2_rate:.1f}%)"

# Test 2: Win rate is strictly decreasing from P1 to P5
for pos in range(1, 5):
    r_curr = grid_stats[grid_stats["grid_bucket"] == pos]["win_rate"].values[0]
    r_next = grid_stats[grid_stats["grid_bucket"] == pos + 1]["win_rate"].values[0]
    assert r_curr > r_next, f"FAIL: win rate should decrease from P{pos} to P{pos+1}"

# Test 3: 11+ bucket has a very low win rate (< 5%)
tail_rate = grid_stats[grid_stats["grid_bucket"] == 11]["win_rate"].values[0]
assert tail_rate < 5, f"FAIL: 11+ win rate is {tail_rate:.1f}%, expected < 5%"

# Test 4: Exactly one winner per race (total wins == total unique races)
total_wins = grid_stats["wins"].sum()
total_races = df["raceId"].nunique()
assert total_wins == total_races, f"FAIL: {total_wins} wins but {total_races} races"

# Test 5: Pole win rate is between 40% and 60%
assert 40 <= pole_rate <= 60, f"FAIL: pole win rate {pole_rate:.1f}% outside expected range"

# Test 6: Image was created
assert os.path.exists(os.path.join(PROJECT_ROOT, "plots", "viz_win_rate_by_grid.png")), \
    "FAIL: output image not found"

print("ALL win_rate_by_grid tests PASSED")
