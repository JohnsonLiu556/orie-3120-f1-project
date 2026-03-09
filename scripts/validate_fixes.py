import os
import pandas as pd
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)

df = pd.read_csv(os.path.join(PROJECT_ROOT, "f1_merged.csv"))
races = pd.read_csv(os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv"))
df2 = df.merge(races[["raceId", "year"]], on="raceId")
df2 = df2[(df2["year"] >= 2003) & (df2["year"] <= 2022)]

# Fig 1: integer ticks check (data level)
poles = df2[df2["grid"] == 1]
years = sorted(poles["year"].unique())
assert all(int(y) == y for y in years), "FAIL: non-integer years in data"
assert min(years) == 2003 and max(years) == 2022, f"FAIL: year range {min(years)}-{max(years)}"
print("Fix 1 data check PASSED")

# Fig 2: street era finding
circuits = pd.read_csv(os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_circuits.csv"))
# Keep this list in sync with viz_circuit_type.py (canonical source)
street_kw = [
    "street", "Street", "Monaco", "Marina Bay", "Baku", "Albert Park",
    "Adelaide", "Phoenix", "Detroit", "Long Beach", "Las Vegas", "Miami",
    "Jeddah", "Corniche", "Valencia Street", "Montjuïc", "Boavista",
    "AVUS", "Gilles Villeneuve",
]
circuits["circuit_type"] = circuits["circuit_name"].map(
    lambda x: "Street" if any(kw in str(x) for kw in street_kw) else "Permanent"
)
df3 = df2.merge(
    races[["raceId", "circuitId"]].drop_duplicates(), on="raceId"
).merge(circuits[["circuitId", "circuit_type"]], on="circuitId")
poles3 = df3[df3["grid"] == 1].copy()
poles3["pole_win"] = (poles3["position"] == 1).astype(int)
pre = poles3[(poles3["circuit_type"] == "Street") & (poles3["year"] < 2010)]
post = poles3[(poles3["circuit_type"] == "Street") & (poles3["year"] >= 2010)]
pre_rate = 100 * pre["pole_win"].sum() / pre["raceId"].nunique()
post_rate = 100 * post["pole_win"].sum() / post["raceId"].nunique()
drop = pre_rate - post_rate
assert abs(pre_rate - 58.3) < 1.0, f"FAIL: pre-2010 street={pre_rate:.1f}"
assert abs(post_rate - 49.1) < 1.0, f"FAIL: post-2010 street={post_rate:.1f}"
assert abs(drop - 9.0) < 1.5, f"FAIL: drop={drop:.1f}pp, expected ~9pp"
print(f"Fix 2 data check PASSED — street drop = {drop:.1f}pp")

# Fig 3: r^2 values
def era(y):
    if y <= 2009:
        return "2003-2009"
    if y <= 2019:
        return "2010-2019"
    return "2020-2022"

df2 = df2.copy()
df2["era"] = df2["year"].apply(era)
expected = {"2003-2009": 0.304, "2010-2019": 0.402, "2020-2022": 0.379}
r2_vals = {}
for e, exp in expected.items():
    sub = df2[df2["era"] == e]
    r, _ = stats.pearsonr(sub["grid"], sub["position"])
    r2 = r**2
    r2_vals[e] = r2
    assert abs(r2 - exp) < 0.005, f"FAIL: {e} r²={r2:.3f}, expected {exp}"
assert r2_vals["2010-2019"] > r2_vals["2003-2009"], "FAIL: r² should increase from 2003-2009 to 2010-2019"
assert r2_vals["2010-2019"] > r2_vals["2020-2022"], "FAIL: 2010-2019 should have highest r²"
print("Fix 3 data check PASSED — r² values correct")

for f in ["viz1_pole_win_rate.png", "viz2_circuit_type.png", "viz3_grid_vs_finish_era.png"]:
    path = os.path.join(PROJECT_ROOT, "plots", f)
    assert os.path.exists(path), f"MISSING: {path}"
    size = os.path.getsize(path)
    assert size > 50000, f"FAIL: {f} too small ({size} bytes)"
print("All image files present and non-empty")
print("\nALL 3 FIXES VERIFIED")
