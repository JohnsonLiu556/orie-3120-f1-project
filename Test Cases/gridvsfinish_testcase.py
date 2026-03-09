import os
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(PROJECT_ROOT, "f1_merged.csv"))
races = pd.read_csv(os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv"))
df2 = df.merge(races[['raceId','year']], on='raceId')
df2 = df2[(df2['year'] >= 2003) & (df2['year'] <= 2022)]

def era_bucket(y):
    if y <= 2009:  return "2003-2009"
    if y <= 2019:  return "2010-2019"
    return         "2020-2022"

df2['era'] = df2['year'].apply(era_bucket)

# Test 1: Exactly 3 eras present, no Pre-2000 or 2020-present labels
assert set(df2['era'].unique()) == {"2003-2009","2010-2019","2020-2022"}, \
    f"FAIL: unexpected eras {df2['era'].unique()}"

# Test 2: Each era has enough data
for era in ["2003-2009","2010-2019","2020-2022"]:
    n = len(df2[df2['era']==era])
    assert n >= 500, f"FAIL: {era} has only {n} rows"

# Test 3: Verify exact Pearson r values (within tolerance)
expected = {"2003-2009": 0.5513, "2010-2019": 0.6337, "2020-2022": 0.6189}
for era, exp_r in expected.items():
    sub = df2[df2['era']==era]
    r, _ = stats.pearsonr(sub['grid'], sub['position'])
    assert abs(r - exp_r) < 0.01, \
        f"FAIL: {era} r={r:.4f}, expected {exp_r:.4f}"

# Test 4: Correlation increases from earliest to middle era (stronger predictor in modern F1)
subs = {era: df2[df2['era']==era] for era in ["2003-2009","2010-2019","2020-2022"]}
r_early, _ = stats.pearsonr(subs["2003-2009"]['grid'], subs["2003-2009"]['position'])
r_mid,   _ = stats.pearsonr(subs["2010-2019"]['grid'], subs["2010-2019"]['position'])
assert r_mid > r_early, \
    f"FAIL: r did not increase from 2003-2009 ({r_early:.3f}) to 2010-2019 ({r_mid:.3f})"

# Test 5: All grid and position values are positive integers
assert (df2['grid'] > 0).all(),    "FAIL: non-positive grid values found"
assert (df2['position'] > 0).all(),"FAIL: non-positive position values found"

# Test 6: Image created
assert os.path.exists(os.path.join(PROJECT_ROOT, "plots", "viz3_grid_vs_finish_era.png")), "FAIL: image not found"

print("ALL viz_grid_finish_era tests PASSED")