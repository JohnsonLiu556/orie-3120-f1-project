import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("f1_merged.csv")
races = pd.read_csv("Formula 1 Dataset Cleaned/cleaned_races.csv")
df2 = df.merge(races[['raceId','year']], on='raceId')
df2 = df2[(df2['year'] >= 2003) & (df2['year'] <= 2022)]

poles = df2[df2['grid']==1].copy()
poles['pole_win'] = (poles['position']==1).astype(int)
season_stats = poles.groupby('year').agg(
    total=('raceId','nunique'),
    wins=('pole_win','sum')
).reset_index()
season_stats['rate'] = 100 * season_stats['wins'] / season_stats['total']

# Test 1: No year outside 2003-2022
assert season_stats['year'].min() == 2003, "FAIL: years start before 2003"
assert season_stats['year'].max() == 2022, "FAIL: years go past 2022"

# Test 2: No season has fewer than 16 races (data completeness)
assert season_stats['total'].min() >= 16, \
    f"FAIL: season with {season_stats['total'].min()} races — incomplete data"

# Test 3: No rate is 0% or 100% (would indicate bad data)
assert season_stats['rate'].min() > 0, "FAIL: a season shows 0% conversion — data artifact"
assert season_stats['rate'].max() < 100, "FAIL: a season shows 100% conversion — data artifact"

# Test 4: Verify specific known values
r2003 = season_stats[season_stats['year']==2003]['rate'].values[0]
assert abs(r2003 - 56.25) < 0.1, f"FAIL: 2003 rate is {r2003:.2f}, expected 56.25"

r2016 = season_stats[season_stats['year']==2016]['rate'].values[0]
assert abs(r2016 - 61.90) < 0.1, f"FAIL: 2016 rate is {r2016:.2f}, expected 61.90"

r2022 = season_stats[season_stats['year']==2022]['rate'].values[0]
assert abs(r2022 - 45.45) < 0.1, f"FAIL: 2022 rate is {r2022:.2f}, expected 45.45"

# Test 5: Overall mean rate should be between 45% and 60%
overall_mean = season_stats['rate'].mean()
assert 45 <= overall_mean <= 60, f"FAIL: mean rate {overall_mean:.1f}% outside expected range"

# Test 6: Image file was created and is non-empty
import os
assert os.path.exists("plots/viz1_pole_win_rate.png"), "FAIL: output image not found"
assert os.path.getsize("plots/viz1_pole_win_rate.png") > 50000, "FAIL: image file suspiciously small"

print("ALL viz_pole_win tests PASSED")