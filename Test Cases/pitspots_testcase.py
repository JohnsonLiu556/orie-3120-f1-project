import pandas as pd
from scipy import stats

df = pd.read_csv("f1_merged.csv")
races = pd.read_csv("Formula 1 Dataset Cleaned/cleaned_races.csv")
pit = pd.read_csv("Formula 1 Dataset Cleaned/cleaned_pit_stops.csv")

valid_races = races[(races['year'] >= 2011) & (races['year'] <= 2022)]['raceId']
pit_filtered = pit[pit['raceId'].isin(valid_races)]
pit_counts = pit_filtered.groupby(['raceId','driverId']).size().reset_index(name='n_stops')

pole_sitters = df[df['grid']==1][['raceId','driverId']].rename(columns={'driverId':'pole_driverId'})
winners      = df[df['position']==1][['raceId','driverId']].rename(columns={'driverId':'winner_driverId'})
race_info    = pole_sitters.merge(winners, on='raceId')
race_info['pole_win'] = race_info['pole_driverId'] == race_info['winner_driverId']

winner_stops = race_info.merge(
    pit_counts, left_on=['raceId','winner_driverId'], right_on=['raceId','driverId'], how='left'
)
analysis = winner_stops.dropna(subset=['n_stops'])
wins   = analysis[analysis['pole_win']]['n_stops']
losses = analysis[~analysis['pole_win']]['n_stops']
t_stat, p_val = stats.ttest_ind(wins, losses)

# Test 1: Both groups have enough samples
assert len(wins)   >= 100, f"FAIL: pole-win group has only {len(wins)} races"
assert len(losses) >= 100, f"FAIL: pole-lose group has only {len(losses)} races"

# Test 2: Mean stops are close (< 0.5 apart) — confirms similar strategy
assert abs(wins.mean() - losses.mean()) < 0.5, \
    f"FAIL: means differ by {abs(wins.mean()-losses.mean()):.2f}, expected < 0.5"

# Test 3: t-test is NOT significant (p > 0.05)
assert p_val > 0.05, f"FAIL: p={p_val:.4f} is significant — unexpected result, check data"

# Test 4: Verify ground-truth mean values
assert abs(wins.mean() - 1.869) < 0.1,   f"FAIL: wins mean {wins.mean():.3f}, expected ~1.869"
assert abs(losses.mean() - 2.025) < 0.1, f"FAIL: losses mean {losses.mean():.3f}, expected ~2.025"

# Test 5: No pit stop counts of 0 (would indicate a join error)
assert (analysis['n_stops'] > 0).all(), "FAIL: zero pit stop counts found — join error"

# Test 6: Max pit stops is reasonable (<=6)
assert analysis['n_stops'].max() <= 8, \
    f"FAIL: max pit stops = {analysis['n_stops'].max()} — likely a data error"

# Test 7: All races in analysis are from 2011-2022
race_years = analysis.merge(races[['raceId','year']], on='raceId')['year']
assert race_years.min() >= 2011, f"FAIL: pit data includes pre-2011 races"
assert race_years.max() <= 2022, f"FAIL: pit data includes post-2022 races"

# Test 8: Image created
import os
assert os.path.exists("plots/viz_pitstops.png"), "FAIL: image not found"

print("ALL viz_pitstops tests PASSED")
