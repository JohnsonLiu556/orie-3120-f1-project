import pandas as pd
df = pd.read_csv("f1_merged.csv")
races = pd.read_csv("Formula 1 Dataset Cleaned/cleaned_races.csv")
df2 = df.merge(races[['raceId','year']], on='raceId')

# Test 1: No pre-2003 data
assert df2['year'].min() == 2003, f"FAIL: min year is {df2['year'].min()}, expected 2003"

# Test 2: No post-2022 data
assert df2['year'].max() == 2022, f"FAIL: max year is {df2['year'].max()}, expected 2022"

# Test 3: Every year from 2003-2022 is present
for yr in range(2003, 2023):
    count = df2[df2['year']==yr]['raceId'].nunique()
    assert count >= 16, f"FAIL: {yr} has only {count} races, expected >= 16"

# Test 4: Total unique races is between 380 and 420
n_races = df['raceId'].nunique()
assert 380 <= n_races <= 420, f"FAIL: {n_races} unique races, expected 380-420"

# Test 5: No null values in any column
assert df.isnull().sum().sum() == 0, f"FAIL: nulls found:\n{df.isnull().sum()}"

# Test 6: No duplicate raceId+driverId combinations
dupes = df.duplicated(subset=['raceId','driverId']).sum()
assert dupes == 0, f"FAIL: {dupes} duplicate raceId+driverId rows"

# Test 7: 2003 should have exactly 16 races
assert df2[df2['year']==2003]['raceId'].nunique() == 16, "FAIL: 2003 race count wrong"

# Test 8: 2016 should have exactly 21 races
assert df2[df2['year']==2016]['raceId'].nunique() == 21, "FAIL: 2016 race count wrong"

print("ALL f1_merge tests PASSED")