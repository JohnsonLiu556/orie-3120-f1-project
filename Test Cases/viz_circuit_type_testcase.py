import os
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(PROJECT_ROOT, "f1_merged.csv"))
races = pd.read_csv(os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv"))
circuits = pd.read_csv(os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_circuits.csv"))

df2 = df.merge(races[['raceId','year','circuitId']], on='raceId')
df2 = df2[(df2['year'] >= 2003) & (df2['year'] <= 2022)]

street_keywords = [
    "street","Street","Monaco","Marina Bay","Baku","Albert Park",
    "Adelaide","Phoenix","Detroit","Long Beach","Las Vegas",
    "Miami","Jeddah","Corniche","Valencia Street","Montjuïc",
    "Boavista","AVUS","Gilles Villeneuve",
]
circuits['circuit_type'] = circuits['circuit_name'].map(
    lambda x: "Street" if any(kw in str(x) for kw in street_keywords) else "Permanent"
)
df2 = df2.merge(circuits[['circuitId','circuit_type']], on='circuitId')
poles = df2[df2['grid']==1].copy()
poles['pole_win'] = (poles['position']==1).astype(int)

# Test 1: Both circuit types have enough races to be meaningful
for ct in ['Street','Permanent']:
    n = poles[poles['circuit_type']==ct]['raceId'].nunique()
    assert n >= 20, f"FAIL: {ct} has only {n} races"

# Test 2: Verify known ground-truth rates (2003–2022)
street_rate = 100 * poles[poles['circuit_type']=='Street']['pole_win'].sum() / \
              poles[poles['circuit_type']=='Street']['raceId'].nunique()
perm_rate   = 100 * poles[poles['circuit_type']=='Permanent']['pole_win'].sum() / \
              poles[poles['circuit_type']=='Permanent']['raceId'].nunique()
assert abs(street_rate - 51.85) < 1.0, f"FAIL: street rate {street_rate:.2f}, expected ~51.85"
assert abs(perm_rate - 50.83) < 1.0,   f"FAIL: permanent rate {perm_rate:.2f}, expected ~50.83"

# Test 3: Difference between circuit types < 5 percentage points (confirms no strong effect)
assert abs(street_rate - perm_rate) < 5, \
    f"FAIL: difference {abs(street_rate-perm_rate):.1f}pp is too large — check classification"

# Test 4: Pre-2010 street rate should be ~58.3% and permanent ~51.5%
pre_street = poles[(poles['circuit_type']=='Street')&(poles['year']<2010)]
pre_street_rate = 100 * pre_street['pole_win'].sum() / pre_street['raceId'].nunique()
assert abs(pre_street_rate - 58.33) < 1.0, \
    f"FAIL: pre-2010 street rate {pre_street_rate:.1f}, expected ~58.33"

# Test 5: t-test between street and permanent should NOT be significant (p > 0.05)
street_wins = poles[poles['circuit_type']=='Street'].groupby('raceId')['pole_win'].max()
perm_wins   = poles[poles['circuit_type']=='Permanent'].groupby('raceId')['pole_win'].max()
_, p = stats.ttest_ind(street_wins, perm_wins)
assert p > 0.05, f"FAIL: p={p:.4f}, expected p > 0.05 (no significant difference)"

# Test 6: Image was created
assert os.path.exists(os.path.join(PROJECT_ROOT, "plots", "viz2_circuit_type.png")), "FAIL: image not found"

print("ALL viz_circuit_type tests PASSED")