import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path   = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path    = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
circuits_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_circuits.csv")
out_plot  = os.path.join(PROJECT_ROOT, "plots", "analysis2_residual_plots.png")
out_tests = os.path.join(PROJECT_ROOT, "plots", "analysis2_assumption_tests.txt")

df       = pd.read_csv(merged_path)
races    = pd.read_csv(races_path)[["raceId", "year", "circuitId"]]
circuits = pd.read_csv(circuits_path)

df = df.merge(races, on="raceId")
df = df.merge(circuits[["circuitId"] + (
    ["circuit_name"] if "circuit_name" in circuits.columns else ["name"]
)], on="circuitId")
df = df[(df["year"] >= 2003) & (df["year"] <= 2022)]

street_keywords = [
    "street", "Street", "Monaco", "Marina Bay", "Baku", "Albert Park",
    "Adelaide", "Phoenix", "Detroit", "Long Beach", "Las Vegas",
    "Miami", "Jeddah", "Corniche", "Valencia Street", "Montjuïc",
    "Boavista", "AVUS", "Gilles Villeneuve",
]

def is_street(name):
    if pd.isna(name):
        return False
    return any(kw in str(name) for kw in street_keywords)

if "type" in circuits.columns:
    ctype = circuits[["circuitId", "type"]].copy()
    ctype["circuit_type"] = ctype["type"].str.strip().str.lower().map(
        lambda x: "Street" if x == "street" else "Permanent"
    )
else:
    name_col = "circuit_name" if "circuit_name" in circuits.columns else "name"
    ctype = circuits[["circuitId", name_col]].copy()
    ctype["circuit_type"] = ctype[name_col].map(
        lambda x: "Street" if is_street(x) else "Permanent"
    )

df = df.merge(ctype[["circuitId", "circuit_type"]], on="circuitId", how="left")

def era_bucket(y):
    if y <= 2009:
        return "2003-2009"
    if y <= 2019:
        return "2010-2019"
    return "2020-2022"

df["era"] = df["year"].apply(era_bucket)
df["era_2010_2019"]       = (df["era"] == "2010-2019").astype(int)
df["era_2020_2022"]       = (df["era"] == "2020-2022").astype(int)
df["circuit_type_street"] = (df["circuit_type"] == "Street").astype(int)

model_cols = ["position", "grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"]
model_df = df[model_cols].dropna()

X = sm.add_constant(model_df[["grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"]])
y = model_df["position"]
result = sm.OLS(y, X).fit()

fitted    = result.fittedvalues
residuals = result.resid
influence = result.get_influence()
std_resid = influence.resid_studentized_internal  

bp_lm, bp_pval, bp_fstat, bp_fpval = het_breuschpagan(residuals, result.model.exog)

rng = np.random.default_rng(42)
sw_sample = rng.choice(residuals.values, size=min(5000, len(residuals)), replace=False)
sw_stat, sw_pval = stats.shapiro(sw_sample)

lines = [
    "OLS Regression Assumption Tests",
    "Model: finishing position ~ grid + era_2010_2019 + era_2020_2022 + circuit_type_street",
    "=" * 70,
    "",
    "1. Test for constant variance ",
    f"   LM statistic : {bp_lm:.4f}",
    f"   LM p-value   : {bp_pval:.4e}",
    f"   F statistic  : {bp_fstat:.4f}",
    f"   F p-value    : {bp_fpval:.4e}",
    f"   Interpretation: {'Evidence of heteroscedasticity (p < 0.05)' if bp_pval < 0.05 else 'No significant heteroscedasticity detected (p >= 0.05)'}",
    "",
    "2. Test for normality of residuals",
    f"   (subsample n = {len(sw_sample):,} drawn randomly with seed 42)",
    f"   W statistic  : {sw_stat:.6f}",
    f"   p-value      : {sw_pval:.4e}",
    f"   Interpretation: {'Residuals are not normally distributed (p < 0.05)' if sw_pval < 0.05 else 'Residuals are consistent with normality (p >= 0.05)'}",
]

test_output = "\n".join(lines)
print(test_output)
with open(out_tests, "w") as f:
    f.write(test_output + "\n")
print(f"\nTest results saved to {out_tests}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "OLS regression diagnostic plots\n"
    "finishing position ~ grid + era (2010-2019, 2020-2022) + circuit type  (2003–2022)",
    fontsize=13, y=1.01,
)

ax = axes[0, 0]
ax.scatter(fitted, residuals, alpha=0.08, s=6, color="steelblue")
ax.axhline(0, color="firebrick", linewidth=1.5, linestyle="--")

lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.2)
ax.plot(lowess[:, 0], lowess[:, 1], color="darkorange", linewidth=1.8, label="LOWESS")
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted\n(tests for non-linearity)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

ax = axes[0, 1]
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
ax.scatter(osm, osr, alpha=0.15, s=6, color="steelblue")
x_ref = np.array([osm[0], osm[-1]])
ax.plot(x_ref, slope * x_ref + intercept, color="firebrick", linewidth=1.5, linestyle="--",
        label=f"Ref line  (r = {r:.3f})")
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Sample quantiles")
ax.set_title("Normal Q-Q plot\n(tests whether residuals are normally distributed)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

ax = axes[1, 0]
sqrt_abs_std = np.sqrt(np.abs(std_resid))
ax.scatter(fitted, sqrt_abs_std, alpha=0.08, s=6, color="steelblue")
lowess2 = sm.nonparametric.lowess(sqrt_abs_std, fitted, frac=0.2)
ax.plot(lowess2[:, 0], lowess2[:, 1], color="darkorange", linewidth=1.8, label="LOWESS")
ax.set_xlabel("Fitted values")
ax.set_ylabel(r"$\sqrt{|\mathrm{Standardised\ residuals}|}$")
ax.set_title("Scale-Location\n(tests for constant variance)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

ax = axes[1, 1]
n_bins = min(60, int(np.sqrt(len(residuals))))
counts, bin_edges, _ = ax.hist(residuals, bins=n_bins, density=True,
                                color="steelblue", alpha=0.6, edgecolor="white", linewidth=0.4)
x_norm = np.linspace(residuals.min(), residuals.max(), 300)
ax.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
        color="firebrick", linewidth=2, label="Normal PDF")
sw_label = f"Shapiro-Wilk p = {sw_pval:.2e}\n(n = {len(sw_sample):,} subsample)"
ax.text(0.97, 0.95, sw_label, transform=ax.transAxes, fontsize=8,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.set_xlabel("Residual")
ax.set_ylabel("Density")
ax.set_title("Residual distribution\n(tests for normality of errors)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig(out_plot, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to {out_plot}")
