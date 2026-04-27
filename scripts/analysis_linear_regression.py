import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path   = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path    = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
circuits_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_circuits.csv")
out_txt  = os.path.join(PROJECT_ROOT, "plots", "analysis1_linear_regression_summary.txt")
out_plot = os.path.join(PROJECT_ROOT, "plots", "analysis1_linear_regression.png")

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

df["era_2010_2019"]     = (df["era"] == "2010-2019").astype(int)
df["era_2020_2022"]     = (df["era"] == "2020-2022").astype(int)
df["circuit_type_street"] = (df["circuit_type"] == "Street").astype(int)

model_cols = ["position", "grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"]
model_df = df[model_cols].dropna()

print(f"Regression sample: {len(model_df):,} observations")
print(f"  Street circuit rows : {model_df['circuit_type_street'].sum():,}")
print(f"  2010–2019 era rows  : {model_df['era_2010_2019'].sum():,}")
print(f"  2020–2022 era rows  : {model_df['era_2020_2022'].sum():,}")
print()

X = sm.add_constant(model_df[["grid", "era_2010_2019", "era_2020_2022", "circuit_type_street"]])
y = model_df["position"]
model  = sm.OLS(y, X)
result = model.fit()

summary_str = result.summary().as_text()
print(summary_str)

with open(out_txt, "w") as f:
    f.write("OLS Regression: finishing position ~ grid + era_2010_2019 + era_2020_2022 + circuit_type_street\n")
    f.write("Baseline era: 2003–2009  |  Baseline circuit type: Permanent\n")
    f.write("=" * 78 + "\n\n")
    f.write(summary_str)
print(f"\nSummary saved to {out_txt}")

slope_grid = result.params["grid"]
intercept  = result.params["const"]

dummy_means = model_df[["era_2010_2019", "era_2020_2022", "circuit_type_street"]].mean()
offset = (result.params["era_2010_2019"] * dummy_means["era_2010_2019"]
          + result.params["era_2020_2022"] * dummy_means["era_2020_2022"]
          + result.params["circuit_type_street"] * dummy_means["circuit_type_street"])

x_grid = np.linspace(model_df["grid"].min(), model_df["grid"].max(), 200)
y_hat  = intercept + offset + slope_grid * x_grid

grid_bins = np.arange(model_df["grid"].min() - 0.5, model_df["grid"].max() + 1.5, 1)
pos_bins  = np.arange(model_df["position"].min() - 0.5, model_df["position"].max() + 1.5, 1)

fig, ax = plt.subplots(figsize=(8, 6))
_, _, _, img = ax.hist2d(
    model_df["grid"], model_df["position"],
    bins=[grid_bins, pos_bins],
    cmap="Blues",
)
plt.colorbar(img, ax=ax, label="Count")
ax.plot(x_grid, y_hat, color="firebrick", linewidth=2,
        label=f"OLS fit  (grid coef = {slope_grid:.3f})")

ax.set_xlabel("Grid position (starting position)")
ax.set_ylabel("Finishing position")
ax.set_title(
    "Grid position vs finishing position — OLS regression (2003–2022)\n"
    "Line evaluated at sample-mean era & circuit-type composition",
    fontsize=11,
)
stats_text = (
    f"R² = {result.rsquared:.3f}\n"
    f"Adj. R² = {result.rsquared_adj:.3f}\n"
    f"n = {len(model_df):,}"
)
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_plot, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to {out_plot}")
