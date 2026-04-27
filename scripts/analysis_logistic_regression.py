import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path   = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path    = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
circuits_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_circuits.csv")
out_txt  = os.path.join(PROJECT_ROOT, "plots", "analysis3_logistic_regression_summary.txt")
out_plot = os.path.join(PROJECT_ROOT, "plots", "analysis3_odds_ratios.png")

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

df["pole"] = (df["grid"] == 1).astype(int)
df["win"]  = (df["position"] == 1).astype(int)

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

model_cols = ["win", "pole", "era_2010_2019", "era_2020_2022", "circuit_type_street"]
model_df = df[model_cols].dropna()

print(f"Logistic regression sample: {len(model_df):,} observations")
print(f"  Wins (y=1)          : {model_df['win'].sum():,}  ({100*model_df['win'].mean():.1f}%)")
print(f"  Pole sitters        : {model_df['pole'].sum():,}")
print(f"  Street circuit rows : {model_df['circuit_type_street'].sum():,}")
print()

X = sm.add_constant(model_df[["pole", "era_2010_2019", "era_2020_2022", "circuit_type_street"]])
y = model_df["win"]
result = sm.Logit(y, X).fit(maxiter=200, disp=False)

params    = result.params
conf      = result.conf_int()          
or_vals   = np.exp(params)
or_lo     = np.exp(conf[0])
or_hi     = np.exp(conf[1])

or_df = pd.DataFrame({
    "Coefficient (log-odds)": params,
    "Odds Ratio":             or_vals,
    "OR 95% CI lower":        or_lo,
    "OR 95% CI upper":        or_hi,
    "p-value":                result.pvalues,
}).drop(index="const")

summary_str = result.summary().as_text()
or_str      = or_df.to_string(float_format=lambda x: f"{x:.4f}")

print(summary_str)
print("\nOdds Ratios (exponentiated coefficients):")
print(or_str)

with open(out_txt, "w") as f:
    f.write("Logistic Regression: win ~ pole + era_2010_2019 + era_2020_2022 + circuit_type_street\n")
    f.write("Baseline era: 2003–2009  |  Baseline circuit type: Permanent\n")
    f.write("=" * 78 + "\n\n")
    f.write(summary_str)
    f.write("\n\nOdds Ratios (exponentiated coefficients):\n")
    f.write(or_str + "\n")
print(f"\nSummary saved to {out_txt}")

labels = {
    "pole":               "Pole position",
    "era_2010_2019":      "Era 2010–2019\n(vs 2003–2009)",
    "era_2020_2022":      "Era 2020–2022\n(vs 2003–2009)",
    "circuit_type_street": "Street circuit\n(vs Permanent)",
}
plot_vars = list(labels.keys())
display_labels = [labels[v] for v in plot_vars]
ors   = [or_vals[v] for v in plot_vars]
lo_err = [or_vals[v] - or_lo[v] for v in plot_vars]
hi_err = [or_hi[v] - or_vals[v] for v in plot_vars]
pvals  = [result.pvalues[v] for v in plot_vars]

colors = ["#2ecc71" if o >= 1 else "#e74c3c" for o in ors]

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(plot_vars))
bars = ax.bar(x_pos, ors, color=colors, alpha=0.85, width=0.5,
              yerr=[lo_err, hi_err], capsize=6,
              error_kw=dict(elinewidth=1.4, ecolor="black"))

ax.axhline(1.0, color="gray", linewidth=1.2, linestyle="--", label="OR = 1 (no effect)")

def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

for i, (o, p, lo, hi) in enumerate(zip(ors, pvals, lo_err, hi_err)):
    top = o + hi
    ax.text(i, top + 0.04, f"{o:.2f}{sig_stars(p)}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(display_labels, fontsize=10)
ax.set_ylabel("Odds Ratio (95% CI)")
ax.set_title(
    "Logistic regression — odds ratios for winning a race (2003–2022)\n"
    "Predictor: win ~ pole + era (2010-2019, 2020-2022) + circuit type",
    fontsize=11,
)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

legend_text = "*** p<0.001   ** p<0.01   * p<0.05   ns p≥0.05"
fig.text(0.5, -0.02, legend_text, ha="center", fontsize=8, color="gray", style="italic")

fig.tight_layout()
fig.savefig(out_plot, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to {out_plot}")
