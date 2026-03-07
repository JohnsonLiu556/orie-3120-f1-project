import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
out_path = os.path.join(PROJECT_ROOT, "plots", "viz3_grid_vs_finish_era.png")

df = pd.read_csv(merged_path)
races = pd.read_csv(races_path)[["raceId", "year"]]
df = df.merge(races, on="raceId")
df = df[(df["year"] >= 2003) & (df["year"] <= 2022)]


def era_bucket(y):
    if y <= 2009:
        return "2003–2009"
    if y <= 2019:
        return "2010–2019"
    return "2020–2022"

df["era"] = df["year"].apply(era_bucket)
era_order = ["2003–2009", "2010–2019", "2020–2022"]

corr_list = []
for era in era_order:
    sub = df[df["era"] == era]
    r, p = stats.pearsonr(sub["grid"], sub["position"])
    corr_list.append({"era": era, "Pearson r": round(r, 4), "p-value": round(p, 4)})
corr_df = pd.DataFrame(corr_list)
print("Pearson correlation (grid vs finishing position) by era:")
print(corr_df.to_string(index=False))
print()

era_colors = ["#3498db", "#e74c3c", "#9b59b6"]
x_min = df["grid"].min()
x_max = df["grid"].max()
y_min = df["position"].min()
y_max = df["position"].max()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, era in enumerate(era_order):
    ax = axes[i]
    sub = df[df["era"] == era]
    ax.scatter(sub["grid"], sub["position"], alpha=0.15, s=8, c=era_colors[i])
    slope, intercept, r_val, p_val, se = stats.linregress(sub["grid"], sub["position"])
    x_line = np.array([x_min, x_max])
    ax.plot(x_line, slope * x_line + intercept, color=era_colors[i], linewidth=2)
    ax.text(
        0.05, 0.95,
        f"r  = {r_val:.3f}\nr² = {r_val**2:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Grid position")
    ax.set_ylabel("Finishing position")
    ax.set_title(f"{era}\n(n={len(sub):,} entries)")
    ax.set_aspect("equal", adjustable="box")
fig.suptitle(
    "Grid position vs finishing position by era (2003–2022)\nPearson r increases over time — r² = proportion of finishing position variance explained by grid",
    fontsize=14,
    y=1.06,
)
fig.tight_layout()
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved", out_path)
