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
x_min = int(df["grid"].min())
x_max = int(df["grid"].max())
y_min = int(df["position"].min())
y_max = int(df["position"].max())

# Integer-spaced bins so each grid/finish slot gets its own cell
grid_bins = np.arange(x_min - 0.5, x_max + 1.5, 1)
pos_bins  = np.arange(y_min - 0.5, y_max + 1.5, 1)

# Pre-compute regression lines so the combined panel reuses them
reg = {}
for era in era_order:
    sub = df[df["era"] == era]
    slope, intercept, r_val, p_val, _ = stats.linregress(sub["grid"], sub["position"])
    reg[era] = dict(slope=slope, intercept=intercept, r=r_val, n=len(sub))

fig, axes = plt.subplots(1, 4, figsize=(23, 5.5))

# ── Panels 0-2: per-era heatmaps ────────────────────────────────────────────
for i, era in enumerate(era_order):
    ax = axes[i]
    sub = df[df["era"] == era]
    _, _, _, img = ax.hist2d(
        sub["grid"], sub["position"],
        bins=[grid_bins, pos_bins],
        cmap="Blues",
    )
    plt.colorbar(img, ax=ax, label="Count", shrink=0.85)

    r = reg[era]
    x_line = np.array([x_min, x_max])
    ax.plot(x_line, r["slope"] * x_line + r["intercept"],
            color=era_colors[i], linewidth=2, label="OLS fit")

    ax.text(
        0.05, 0.95,
        f"r  = {r['r']:.3f}\nr² = {r['r']**2:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Grid position")
    ax.set_ylabel("Finishing position")
    ax.set_title(f"{era}\n(n={r['n']:,} entries)")
    ax.set_aspect("equal", adjustable="box")

ax4 = axes[3]
x_line = np.array([x_min, x_max])
for i, era in enumerate(era_order):
    r = reg[era]
    ax4.plot(
        x_line, r["slope"] * x_line + r["intercept"],
        color=era_colors[i], linewidth=2.5,
        label=f"{era}  (r = {r['r']:.3f})",
    )

ax4.set_xlim(x_min, x_max)
ax4.set_ylim(y_min, y_max)
ax4.set_xlabel("Grid position")
ax4.set_ylabel("Finishing position")
ax4.set_title("Regression comparison\n(all eras overlaid)")
ax4.legend(fontsize=9, loc="upper left")
ax4.set_aspect("equal", adjustable="box")
ax4.grid(True, alpha=0.3)

fig.suptitle(
    "Grid position vs finishing position by era (2003–2022)\n"
    "Heatmap density reveals point overlap; Pearson r increases over time — "
    "r² = proportion of finishing-position variance explained by grid",
    fontsize=13,
    y=1.03,
)
fig.tight_layout()
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved", out_path)
