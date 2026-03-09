import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
out_path = os.path.join(PROJECT_ROOT, "plots", "viz_win_rate_by_grid.png")

df = pd.read_csv(merged_path)
races = pd.read_csv(races_path)[["raceId", "year"]]
df = df.merge(races, on="raceId")
df = df[(df["year"] >= 2003) & (df["year"] <= 2022)]

df["win"] = (df["position"] == 1).astype(int)
df["grid_bucket"] = df["grid"].apply(lambda g: g if g <= 10 else 11)

grid_stats = df.groupby("grid_bucket").agg(
    total=("win", "count"),
    wins=("win", "sum"),
).reset_index()
grid_stats["win_rate"] = 100 * grid_stats["wins"] / grid_stats["total"]

labels = [str(g) if g <= 10 else "11+" for g in grid_stats["grid_bucket"]]
colors = ["steelblue" if g <= 10 else "lightgray" for g in grid_stats["grid_bucket"]]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, grid_stats["win_rate"], color=colors, edgecolor="white")

for bar, rate, total in zip(bars, grid_stats["win_rate"], grid_stats["total"]):
    ax.annotate(
        f"{rate:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
    ax.annotate(
        f"n={total}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color="gray",
    )

ax.set_xlabel("Starting grid position")
ax.set_ylabel("Win rate (%)")
ax.set_title("Win rate by starting grid position (2003–2022)\nAdvantage drops sharply after P1; positions 11+ rarely produce winners")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved", out_path)
