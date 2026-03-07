import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
out_path = os.path.join(PROJECT_ROOT, "plots", "viz1_pole_win_rate.png")

df = pd.read_csv(merged_path)
races = pd.read_csv(races_path)[["raceId", "year"]]
df = df.merge(races, on="raceId")
df = df[(df["year"] >= 2003) & (df["year"] <= 2022)]

pole_sitters = df[df["grid"] == 1][["raceId", "year", "position"]].copy()
pole_sitters["pole_win"] = (pole_sitters["position"] == 1).astype(int)

season_stats = pole_sitters.groupby("year").agg(
    total_races=("raceId", "nunique"),
    pole_wins=("pole_win", "sum"),
).reset_index()
season_stats["conversion_rate"] = (
    100 * season_stats["pole_wins"] / season_stats["total_races"]
)
season_stats["conversion_rate_3y"] = season_stats["conversion_rate"].rolling(3, min_periods=2).mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    season_stats["year"],
    season_stats["conversion_rate"],
    marker="o",
    markersize=4,
    linestyle="-",
    color="steelblue",
    label="Conversion rate",
)
ax.plot(
    season_stats["year"],
    season_stats["conversion_rate_3y"],
    linestyle="-",
    linewidth=2,
    color="coral",
    label="3-year rolling average",
)

ymin = season_stats["conversion_rate"].min()
events = [
    (2011, "DRS introduction"),
    (2014, "Hybrid engines"),
    (2022, "Ground effect regulations"),
]
for yr, label in events:
    ax.axvline(x=yr, color="gray", linestyle="--", linewidth=1)
    ax.annotate(
        label,
        xy=(yr, ymin),
        xytext=(0, -8),
        textcoords="offset points",
        fontsize=7,
        ha="center",
        rotation=90,
    )

ax.set_xlim(2002.5, 2022.5)
ax.set_xticks(range(2003, 2023, 2))
ax.set_xticklabels(range(2003, 2023, 2), rotation=45)
ax.set_xlabel("Season")
ax.set_ylabel("Pole-to-win conversion rate (%)")
ax.set_title("F1 pole-to-win conversion rate by season (2003–2022)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax.grid(True, alpha=0.3)
fig.tight_layout(rect=[0, 0, 0.82, 1])
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved", out_path)
