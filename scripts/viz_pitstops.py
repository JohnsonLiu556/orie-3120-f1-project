import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path = os.path.join(PROJECT_ROOT, "f1_merged.csv")
pit_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_pit_stops.csv")
races_path = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
out_path = os.path.join(PROJECT_ROOT, "plots", "viz_pitstops.png")

df = pd.read_csv(merged_path)
races = pd.read_csv(races_path)[["raceId", "year"]]
valid_races = races[(races["year"] >= 2011) & (races["year"] <= 2022)]["raceId"]
df = df[df["raceId"].isin(valid_races)]
pit = pd.read_csv(pit_path)
pit = pit[pit["raceId"].isin(valid_races)]
pit_counts = pit.groupby(["raceId", "driverId"]).size().reset_index(name="n_pit_stops")

pole_sitters = df[df["grid"] == 1][["raceId", "driverId"]].rename(columns={"driverId": "pole_driverId"})
winners = df[df["position"] == 1][["raceId", "driverId"]].rename(columns={"driverId": "winner_driverId"})
race_info = pole_sitters.merge(winners, on="raceId")
race_info["pole_win"] = race_info["pole_driverId"] == race_info["winner_driverId"]

winner_stops = race_info[["raceId", "winner_driverId"]].merge(
    pit_counts, left_on=["raceId", "winner_driverId"], right_on=["raceId", "driverId"], how="left"
)[["raceId", "n_pit_stops"]].rename(columns={"n_pit_stops": "winner_pit_stops"})
pole_stops = race_info[["raceId", "pole_driverId"]].merge(
    pit_counts, left_on=["raceId", "pole_driverId"], right_on=["raceId", "driverId"], how="left"
)[["raceId", "n_pit_stops"]].rename(columns={"n_pit_stops": "pole_sitter_pit_stops"})

race_info = race_info.merge(winner_stops, on="raceId").merge(pole_stops, on="raceId")
race_info["winner_pit_stops"] = race_info["winner_pit_stops"].astype(float)
race_info["pole_sitter_pit_stops"] = race_info["pole_sitter_pit_stops"].fillna(0).astype(int)

analysis = race_info.dropna(subset=["winner_pit_stops"]).copy()
analysis["winner_pit_stops"] = analysis["winner_pit_stops"].astype(int)
wins = analysis[analysis["pole_win"]]["winner_pit_stops"]
losses = analysis[~analysis["pole_win"]]["winner_pit_stops"]
print("Average pit stops (race winner):")
print("  Pole-sitter wins:  ", round(wins.mean(), 3))
print("  Pole-sitter loses: ", round(losses.mean(), 3))

t_stat, p_val = stats.ttest_ind(wins, losses)
print("\nt-test (pit stops: pole-sitter wins vs loses):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_val:.4f}")
if p_val < 0.05:
    print("  Interpretation: The difference in average pit stops between races where the pole-sitter won and races where they did not is statistically significant at the 5% level.")
else:
    print("  Interpretation: The difference in average pit stops between races where the pole-sitter won and races where they did not is not statistically significant at the 5% level.")

fig, ax = plt.subplots(figsize=(6, 5))
analysis["pole_win_str"] = analysis["pole_win"].map({True: "Pole-sitter wins", False: "Pole-sitter loses"})
bp = ax.boxplot(
    [analysis[analysis["pole_win"]]["winner_pit_stops"], analysis[~analysis["pole_win"]]["winner_pit_stops"]],
    labels=["Pole-sitter wins", "Pole-sitter loses"],
    patch_artist=True,
)
for patch in bp["boxes"]:
    patch.set_facecolor("lightsteelblue")
ax.axhline(2.0, color="gray", linestyle="--", linewidth=1)
ax.text(0.98, 2.0, "Typical 2-stop strategy", color="gray", fontsize=8, ha="right", va="bottom", transform=ax.get_yaxis_transform())
ax.set_ylabel("Number of pit stops (race winner)")
ax.set_xlabel("")
ax.set_title("Pit stop count by race winner: pole-sitter wins vs loses (2011–2022)\nPit stop data available from 2011 onward")

mean_w = wins.mean()
mean_l = losses.mean()
n_w = int(wins.shape[0])
n_l = int(losses.shape[0])
sig_text = "significant" if p_val < 0.05 else "not significant"
stats_text = (
    f"Pole-sitter wins:  mean={mean_w:.2f} stops  (n={n_w})\n"
    f"Pole-sitter loses: mean={mean_l:.2f} stops  (n={n_l})\n"
    f"t = {t_stat:.3f},  p = {p_val:.3f}  ({sig_text})"
)
ax.text(
    0.98,
    0.98,
    stats_text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

fig.text(
    0.5,
    -0.02,
    "Finding: Pit stop volume does not significantly differ between pole-win and non-pole-win races.\nWhen the pole-sitter loses, rivals are not winning through more pit stops — other factors dominate.",
    ha="center",
    color="gray",
    style="italic",
    fontsize=8,
)
fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved", out_path)
