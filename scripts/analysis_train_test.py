import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

script_dir   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
merged_path    = os.path.join(PROJECT_ROOT, "f1_merged.csv")
races_path     = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_races.csv")
circuits_path  = os.path.join(PROJECT_ROOT, "Formula 1 Dataset Cleaned", "cleaned_circuits.csv")
out_summary    = os.path.join(PROJECT_ROOT, "plots", "analysis_train_test_summary.txt")
out_comparison = os.path.join(PROJECT_ROOT, "plots", "analysis_train_test_comparison.png")
out_era        = os.path.join(PROJECT_ROOT, "plots", "analysis_train_test_era.png")

# Tee all print() output to both stdout and a buffer for the summary file
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

_buf = io.StringIO()
sys.stdout = _Tee(sys.__stdout__, _buf)

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
df       = pd.read_csv(merged_path)
races    = pd.read_csv(races_path)[["raceId", "year", "circuitId"]]
circuits = pd.read_csv(circuits_path)

df = df.merge(races, on="raceId")
df = df.merge(
    circuits[["circuitId"] + (
        ["circuit_name"] if "circuit_name" in circuits.columns else ["name"]
    )],
    on="circuitId",
)
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

df["era"]               = df["year"].apply(era_bucket)
df["pole"]              = (df["grid"]     == 1).astype(int)
df["win"]               = (df["position"] == 1).astype(int)
df["era_2010_2019"]     = (df["era"] == "2010-2019").astype(int)
df["era_2020_2022"]     = (df["era"] == "2020-2022").astype(int)
df["circuit_type_street"] = (df["circuit_type"] == "Street").astype(int)

model_cols = ["win", "pole", "era_2010_2019", "era_2020_2022", "circuit_type_street"]
model_df   = df[model_cols].dropna()

FEATURES_A = ["era_2010_2019", "era_2020_2022", "circuit_type_street"]
FEATURES_B = ["pole", "era_2010_2019", "era_2020_2022", "circuit_type_street"]
SEEDS      = list(range(10))
ERAS       = ["2003-2009", "2010-2019", "2020-2022"]

X_A = model_df[FEATURES_A].values
X_B = model_df[FEATURES_B].values
y   = model_df["win"].values

print("=" * 70)
print("F1 Train/Test Logistic Regression Analysis (2003–2022)")
print("=" * 70)
print(f"\nFull dataset : {len(model_df):,} observations  |  70 % train / 30 % test")
print(f"  Wins (y=1)           : {model_df['win'].sum():,}  ({100*model_df['win'].mean():.1f}%)")
print(f"  Pole sitters         : {model_df['pole'].sum():,}")
print(f"  Street circuit rows  : {model_df['circuit_type_street'].sum():,}")


def evaluate_both(clf, X_tr, X_te, y_tr, y_te):
    """Fit clf on training data; return metrics on both train and test."""
    clf.fit(X_tr, y_tr)
    results = {}
    for split, Xs, ys in [("train", X_tr, y_tr), ("test", X_te, y_te)]:
        y_pred  = clf.predict(Xs)
        y_proba = clf.predict_proba(Xs)[:, 1]
        results[f"acc_{split}"]  = accuracy_score(ys, y_pred)
        results[f"ll_{split}"]   = log_loss(ys, y_proba)
        results[f"auc_{split}"]  = roc_auc_score(ys, y_proba)
    return results


records_A, records_B = [], []

for seed in SEEDS:
    X_A_tr, X_A_te, y_tr, y_te = train_test_split(
        X_A, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_B_tr, X_B_te, _,    _    = train_test_split(
        X_B, y, test_size=0.30, random_state=seed, stratify=y
    )
    records_A.append(
        evaluate_both(LogisticRegression(max_iter=500, random_state=42),
                      X_A_tr, X_A_te, y_tr, y_te)
    )
    records_B.append(
        evaluate_both(LogisticRegression(max_iter=500, random_state=42),
                      X_B_tr, X_B_te, y_tr, y_te)
    )

def mean_std(records, key):
    vals = [r[key] for r in records]
    return np.mean(vals), np.std(vals)

def ms(records, key):
    m, s = mean_std(records, key)
    return m, s

print("\n" + "=" * 70)
print("TRAINING vs TEST COMPARISON  (mean ± std over 10 random splits)")
print("70 % train / 30 % test, stratified by win")
print("=" * 70)

col_w = 18
h0 = f"{'Metric':<12}"
h1 = f"{'Model A — no pole':^{col_w*2+3}}"
h2 = f"{'Model B — with pole':^{col_w*2+3}}"
print(f"{h0}  {h1}  {h2}")
print(f"{'':12}  {'Train':^{col_w}} {'Test':^{col_w}}  {'Train':^{col_w}} {'Test':^{col_w}}")
print("-" * (12 + 2 + col_w*4 + 6))

def fmt(records, key_tr, key_te):
    mt, st = ms(records, key_tr)
    me, se = ms(records, key_te)
    return f"{mt:.4f}±{st:.4f}", f"{me:.4f}±{se:.4f}"

for label, key_base in [("Accuracy", "acc"), ("Log-Loss", "ll"), ("ROC-AUC", "auc")]:
    a_tr, a_te = fmt(records_A, f"{key_base}_train", f"{key_base}_test")
    b_tr, b_te = fmt(records_B, f"{key_base}_train", f"{key_base}_test")
    print(f"{label:<12}  {a_tr:^{col_w}} {a_te:^{col_w}}  {b_tr:^{col_w}} {b_te:^{col_w}}")

print("-" * (12 + 2 + col_w*4 + 6))

auc_A_tr, _ = ms(records_A, "auc_train");  auc_A_te, auc_A_te_std = ms(records_A, "auc_test")
auc_B_tr, _ = ms(records_B, "auc_train");  auc_B_te, auc_B_te_std = ms(records_B, "auc_test")
ll_A_te,  ll_A_te_std  = ms(records_A, "ll_test")
ll_B_te,  ll_B_te_std  = ms(records_B, "ll_test")

auc_diff_pct = (auc_B_te - auc_A_te) * 100
direction    = "improves" if auc_B_te > auc_A_te else "does not improve"
print(
    f"\nInterpretation: Adding pole position as a predictor {direction} "
    f"out-of-sample prediction by {abs(auc_diff_pct):.2f} percentage points "
    f"(mean test ROC-AUC: {auc_A_te:.4f} → {auc_B_te:.4f})."
)


gap    = auc_B_tr - auc_B_te
verdict = (
    "generalizes well (training ≈ test)"
    if gap < 0.03
    else "shows signs of overfitting (training >> test)"
)
print("\n" + "=" * 70)
print("OVERFITTING CHECK — Model B (with pole)")
print("=" * 70)
print(f"  Mean training ROC-AUC : {auc_B_tr:.4f}")
print(f"  Mean test ROC-AUC     : {auc_B_te:.4f}")
print(f"  Gap (train − test)    : {gap:.4f}")
print(f"\n  Verdict: Model B {verdict}.")
if gap < 0.03:
    print("  The small gap indicates the model is not memorising the training")
    print("  data; its discrimination ability is stable across unseen splits.")
else:
    print("  The gap suggests the model captures training-set patterns that do")
    print("  not fully transfer to held-out data.")

era_auc_lists = {era: [] for era in ERAS}
era_sizes     = {}

for era in ERAS:
    era_sub = df[df["era"] == era][model_cols].dropna()
    era_sizes[era] = len(era_sub)
    if len(era_sub) < 20 or era_sub["win"].sum() < 5:
        continue
    X_era = era_sub[["pole"]].values
    y_era = era_sub["win"].values
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_era, y_era, test_size=0.30, random_state=seed, stratify=y_era
        )
        clf_era = LogisticRegression(max_iter=500, random_state=42)
        clf_era.fit(X_tr, y_tr)
        era_auc_lists[era].append(
            roc_auc_score(y_te, clf_era.predict_proba(X_te)[:, 1])
        )

print("\n" + "=" * 70)
print("ERA-SEGMENTED ANALYSIS — Predictor: pole only  (mean ± std, 10 seeds)")
print("=" * 70)
print(f"{'Era':<15} {'n':>8} {'Wins':>8} {'ROC-AUC (mean±std)':>22}")
print("-" * 57)

era_auc_means, era_auc_stds = {}, {}
for era in ERAS:
    era_sub = df[df["era"] == era][model_cols].dropna()
    vals    = era_auc_lists[era]
    if not vals:
        print(f"{era:<15} {'N/A':>8} {'N/A':>8} {'N/A':>22}")
        era_auc_means[era] = np.nan
        era_auc_stds[era]  = np.nan
        continue
    m, s = np.mean(vals), np.std(vals)
    era_auc_means[era] = m
    era_auc_stds[era]  = s
    print(f"{era:<15} {len(era_sub):>8,} {int(era_sub['win'].sum()):>8,} {m:>12.4f} ± {s:.4f}")

print("-" * 57)


color_A = "#5b9bd5"
color_B = "#ed7d31"
n_obs   = len(model_df)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(
    f"Logistic regression — test-set performance  (n = {n_obs:,}, 70/30 stratified split)\n"
    "Model A: era + circuit type only   |   Model B: adds pole position\n"
    "Error bars = ±1 std over 10 random seeds",
    fontsize=10, y=1.03,
)

subplot_cfg = [
    ("Log-Loss",  "ll_test",  ll_A_te,  ll_A_te_std,  ll_B_te,  ll_B_te_std,
     "Log-Loss (lower is better ↓)", True),
    ("ROC-AUC",   "auc_test", auc_A_te, auc_A_te_std, auc_B_te, auc_B_te_std,
     "ROC-AUC (higher is better ↑)", False),
]

for ax, (metric_label, key, va, sa, vb, sb, ylabel, lower_better) in zip(axes, subplot_cfg):
    x     = np.array([0, 1])
    vals  = [va, vb]
    stds  = [sa, sb]
    bars  = ax.bar(
        x, vals, width=0.45,
        color=[color_A, color_B], alpha=0.88,
        yerr=stds, capsize=7,
        error_kw=dict(elinewidth=1.5, ecolor="black"),
    )
    for bar, val, std in zip(bars, vals, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.004,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Model A\n(no pole)", "Model B\n(with pole)"], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(metric_label, fontsize=11, fontweight="bold")
    lo = max(0, min(vals) - max(stds) * 3)
    hi = max(vals) + max(stds) * 4 + 0.02
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(out_comparison, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {out_comparison}")


valid_eras  = [e for e in ERAS if not np.isnan(era_auc_means.get(e, np.nan))]
auc_vals    = [era_auc_means[e] for e in valid_eras]
auc_stds_v  = [era_auc_stds[e]  for e in valid_eras]
era_n_vals  = [era_sizes[e]      for e in valid_eras]
era_colors  = ["#2ecc71", "#3498db", "#e74c3c"][:len(valid_eras)]

fig, ax = plt.subplots(figsize=(7, 5.5))
bars = ax.bar(
    valid_eras, auc_vals,
    color=era_colors, alpha=0.88, width=0.45,
    yerr=auc_stds_v, capsize=8,
    error_kw=dict(elinewidth=1.5, ecolor="black"),
)

for bar, val, std, n in zip(bars, auc_vals, auc_stds_v, era_n_vals):
    # ROC-AUC value above bar
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.012,
        f"{val:.4f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
    # Sample size below bar (inside, near base)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        0.01,
        f"n = {n:,}",
        ha="center", va="bottom", fontsize=9, color="white", fontweight="bold",
    )

ax.axhline(0.5, color="gray", linewidth=1.2, linestyle="--", label="Random baseline (AUC = 0.5)")
ax.set_ylabel("ROC-AUC  (pole position → win)", fontsize=11)
ax.set_xlabel("Era", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title(
    "Predictive power of pole position by era (2003–2022)\n"
    "Logistic regression (pole only), 70/30 split\n"
    "Error bars = ±1 std over 10 random seeds",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out_era, dpi=300, bbox_inches="tight")
plt.close()
print(f"Plot saved to {out_era}")


sys.stdout = sys.__stdout__
summary_text = _buf.getvalue()
with open(out_summary, "w") as f:
    f.write(summary_text)
print(summary_text, end="")
print(f"\nSummary saved to {out_summary}")
