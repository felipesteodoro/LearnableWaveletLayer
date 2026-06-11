#!/usr/bin/env python
"""Aggregate synthetic experiment results for tese cap5 §5.4."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parents[1] / "tests/synthetic/results/2026-05-09_095357"
OUT = Path("/Users/fteodoro/Dropbox/Doutorado/Tese/figuras/prelim")
OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT = Path(__file__).resolve().parent / "synth_outputs"
CSV_OUT.mkdir(parents=True, exist_ok=True)

ARCHS = ["MLP", "CNN", "LSTM", "CNN_LSTM", "Transformer"]
MODES = ["raw", "db4", "learned_wavelet_no_warmup", "learned_wavelet"]
MODE_LABEL = {
    "raw": "Raw",
    "db4": "Db4",
    "learned_wavelet_no_warmup": "LWT",
    "learned_wavelet": "LWT-Warm",
}

rows = []
for arch in ARCHS:
    for mode in MODES:
        d = BASE / f"{arch}_{mode}"
        if not d.exists():
            continue
        for cfg in sorted(d.glob("cfg*")):
            mj = cfg / "metrics.json"
            if not mj.exists():
                continue
            try:
                with mj.open() as f:
                    m = json.load(f)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(m, dict) or "test_rmse" not in m:
                continue
            rows.append({
                "arch": arch,
                "mode": mode,
                "mode_label": MODE_LABEL[mode],
                "cfg": cfg.name,
                "cfg_idx": int(cfg.name.replace("cfg", "")),
                "test_rmse": m.get("test_rmse"),
                "test_mae": m.get("test_mae"),
                "test_r2": m.get("test_r2"),
                "val_rmse": m.get("val_rmse"),
                "epochs_trained": m.get("epochs_trained"),
            })

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} runs.")
df.to_csv(CSV_OUT / "synth_all_runs.csv", index=False)

# 1) Means per (arch, mode)
agg = df.groupby(["arch", "mode_label"]).agg(
    rmse_mean=("test_rmse", "mean"),
    rmse_std=("test_rmse", "std"),
    mae_mean=("test_mae", "mean"),
    mae_std=("test_mae", "std"),
    r2_mean=("test_r2", "mean"),
    r2_std=("test_r2", "std"),
    n=("test_rmse", "count"),
).reset_index()
agg.to_csv(CSV_OUT / "synth_means.csv", index=False)
print("Means saved.")

# 2) Global aggregate per mode
global_agg = df.groupby("mode_label").agg(
    rmse_mean=("test_rmse", "mean"),
    rmse_std=("test_rmse", "std"),
    mae_mean=("test_mae", "mean"),
    mae_std=("test_mae", "std"),
    r2_mean=("test_r2", "mean"),
    r2_std=("test_r2", "std"),
    n=("test_rmse", "count"),
).reindex(["Raw", "Db4", "LWT", "LWT-Warm"]).reset_index()
global_agg.to_csv(CSV_OUT / "synth_global.csv", index=False)

# 3) Best cfg per (arch, mode) by val_rmse
idx = df.groupby(["arch", "mode_label"])["val_rmse"].idxmin()
best = df.loc[idx].sort_values(["arch", "mode_label"]).reset_index(drop=True)
best.to_csv(CSV_OUT / "synth_best.csv", index=False)

# 4) Paired Wilcoxon — pair on (arch, cfg_idx)
comparisons = [
    ("LWT-Warm", "Raw"),
    ("LWT-Warm", "Db4"),
    ("LWT-Warm", "LWT"),
    ("LWT", "Raw"),
    ("LWT", "Db4"),
]
pivot = df.pivot_table(index=["arch", "cfg_idx"], columns="mode_label", values="test_rmse")
paired_rows = []
for a, b in comparisons:
    sub = pivot[[a, b]].dropna()
    if len(sub) < 5:
        continue
    diff = sub[a] - sub[b]  # negative = a better
    try:
        stat, p = wilcoxon(diff, alternative="two-sided")
    except Exception:
        stat, p = float("nan"), float("nan")
    win = (diff < 0).sum()
    n = len(diff)
    paired_rows.append({
        "A": a, "B": b, "n": n,
        "median_delta_rmse": float(np.median(diff)),
        "mean_delta_rmse": float(np.mean(diff)),
        "wins_A": int(win),
        "winrate_A": win / n,
        "wilcoxon_stat": float(stat),
        "p_value_raw": float(p),
    })

paired = pd.DataFrame(paired_rows)
# Holm-Bonferroni
m = len(paired)
order = paired["p_value_raw"].rank(method="first").astype(int)
sorted_idx = paired["p_value_raw"].sort_values().index
holm_adj = np.zeros(m)
prev = 0.0
for k, i in enumerate(sorted_idx):
    adj = (m - k) * paired.loc[i, "p_value_raw"]
    adj = min(1.0, max(adj, prev))
    holm_adj[i] = adj
    prev = adj
paired["p_value_holm"] = holm_adj
paired["significant_0_05"] = paired["p_value_holm"] < 0.05
paired.to_csv(CSV_OUT / "synth_paired.csv", index=False)
print("Paired tests done.")
print(paired.to_string())

# 5) Plot — boxplot of delta-RMSE LWT-Warm vs each baseline
fig, ax = plt.subplots(figsize=(6.5, 4))
data, labels = [], []
for b in ["Raw", "Db4", "LWT"]:
    sub = pivot[["LWT-Warm", b]].dropna()
    data.append((sub["LWT-Warm"] - sub[b]).values)
    labels.append(f"LWT-Warm − {b}")
bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#cccccc")
ax.axhline(0, color="r", ls="--", lw=0.8)
ax.set_ylabel(r"$\Delta$ RMSE")
ax.set_title("LWT-Warm vs baselines (negativo = LWT-Warm melhor)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_synth_paired_boxplot.pdf")
plt.close()

# 6) Win-rate bar plot per arch (LWT-Warm vs Raw)
fig, ax = plt.subplots(figsize=(6.5, 4))
arch_winrates = {}
for arch in ARCHS:
    sub_df = df[df["arch"] == arch]
    p = sub_df.pivot_table(index="cfg_idx", columns="mode_label", values="test_rmse")
    arch_winrates[arch] = {}
    for b in ["Raw", "Db4", "LWT"]:
        ss = p[["LWT-Warm", b]].dropna()
        arch_winrates[arch][b] = (ss["LWT-Warm"] < ss[b]).mean() if len(ss) else float("nan")
xs = np.arange(len(ARCHS))
w = 0.25
for i, b in enumerate(["Raw", "Db4", "LWT"]):
    vals = [arch_winrates[a][b] for a in ARCHS]
    ax.bar(xs + (i - 1) * w, vals, w, label=f"vs {b}")
ax.set_xticks(xs)
ax.set_xticklabels(ARCHS, rotation=15)
ax.axhline(0.5, color="gray", ls="--", lw=0.7)
ax.set_ylabel("Win-rate (LWT-Warm)")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "fig_synth_winrate.pdf")
plt.close()

print("Figures saved to", OUT)
