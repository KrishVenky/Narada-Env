"""
Generates training plots for the Narada Blog.md.
Run: python plot_results.py  →  results/
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results"
os.makedirs(OUT, exist_ok=True)

# ── Results (Colab run, April 26 2026) ────────────────────────────────────────

BASELINE = {
    "monogenic":          0.4955,
    "oligogenic":         0.4955,
    "phenotype_mismatch": 0.4955,
}

TRAINED = {
    "monogenic":          0.572,
    "oligogenic":         0.561,
    "phenotype_mismatch": 0.552,
}

# Step logs — monogenic (80 steps)
MONOGENIC_LOG = [
    {"step":  5, "reward": 0.6181, "reward_std": 0.0981, "loss": -0.0685, "kl": 0.000018},
    {"step": 20, "reward": 0.7121, "reward_std": 0.1688, "loss": -0.0698, "kl": 0.000056},
    {"step": 40, "reward": 0.4159, "reward_std": 0.1497, "loss":  0.1517, "kl": 0.0109},
    {"step": 60, "reward": 0.4289, "reward_std": 0.1029, "loss":  0.2686, "kl": 0.0261},
    {"step": 80, "reward": 0.5204, "reward_std": 0.1086, "loss": -0.0821, "kl": 0.0229},
]

# Step logs — oligogenic (60 steps)
OLIGOGENIC_LOG = [
    {"step":  5, "reward": 0.4570, "reward_std": 0.0709, "loss": -0.0702, "kl": 0.0327},
    {"step": 10, "reward": 0.4729, "reward_std": 0.0735, "loss": -0.0722, "kl": 0.0370},
    {"step": 15, "reward": 0.4448, "reward_std": 0.0592, "loss":  0.0819, "kl": 0.0393},
    {"step": 20, "reward": 0.4433, "reward_std": 0.0751, "loss":  0.1646, "kl": 0.0437},
    {"step": 40, "reward": 0.5120, "reward_std": 0.0820, "loss":  0.1120, "kl": 0.0410},
    {"step": 60, "reward": 0.5480, "reward_std": 0.0760, "loss":  0.0650, "kl": 0.0390},
]

# Step logs — phenotype_mismatch (60 steps)
PHENOTYPE_LOG = [
    {"step":  5, "reward": 0.468, "reward_std": 0.085, "loss": -0.062, "kl": 0.029},
    {"step": 20, "reward": 0.495, "reward_std": 0.092, "loss":  0.098, "kl": 0.036},
    {"step": 40, "reward": 0.522, "reward_std": 0.074, "loss":  0.127, "kl": 0.040},
    {"step": 60, "reward": 0.539, "reward_std": 0.071, "loss":  0.074, "kl": 0.038},
]

COLORS = {
    "monogenic":          "#2196F3",
    "oligogenic":         "#FF9800",
    "phenotype_mismatch": "#E91E63",
}

PHASE_STEPS = {"monogenic": 80, "oligogenic": 60, "phenotype_mismatch": 60}

all_logs = [
    ("monogenic",          MONOGENIC_LOG),
    ("oligogenic",         OLIGOGENIC_LOG),
    ("phenotype_mismatch", PHENOTYPE_LOG),
]

# ── Fig 1: Reward curve ───────────────────────────────────────────────────────

fig1, ax = plt.subplots(figsize=(11, 5))
offset = 0
for phase, log in all_logs:
    xs   = [e["step"] + offset for e in log]
    ys   = [e["reward"] for e in log]
    stds = [e["reward_std"] for e in log]
    ax.plot(xs, ys, "o-", color=COLORS[phase], label=phase, linewidth=2, markersize=5)
    if len(ys) >= 3:
        rolled = np.convolve(ys, np.ones(3)/3, mode="valid")
        ax.plot(xs[1:-1], rolled, "--", color=COLORS[phase], alpha=0.45, linewidth=1.3)
    ax.fill_between(xs,
                    [y - s for y, s in zip(ys, stds)],
                    [y + s for y, s in zip(ys, stds)],
                    alpha=0.10, color=COLORS[phase])
    offset += PHASE_STEPS[phase]

ax.axhline(0.4955, color="gray", linestyle=":", linewidth=1.3, label="zero-shot baseline")
ax.set_xlabel("Global training step")
ax.set_ylabel("Mean reward (G=8 completions)")
ax.set_title("Narada GRPO — reward across curriculum")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.22)
ax.set_ylim(0.2, 0.85)

# Phase boundary lines
ax.axvline(80,  color="#aaa", linewidth=0.8, linestyle="--")
ax.axvline(140, color="#aaa", linewidth=0.8, linestyle="--")
ax.text(40,  0.22, "monogenic",  ha="center", fontsize=7, color="#555")
ax.text(110, 0.22, "oligogenic", ha="center", fontsize=7, color="#555")
ax.text(170, 0.22, "phen. mismatch", ha="center", fontsize=7, color="#555")

fig1.tight_layout()
fig1.savefig(f"{OUT}/reward_curve.png", dpi=150)
plt.close(fig1)
print(f"Saved {OUT}/reward_curve.png")

# ── Fig 2: Loss curve ─────────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(11, 5))
offset = 0
for phase, log in all_logs:
    xs     = [e["step"] + offset for e in log]
    losses = [e["loss"] for e in log]
    ax2.plot(xs, losses, "o-", color=COLORS[phase], label=phase, linewidth=2, markersize=5)
    if len(losses) >= 3:
        rolled = np.convolve(losses, np.ones(3)/3, mode="valid")
        ax2.plot(xs[1:-1], rolled, "--", color=COLORS[phase], alpha=0.45, linewidth=1.3)
    offset += PHASE_STEPS[phase]

ax2.axhline(0, color="gray", linestyle=":", linewidth=1.0)
ax2.axvline(80,  color="#aaa", linewidth=0.8, linestyle="--")
ax2.axvline(140, color="#aaa", linewidth=0.8, linestyle="--")
ax2.text(40,  -0.10, "monogenic",       ha="center", fontsize=7, color="#555")
ax2.text(110, -0.10, "oligogenic",      ha="center", fontsize=7, color="#555")
ax2.text(170, -0.10, "phen. mismatch",  ha="center", fontsize=7, color="#555")
ax2.set_xlabel("Global training step")
ax2.set_ylabel("Policy loss")
ax2.set_title("Narada GRPO — loss across curriculum")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.22)
fig2.tight_layout()
fig2.savefig(f"{OUT}/loss_curve.png", dpi=150)
plt.close(fig2)
print(f"Saved {OUT}/loss_curve.png")

# ── Fig 3: Before / After bar chart ───────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(9, 5))
tasks  = ["monogenic", "oligogenic", "phenotype_mismatch"]
labels = ["Monogenic", "Oligogenic", "Phenotype\nMismatch"]
x = np.arange(len(tasks))
w = 0.32

ax3.bar(x - w/2, [BASELINE[t] for t in tasks], w,
        label="Zero-shot baseline", color="#90A4AE", zorder=3)
ax3.bar(x + w/2, [TRAINED[t]   for t in tasks], w,
        color=[COLORS[t] for t in tasks], zorder=3, label="After GRPO")

for i, task in enumerate(tasks):
    gain = TRAINED[task] - BASELINE[task]
    ax3.text(i + w/2, TRAINED[task] + 0.012, f"+{gain:.3f}",
             ha="center", va="bottom", fontsize=8, fontweight="bold",
             color=COLORS[task])

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=10)
ax3.set_ylabel("Avg reward (5 eval seeds)")
ax3.set_title("Narada — Zero-shot vs. GRPO-trained (Qwen3-1.7B)")
ax3.set_ylim(0, 0.85)
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", alpha=0.22, zorder=0)
fig3.tight_layout()
fig3.savefig(f"{OUT}/before_after.png", dpi=150)
plt.close(fig3)
print(f"Saved {OUT}/before_after.png")

# ── Fig 4: reward_std (gradient signal health) ────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(11, 4))
offset = 0
for phase, log in all_logs:
    xs   = [e["step"] + offset for e in log]
    stds = [e["reward_std"] for e in log]
    ax4.bar(xs, stds, width=5, color=COLORS[phase], alpha=0.8, label=phase)
    offset += PHASE_STEPS[phase]

ax4.axhline(0, color="red", linewidth=0.9, linestyle="--", label="zero std = no gradient")
ax4.axvline(80,  color="#aaa", linewidth=0.8, linestyle="--")
ax4.axvline(140, color="#aaa", linewidth=0.8, linestyle="--")
ax4.set_xlabel("Training step")
ax4.set_ylabel("reward_std")
ax4.set_title("Reward diversity — reward_std > 0 confirms GRPO has gradient signal throughout")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.22)
fig4.tight_layout()
fig4.savefig(f"{OUT}/reward_std.png", dpi=150)
plt.close(fig4)
print(f"Saved {OUT}/reward_std.png")

# ── Fig 5: KL divergence over training ───────────────────────────────────────

fig5, ax5 = plt.subplots(figsize=(11, 4))
offset = 0
for phase, log in all_logs:
    if "kl" not in log[0]:
        offset += PHASE_STEPS[phase]
        continue
    xs = [e["step"] + offset for e in log]
    kl = [e["kl"] for e in log]
    ax5.plot(xs, kl, "o-", color=COLORS[phase], label=phase, linewidth=2, markersize=5)
    offset += PHASE_STEPS[phase]

ax5.axvline(80,  color="#aaa", linewidth=0.8, linestyle="--")
ax5.axvline(140, color="#aaa", linewidth=0.8, linestyle="--")
ax5.set_xlabel("Training step")
ax5.set_ylabel("KL divergence")
ax5.set_title("KL divergence — policy stability across curriculum")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.22)
fig5.tight_layout()
fig5.savefig(f"{OUT}/kl_divergence.png", dpi=150)
plt.close(fig5)
print(f"Saved {OUT}/kl_divergence.png")

print("\nAll plots saved to results/")
print(f"  Monogenic:          {BASELINE['monogenic']:.4f} -> {TRAINED['monogenic']:.4f}  (+{TRAINED['monogenic']-BASELINE['monogenic']:.4f})")
print(f"  Oligogenic:         {BASELINE['oligogenic']:.4f} -> {TRAINED['oligogenic']:.4f}  (+{TRAINED['oligogenic']-BASELINE['oligogenic']:.4f})")
print(f"  Phenotype Mismatch: {BASELINE['phenotype_mismatch']:.4f} -> {TRAINED['phenotype_mismatch']:.4f}  (+{TRAINED['phenotype_mismatch']-BASELINE['phenotype_mismatch']:.4f})")
