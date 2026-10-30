"""Figure 3: removing noise removes forgetting. Each arm sees the same examples from the same starting point; points
are mean +/- 95% CI over 6 seeds (seed averages use disjoint seed groups where possible)."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.analysis import mean
from lwn.fig import COLORS, SEEDS, context, deltas, finish, mean_ci
from lwn.runs import run

LR, STEPS = 1e-3, 1000
ARMS = {
    "baseline": {},
    "lower LR (÷3)": {"lr": LR / 3},
    "batch ×4": {"batch": 128, "steps": STEPS // 4, "snapshots": (STEPS // 4,)},
    "cosine decay": {"schedule": "cosine"},
    "cap 2%": {"cap": 0.02},
    "adaptive WD": {"awd": 0.05},
    "GPM 0.99": {"method": "gpm", "gpm_energy": 0.99},
}

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    rows = {}
    for name, kw in ARMS.items():
        kw = {"lr": LR, "steps": STEPS, "snapshots": (STEPS,), **kw}
        runs = [run(ctx, seed=s, **kw) for s in SEEDS]
        snaps = [r["snaps"][max(r["snaps"])] for r in runs]
        rows[name] = [deltas(ctx.eval_delta(s["raw"]), base) for s in snaps]
        if name == "baseline":
            rows["EMA export"] = [deltas(ctx.eval_delta(s["ema"]), base) for s in snaps]
            cd = [run(ctx, seed=s, lr=LR, steps=STEPS, snapshots=(STEPS,), cooldown_from=(int(STEPS / 1.2),)) for s in SEEDS]
            rows["cooldown (last 17%)"] = [deltas(ctx.eval_delta(next(iter(r["branches"].values()))["raw"]), base) for r in cd]
            emas = [s["ema"] for s in snaps]
            rows["avg of 2 seeds"] = [deltas(ctx.eval_delta(mean(emas[i:i + 2])), base) for i in (0, 2, 4)]
            rows["avg of 3 seeds"] = [deltas(ctx.eval_delta(mean(emas[i:i + 3])), base) for i in (0, 3)]
            rows["avg of 6 seeds"] = [deltas(ctx.eval_delta(mean(emas)), base)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(rows)
    for ax, k, title in ((axes[0], "bio_nll", "Knowledge loss: Δ NLL vs base (lower is better)"),
                         (axes[1], "lm", "Language loss: Δ vs base (lower is better)")):
        stats = [mean_ci([r[k] for r in rows[n]]) for n in names]
        y = np.arange(len(names))
        ax.barh(y, [m for m, _ in stats], xerr=[c for _, c in stats], color=COLORS[0], alpha=0.85)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.axvline(0, color="#495057", lw=1)
        ax.set_title(title)
    for i, n in enumerate(names):
        t, _ = mean_ci([r["task"] for r in rows[n]])
        axes[1].annotate(f"task {100 * t:.1f}%", (axes[1].get_xlim()[1], i), fontsize=7, ha="right", va="center")
    finish(fig, "f3_mitigations", {"base": base, "rows": rows})
