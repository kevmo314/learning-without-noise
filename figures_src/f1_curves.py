"""Figure 1: the task is learned early; forgetting keeps growing with training, faster at higher learning rates,
and the EMA of the same run forgets less. Six seeds per learning rate, mean +/- 95% CI."""

import matplotlib.pyplot as plt

from lwn.fig import COLORS, SEEDS, band, context, deltas, finish
from lwn.runs import run

LRS = (1e-4, 3e-4, 1e-3)
STEPS = (125, 250, 500, 1000, 2000)

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    data = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, lr in zip(COLORS, LRS):
        runs = [run(ctx, lr=lr, seed=s, steps=2000, snapshots=STEPS) for s in SEEDS]
        for kind, style in (("raw", "-"), ("ema", "--")):
            ys = [[deltas(ctx.eval_delta(r["snaps"][t][kind]), base) for t in STEPS] for r in runs]
            data[f"{lr:g}/{kind}"] = ys
            for ax, k in zip(axes, ("task", "bio_nll", "lm")):
                band(ax, STEPS, [[y[k] for y in row] for row in ys], f"LR {lr:g} {kind}", c, style)
    for ax, title in zip(axes, ("Task accuracy (held out)", "Knowledge loss: Δ NLL vs base", "Language loss: Δ vs base")):
        ax.set_xscale("log")
        ax.set_xlabel("fine-tuning steps")
        ax.set_title(title)
    axes[0].axhline(base["task"]["acc"], color="#868e96", lw=1, ls=":")
    axes[0].legend(fontsize=7, ncol=2)
    finish(fig, "f1_curves", {"steps": STEPS, "base": base, "series": data})
