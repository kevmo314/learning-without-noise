"""Figure 1: the task is learned early; forgetting keeps growing with training, faster at higher learning rates,
and the EMA of the same run forgets less. Six seeds per learning rate, mean +/- 95% CI."""

import matplotlib.pyplot as plt

from lwn.fig import load, COLORS, SEEDS, band, context, curve, deltas, finish, log_steps
from lwn.runs import run

LRS = (1e-4, 3e-4, 1e-3)
STEPS = log_steps(2000)  # ~32 log-spaced evaluations per run, raw and EMA

def compute():
    ctx = context()
    base = ctx.base_metrics()
    series = {}
    for lr in LRS:
        runs = [run(ctx, lr=lr, seed=s, steps=2000, snapshots=(), evals=STEPS) for s in SEEDS]
        for kind, key in (("raw", "metrics"), ("ema", "ema_metrics")):
            series[f"{lr:g}/{kind}"] = [[deltas(m, base) for _, m in curve(r, key)] for r in runs]
    return {"steps": STEPS, "base": base, "series": series}

def plot(data):
    steps = data["steps"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for c, lr in zip(COLORS, LRS):
        for kind, style in (("raw", "-"), ("ema", "--")):
            ys = data["series"][f"{lr:g}/{kind}"]
            for ax, k in zip(axes, ("task", "bio_nll", "lm")):
                band(ax, steps, [[y[k] for y in row] for row in ys], f"LR {lr:g}" + (" (EMA)" if kind == "ema" else ""),
                     c, style)
    for ax, title in zip(axes, ("Task accuracy (held out)", "Knowledge loss increase (NLL)", "Language loss increase")):
        ax.set_xscale("log")
        ticks = [10, 30, 100, 300, 1000, 2000]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("fine-tuning steps")
        ax.set_title(title)
    axes[0].axhline(data["base"]["task"]["acc"], color="#868e96", lw=1, ls=":")
    axes[0].legend(fontsize=7, ncol=2)
    finish(fig, "f1_curves", data)

if __name__ == "__main__":
    data = load("f1_curves")
    if data is None:
        data = compute()
    plot(data)
