"""Figure 7: noisier labels -> more forgetting at the same distance, and which regulariser tracks it.
Labels are replaced by a random answer at several rates; for each rate, forgetting of plain training, a fixed cap
and adaptive weight decay toward the base at several strengths. 4 seeds, mean +/- 95% CI."""

import matplotlib.pyplot as plt

from lwn.fig import load, COLORS, band, context, deltas, finish
from lwn.runs import run

NOISE = tuple(round(0.05 * i, 2) for i in range(13))  # 0% to 60% in steps of 5%
SEEDS = (0, 1, 2, 3)
ARMS = {"plain": {}, "cap 2%": {"cap": 0.02}, "AWD 0.3": {"awd": 0.3}, "AWD 1": {"awd": 1.0}, "AWD 3": {"awd": 3.0}}

def compute():
    ctx = context()
    base = ctx.base_metrics()
    series = {}
    for name, kw in ARMS.items():
        series[name] = [[deltas(ctx.eval_delta(run(ctx, lr=1e-3, steps=1000, snapshots=(1000,), seed=s, label_noise=q, **kw)
                                               ["snaps"][1000]["ema"]), base) for q in NOISE] for s in SEEDS]
    return {"noise": NOISE, "base": base, "series": series}


def plot(data):
    x = [100 * q for q in data["noise"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for c, (name, ys) in zip([COLORS[5], COLORS[3], COLORS[0], COLORS[2], COLORS[4]], data["series"].items()):
        band(axes[0], x, [[y["task"] for y in row] for row in ys], name, c)
        band(axes[1], x, [[y["bio_nll"] for y in row] for row in ys], name, c, log=True, floor=1e-3)
        band(axes[2], x, [[y["kl"] for y in row] for row in ys], name, c, log=True, floor=1e-3)
    for ax, title in zip(axes, ("Task accuracy (held out)", "Knowledge loss increase (NLL)", "KL from base")):
        ax.set_xticks(range(0, 61, 10))
        ax.set_xlabel("% of fine-tuning labels randomised")
        ax.set_title(title)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=8)
    finish(fig, "f7_data_quality", data)


if __name__ == "__main__":
    data = load("f7_data_quality")
    if data is None:
        data = compute()
    plot(data)
