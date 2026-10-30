"""Figure 7: noisier labels -> more forgetting at the same distance, and which regulariser tracks it.
Labels are replaced by a random answer at several rates; for each rate, forgetting of plain training, a fixed cap
and adaptive weight decay toward the base at several strengths. 4 seeds, mean +/- 95% CI."""

import matplotlib.pyplot as plt

from lwn.fig import COLORS, band, context, deltas, finish
from lwn.runs import run

NOISE = (0.0, 0.2, 0.4, 0.6)
SEEDS = (0, 1, 2, 3)
ARMS = {"plain": {}, "cap 2%": {"cap": 0.02}, "AWD 0.3": {"awd": 0.3}, "AWD 1": {"awd": 1.0}, "AWD 3": {"awd": 3.0}}

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    data = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, (name, kw) in zip(COLORS, ARMS.items()):
        ys = [[deltas(ctx.eval_delta(run(ctx, lr=1e-3, steps=1000, snapshots=(1000,), seed=s, label_noise=q, **kw)
                                     ["snaps"][1000]["ema"]), base) for q in NOISE] for s in SEEDS]
        data[name] = ys
        for ax, k in zip(axes, ("task", "bio_nll", "kl")):
            band(ax, [100 * q for q in NOISE], [[y[k] for y in row] for row in ys], name, c)
    for ax, title in zip(axes, ("Task accuracy (held out)", "Knowledge loss: Δ NLL", "KL from base")):
        ax.set_xlabel("% of fine-tuning labels randomised")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    finish(fig, "f7_data_quality", {"noise": NOISE, "base": base, "series": data})
