"""Figure 6: take the noise out of the gradient and the forgetting goes with it. Full-batch AdamW (no sampling
noise) against minibatch AdamW on the same 512 examples at matched epochs, 6 adapter inits each; forgetting is
plotted against task accuracy so progress is matched."""

import matplotlib.pyplot as plt

from lwn.fig import COLORS, SEEDS, band, context, deltas, finish
from lwn.runs import run

N, LR = 512, 1e-3
EPOCHS = (25, 50, 100, 200)

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    arms = {"full batch (no sampling noise)": dict(batch=N, steps=max(EPOCHS), snapshots=EPOCHS),
            "minibatch 32": dict(batch=32, steps=max(EPOCHS) * N // 32, snapshots=tuple(e * N // 32 for e in EPOCHS))}
    data = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, (name, kw) in zip(COLORS, arms.items()):
        runs = [run(ctx, lr=LR, seed=s, subset=N, ema_examples=None, **kw) for s in SEEDS]
        ys = [[deltas(ctx.eval_delta(r["snaps"][t]["raw"]), base) for t in kw["snapshots"]] for r in runs]
        data[name] = ys
        for ax, k in zip(axes, ("task", "bio_nll", "lm")):
            band(ax, EPOCHS, [[y[k] for y in row] for row in ys], name, c)
    for ax, title in zip(axes, ("Task accuracy (held out)", "Knowledge loss: Δ NLL", "Language loss: Δ")):
        ax.set_xscale("log")
        ax.set_xlabel("epochs over the 512 examples")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    finish(fig, "f6_fullbatch", {"epochs": EPOCHS, "base": base, "series": data})
