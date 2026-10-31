"""Figure 6: does forgetting need sampling noise? (It does not: full-batch training forgets about as much.) Full-batch AdamW (no sampling
noise) against minibatch AdamW on the same 512 examples at matched epochs, 6 adapter inits each; forgetting is
plotted against task accuracy so progress is matched."""

import matplotlib.pyplot as plt

from lwn.fig import load, COLORS, SEEDS, band, context, curve, deltas, finish, log_steps
from lwn.runs import run

N, LR = 512, 1e-3
EPOCHS = log_steps(200, first=2, n=28)  # dense in-training evaluation, in epochs

def compute():
    ctx = context()
    base = ctx.base_metrics()
    arms = {"full batch (no sampling noise)": dict(batch=N, steps=max(EPOCHS), evals=EPOCHS),
            "minibatch 32": dict(batch=32, steps=max(EPOCHS) * N // 32, evals=tuple(e * N // 32 for e in EPOCHS))}
    series = {}
    for name, kw in arms.items():
        runs = [run(ctx, lr=LR, seed=s, subset=N, ema_examples=None, snapshots=(), **kw) for s in SEEDS]
        series[name] = [[deltas(m, base) for _, m in curve(r)] for r in runs]
    return {"epochs": EPOCHS, "base": base, "series": series}


def plot(data):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for c, (name, ys) in zip(COLORS, data["series"].items()):
        for ax, k in zip(axes, ("task", "bio_nll", "lm")):
            band(ax, data["epochs"], [[y[k] for y in row] for row in ys], name, c)
    for ax, title in zip(axes, ("Task accuracy (held out)", "Knowledge loss increase (NLL)", "Language loss increase")):
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200], ["2", "5", "10", "20", "50", "100", "200"])
        ax.minorticks_off()
        ax.set_xlabel("epochs over the 512 examples")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    finish(fig, "f6_fullbatch", data)


if __name__ == "__main__":
    data = load("f6_fullbatch")
    if data is None:
        data = compute()
    plot(data)
