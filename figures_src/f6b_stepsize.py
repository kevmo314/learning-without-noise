"""Figure 6b: if full-batch training forgets as much as minibatch (figure 6), is it the step size? Full-batch and
minibatch AdamW on the same 2048 examples at two learning rates, with an EMA so the fast jitter (raw - EMA) can be
measured in both. If damage and jitter scale with the LR in full-batch training too, the harmful "noise" is the
optimizer's own oscillation at a large step size, not minibatch sampling."""

import matplotlib.pyplot as plt

from lwn.analysis import add, dot
from lwn.fig import COLORS, SEEDS, band, context, curve, deltas, finish, load, log_steps
from lwn.runs import run

N = 2048
EPOCHS = log_steps(100, first=1, n=28)  # dense in-training evaluation, in epochs
LRS = (3e-4, 1e-3)


def compute():
    ctx = context()
    base = ctx.base_metrics()
    series = {}
    for lr in LRS:
        for name, b in (("full batch", N), ("minibatch 32", 32)):
            evals = tuple(e * N // b for e in EPOCHS)
            runs = [run(ctx, lr=lr, seed=s, subset=N, batch=b, steps=max(evals), snapshots=(max(evals),), evals=evals,
                        ema_examples=N if b == N else 1024) for s in SEEDS]
            rows = []
            for r in runs:
                row = [deltas(m, base) for _, m in curve(r)]
                raw, ema = r["snaps"][max(evals)]["raw"], r["snaps"][max(evals)]["ema"]
                j = add((1, raw), (-1, ema))
                row[-1]["jitter_share"] = dot(j, j) / dot(raw, raw)  # at the end of training only
                rows.append(row)
            series[f"{name}, LR {lr:g}"] = rows
    return {"epochs": EPOCHS, "base": base, "series": series}


def plot(data):
    # Colour by learning rate, solid for full batch and dashed for minibatch. (Jitter shares are saved in the JSON
    # but not plotted: the EMA window differs between the arms, so they are not comparable.)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for name, ys in data["series"].items():
        c = COLORS[0] if "0.0003" in name else COLORS[1]
        style = "-" if "full" in name else "--"
        for ax, k in zip(axes, ("task", "bio_nll", "lm")):
            band(ax, data["epochs"], [[y[k] for y in row] for row in ys], name, c, style)
    for ax, title in zip(axes, ("Task accuracy", "Knowledge loss increase (NLL)", "Language loss increase")):
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
        ax.minorticks_off()
        ax.set_xlabel("epochs over 2048 examples")
        ax.set_title(title)
    axes[0].legend(fontsize=7)
    finish(fig, "f6b_stepsize", data)


if __name__ == "__main__":
    data = load("f6b_stepsize")
    if data is None:
        data = compute()
    plot(data)
