"""Figure 6b: if full-batch training forgets as much as minibatch (figure 6), is it the step size? Full-batch and
minibatch AdamW on the same 2048 examples at two learning rates, with an EMA so the fast jitter (raw - EMA) can be
measured in both. If damage and jitter scale with the LR in full-batch training too, the harmful "noise" is the
optimizer's own oscillation at a large step size, not minibatch sampling."""

import matplotlib.pyplot as plt

from lwn.analysis import add, dot
from lwn.fig import COLORS, SEEDS, band, context, deltas, finish, load
from lwn.runs import run

N = 2048
EPOCHS = (10, 30, 100)
LRS = (3e-4, 1e-3)


def compute():
    ctx = context()
    base = ctx.base_metrics()
    series = {}
    for lr in LRS:
        for name, b in (("full batch", N), ("minibatch 32", 32)):
            snaps = tuple(e * N // b for e in EPOCHS)
            runs = [run(ctx, lr=lr, seed=s, subset=N, batch=b, steps=max(snaps), snapshots=snaps,
                        ema_examples=N if b == N else 1024) for s in SEEDS[:4]]
            rows = []
            for r in runs:
                row = []
                for t in snaps:
                    raw, ema = r["snaps"][t]["raw"], r["snaps"][t]["ema"]
                    j = add((1, raw), (-1, ema))
                    row.append({**deltas(ctx.eval_delta(raw), base), "jitter_share": dot(j, j) / dot(raw, raw)})
                rows.append(row)
            series[f"{name}, LR {lr:g}"] = rows
    return {"epochs": EPOCHS, "base": base, "series": series}


def plot(data):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for i, (name, ys) in enumerate(data["series"].items()):
        c = COLORS[i // 2 * 2 + (0 if "full" in name else 1)]
        style = "-" if "full" in name else "--"
        for ax, k in zip(axes, ("task", "bio_nll", "lm", "jitter_share")):
            band(ax, data["epochs"], [[y[k] for y in row] for row in ys], name, c, style)
    for ax, title in zip(axes, ("Task accuracy", "Knowledge loss increase", "Language loss increase",
                                "Jitter share ‖raw−EMA‖²/‖ΔW‖²")):
        ax.set_xscale("log")
        ax.set_xticks(data["epochs"], [str(e) for e in data["epochs"]])
        ax.minorticks_off()
        ax.set_xlabel("epochs over 2048 examples")
        ax.set_title(title)
    axes[3].set_yscale("log")
    axes[0].legend(fontsize=7)
    finish(fig, "f6b_stepsize", data)


if __name__ == "__main__":
    data = load("f6b_stepsize")
    if data is None:
        data = compute()
    plot(data)
