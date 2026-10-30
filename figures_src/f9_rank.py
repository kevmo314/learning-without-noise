"""Figure 9: adapter rank. Higher rank gives the noise more room: drift energy should grow with rank (the 2B runs fit
rho^2 ~ S^2 + k * eta^2 * rank * t) and seeds should agree less, while the task barely changes. 6 seeds per rank."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.analysis import energies
from lwn.fig import COLORS, SEEDS, context, deltas, finish, load, mean_ci
from lwn.runs import run

RANKS = (4, 16, 64)


def compute():
    ctx = context()
    base = ctx.base_metrics()
    rows = []
    for r in RANKS:
        runs = [run(ctx, rank=r, lr=1e-3, steps=1000, snapshots=(1000,), seed=s) for s in SEEDS]
        raws = [x["snaps"][1000]["raw"] for x in runs]
        emas = [x["snaps"][1000]["ema"] for x in runs]
        rows.append({"rank": r, **{k: float(v) for k, v in energies(raws, emas, ctx.w0_sq).items()},
                     "metrics": [deltas(ctx.eval_delta(d), base) for d in raws]})
    return {"base": base, "rows": rows}


def plot(data):
    rows = data["rows"]
    ranks = [r["rank"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for k, c in (("signal", COLORS[2]), ("drift", COLORS[0]), ("jitter", COLORS[1])):
        axes[0].plot(ranks, [r[k] for r in rows], "o-", color=c, label=k, lw=2)
    axes[0].set_yscale("log")
    axes[0].set_title("Energy of each part of the update (ρ²)")
    for ax, k, title in ((axes[1], "bio_nll", "Knowledge loss increase (NLL)"), (axes[2], "task", "Task accuracy")):
        st = [mean_ci([m[k] for m in r["metrics"]]) for r in rows]
        ax.errorbar(ranks, [m for m, _ in st], yerr=[c for _, c in st], fmt="o-", color=COLORS[0], capsize=3, lw=2)
        ax.set_title(title)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ranks, [str(r) for r in ranks])
        ax.minorticks_off()
        ax.set_xlabel("LoRA rank")
    axes[0].legend()
    finish(fig, "f9_rank", data)


if __name__ == "__main__":
    data = load("f9_rank")
    if data is None:
        data = compute()
    plot(data)
