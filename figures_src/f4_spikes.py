"""Figure 4: a single LR spike causes silent forgetting. Training loss and task accuracy recover; knowledge and
language do not. One step at k x LR at step 500 of an otherwise identical run (same init, same data order) for
6 seeds; mean +/- 95% CI. The k = 1 curve is the exact counterfactual."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.fig import load, COLORS, SEEDS, band, context, deltas, finish
from lwn.runs import run

FACTORS = (1, 10, 30, 100)
AT, LR = 500, 1e-3

def compute():
    ctx = context()
    base = ctx.base_metrics()
    series = {}
    for k in FACTORS:
        runs = [run(ctx, lr=LR, seed=s, spike=None if k == 1 else (AT, float(k)), eval_every=50, snapshots=(1000,),
                    ema_examples=None) for s in SEEDS]
        steps = [x["step"] for x in runs[0]["curve"] if "metrics" in x and x["step"] >= AT - 100]
        loss = [[np.mean([y["loss"] for y in r["curve"] if t - 25 < y["step"] <= t]) for t in steps] for r in runs]
        met = [[deltas(x["metrics"], base) for x in r["curve"] if "metrics" in x and x["step"] >= AT - 100] for r in runs]
        series[k] = {"steps": steps, "loss": loss, "metrics": met}
    return {"at": AT, "base": base, "series": series}


def plot(data):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for c, (k, s) in zip(COLORS, data["series"].items()):
        label = f"one step at {k}× LR" if int(k) > 1 else "no spike (same run otherwise)"
        band(axes[0], s["steps"], s["loss"], label, c, log=True)
        band(axes[1], s["steps"], [[m["task"] for m in row] for row in s["metrics"]], label, c)
        band(axes[2], s["steps"], [[m["bio_nll"] for m in row] for row in s["metrics"]], label, c, log=True, floor=1e-2)
        band(axes[3], s["steps"], [[m["lm"] for m in row] for row in s["metrics"]], label, c, log=True, floor=1e-2)
    for ax, title in zip(axes, ("Training loss (recovers)", "Task accuracy (mostly recovers)",
                                "Knowledge loss increase (does not)", "Language loss increase (does not)")):
        ax.axvline(data["at"], color="#868e96", ls=":", lw=1)
        ax.set_xlabel("step")
        ax.set_title(title)
    for ax in (axes[0], axes[2], axes[3]):
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    finish(fig, "f4_spikes", data)


if __name__ == "__main__":
    data = load("f4_spikes")
    if data is None:
        data = compute()
    plot(data)
