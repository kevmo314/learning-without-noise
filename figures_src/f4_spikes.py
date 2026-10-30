"""Figure 4: a single LR spike causes silent forgetting. Training loss and task accuracy recover; knowledge and
language do not. One step at k x LR at step 500 of an otherwise identical run (same init, same data order) for
6 seeds; mean +/- 95% CI. The k = 1 curve is the exact counterfactual."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.fig import COLORS, SEEDS, band, context, deltas, finish
from lwn.runs import run

FACTORS = (1, 10, 30, 100)
AT, LR = 500, 1e-3

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    data = {}
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for c, k in zip(COLORS, FACTORS):
        runs = [run(ctx, lr=LR, seed=s, spike=None if k == 1 else (AT, float(k)), eval_every=50, snapshots=(1000,),
                    ema_examples=None) for s in SEEDS]
        steps = [x["step"] for x in runs[0]["curve"] if "metrics" in x and x["step"] >= AT - 100]
        loss = [[np.mean([y["loss"] for y in r["curve"] if t - 25 < y["step"] <= t]) for t in steps] for r in runs]
        met = [[deltas(x["metrics"], base) for x in r["curve"] if "metrics" in x and x["step"] >= AT - 100] for r in runs]
        data[k] = {"steps": steps, "loss": loss, "metrics": met}
        band(axes[0], steps, loss, f"spike ×{k}" if k > 1 else "no spike", c)
        for ax, key in zip(axes[1:], ("task", "bio_nll", "lm")):
            band(ax, steps, [[m[key] for m in row] for row in met], f"spike ×{k}" if k > 1 else "no spike", c)
    for ax, title in zip(axes, ("Training loss", "Task accuracy (held out)", "Knowledge loss: Δ NLL", "Language loss: Δ")):
        ax.axvline(AT, color="#868e96", ls=":", lw=1)
        ax.set_xlabel("step")
        ax.set_title(title)
    axes[0].set_yscale("log")
    for ax in axes[2:]:
        ax.set_yscale("symlog", linthresh=0.01)
    axes[0].legend(fontsize=8)
    finish(fig, "f4_spikes", {"at": AT, "base": base, "series": data})
