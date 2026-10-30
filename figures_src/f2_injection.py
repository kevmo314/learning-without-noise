"""Figure 2: pure noise, no learning. Adding the noise directions that SGD actually produces damages the base model
far more than random noise of the same size. Directions come from 6 seeds' fine-tunes (LR 1e-3, 1000 steps);
random noise is drawn 6 times. Mean +/- 95% CI over the 6 draws."""

import matplotlib.pyplot as plt
import torch

from lwn.analysis import add, dot, mean
from lwn.fig import COLORS, SEEDS, band, context, deltas, finish
from lwn.runs import run

RHOS = (0.005, 0.01, 0.02, 0.04, 0.08)


def scaled(ctx, d, r):
    s = r * (ctx.w0_sq / dot(d, d)) ** 0.5
    return {n: s * x for n, x in d.items()}


if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    runs = [run(ctx, lr=1e-3, seed=s, steps=1000, snapshots=(1000,)) for s in SEEDS]
    raws = [r["snaps"][1000]["raw"] for r in runs]
    emas = [r["snaps"][1000]["ema"] for r in runs]
    sig = mean(emas)
    W = {n: ctx.base_state[n + ".weight"] for n in ctx.names}
    kinds = {"isotropic": [], "random low-rank": [], "SGD drift": [], "SGD jitter": []}
    for s in SEEDS:
        g = torch.Generator(device="cuda").manual_seed(s)
        kinds["isotropic"].append({n: torch.randn(w.shape, generator=g, device="cuda") * w.std() for n, w in W.items()})
        kinds["random low-rank"].append({n: torch.randn(w.shape[0], 16, generator=g, device="cuda")
                                         @ torch.randn(16, w.shape[1], generator=g, device="cuda") for n, w in W.items()})
        kinds["SGD drift"].append(add((1, emas[s]), (-1, sig)))
        kinds["SGD jitter"].append(add((1, raws[s]), (-1, emas[s])))
    data = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, (kind, ds) in zip(COLORS, kinds.items()):
        ys = [[deltas(ctx.eval_delta(scaled(ctx, d, r)), base) for r in RHOS] for d in ds]
        data[kind] = ys
        for ax, k in zip(axes, ("bio_nll", "lm", "kl")):
            band(ax, [100 * r for r in RHOS], [[y[k] for y in row] for row in ys], kind, c)
    for ax, title in zip(axes, ("Knowledge loss: Δ NLL", "Language loss: Δ", "KL from base on held-out text")):
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("size of the perturbation, ρ (% of ‖W₀‖)")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    finish(fig, "f2_injection", {"rhos": RHOS, "base": base, "series": data})
