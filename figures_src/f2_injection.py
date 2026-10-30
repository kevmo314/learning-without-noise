"""Figure 2: pure noise, no learning. Adding the noise directions that SGD actually produces damages the base model
far more than random noise of the same size. Directions come from 6 seeds' fine-tunes (LR 1e-3, 1000 steps);
random noise is drawn 6 times. Mean +/- 95% CI over the 6 draws."""

import matplotlib.pyplot as plt
import torch

from lwn.analysis import add, dot, mean
from lwn.fig import COLORS, SEEDS, band, context, deltas, finish, load, log_axes
from lwn.runs import run

RHOS = (0.005, 0.01, 0.02, 0.04, 0.08)


def scaled(ctx, d, r):
    s = r * (ctx.w0_sq / dot(d, d)) ** 0.5
    return {n: s * x for n, x in d.items()}


def compute():
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
    series = {k: [[deltas(ctx.eval_delta(scaled(ctx, d, r)), base) for r in RHOS] for d in ds] for k, ds in kinds.items()}
    return {"rhos": RHOS, "base": base, "series": series}

def plot(data):
    rhos = [100 * r for r in data["rhos"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for c, (kind, ys) in zip(COLORS, data["series"].items()):
        for ax, k in zip(axes, ("lm", "kl", "bio_nll")):
            band(ax, rhos, [[y[k] for y in row] for row in ys], kind, c, log=True)
    for ax, title in zip(axes, ("Language loss increase", "KL from base on held-out text", "Knowledge loss increase (NLL)")):
        log_axes(ax, rhos)
        ax.set_xlabel("size of the added noise, ρ (% of ‖W₀‖)")
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    finish(fig, "f2_injection", data)


if __name__ == "__main__":
    data = load("f2_injection")
    if data is None:
        data = compute()
    plot(data)
