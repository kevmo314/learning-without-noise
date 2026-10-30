"""Figure 5: where the update lands decides how much it costs. For each adapter type (6 seeds), damage per unit
distance against the share of the update acting on the input directions pretraining text uses, and against its
curvature on held-out text. Methods: LoRA, MiLoRA, PiSSA and OLoRA (LR/10, matched step size), GPM, full FT."""

import matplotlib.pyplot as plt

from lwn.analysis import curvature, dot
from lwn.fig import load, COLORS, SEEDS, context, deltas, finish, mean_ci
from lwn.runs import run

METHODS = {"LoRA": {"method": "lora"}, "MiLoRA": {"method": "milora"}, "PiSSA (LR/10)": {"method": "pissa", "lr": 1e-4},
           "OLoRA (LR/10)": {"method": "olora", "lr": 1e-4}, "GPM 0.9": {"method": "gpm", "gpm_energy": 0.9},
           "GPM 0.99": {"method": "gpm", "gpm_energy": 0.99}, "full FT (LR 1e-4)": {"method": "full", "lr": 1e-4}}

def compute():
    ctx = context()
    base = ctx.base_metrics()
    U = ctx.activation_basis(0.9)
    out = {}
    return {"base": base, "methods": out}


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for c, (name, pts) in zip(COLORS + ["#212529"], data["methods"].items()):
        my, cy = mean_ci([p["kl_per_rho2"] for p in pts])
        for ax, x in ((axes[0], "share_in"), (axes[1], "curv")):
            mx, cx = mean_ci([p[x] for p in pts])
            ax.errorbar(mx, my, xerr=cx, yerr=cy, fmt="o", color=c, ms=7, capsize=2)
            ax.annotate(name, (mx, my), fontsize=8, xytext=(6, -3), textcoords="offset points")
    axes[0].set_xlabel("share of the update on input directions pretraining text uses")
    axes[1].set_xlabel("curvature of the update direction on held-out text")
    axes[1].set_xscale("log")
    axes[0].set_xlim(-0.05, 1.05)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("KL from base per unit ρ²")
    finish(fig, "f5_subspace", data)


if __name__ == "__main__":
    data = load("f5_subspace")
    if data is None:
        data = compute()
    plot(data)
