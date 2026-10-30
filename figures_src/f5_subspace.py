"""Figure 5: where the update lands decides how much it costs. For each adapter type (6 seeds), damage per unit
distance against the share of the update acting on the input directions pretraining text uses, and against its
curvature on held-out text. Methods: LoRA, MiLoRA, PiSSA and OLoRA (LR/10, matched step size), GPM, full FT."""

import matplotlib.pyplot as plt

from lwn.analysis import curvature, dot
from lwn.fig import COLORS, SEEDS, context, deltas, finish, mean_ci
from lwn.runs import run

METHODS = {"LoRA": {"method": "lora"}, "MiLoRA": {"method": "milora"}, "PiSSA (LR/10)": {"method": "pissa", "lr": 1e-4},
           "OLoRA (LR/10)": {"method": "olora", "lr": 1e-4}, "GPM 0.9": {"method": "gpm", "gpm_energy": 0.9},
           "GPM 0.99": {"method": "gpm", "gpm_energy": 0.99}, "full FT (LR 1e-4)": {"method": "full", "lr": 1e-4}}

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    U = ctx.activation_basis(0.9)
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, (name, kw) in zip(COLORS + ["#212529"], METHODS.items()):
        kw = {"lr": 1e-3, "steps": 1000, "snapshots": (1000,), **kw}
        pts = []
        for s in SEEDS:
            d = run(ctx, seed=s, **kw)["snaps"][1000]["ema"]
            rho2 = dot(d, d) / ctx.w0_sq
            inside = {n: U[n] @ (U[n].T @ x) for n, x in d.items()}
            m = deltas(ctx.eval_delta(d), base)
            pts.append({"share_in": dot(inside, inside) / dot(d, d), "curv": curvature(ctx, d),
                        "kl_per_rho2": m["kl"] / rho2, "bio_per_rho2": m["bio_nll"] / rho2, "rho": rho2 ** 0.5, **m})
        out[name] = pts
        for ax, x in ((axes[0], "share_in"), (axes[1], "curv")):
            mx, cx = mean_ci([p[x] for p in pts])
            my, cy = mean_ci([p["kl_per_rho2"] for p in pts])
            ax.errorbar(mx, my, xerr=cx, yerr=cy, fmt="o", color=c, label=name, ms=7, capsize=2)
    axes[0].set_xlabel("share of the update on pretraining-active input directions")
    axes[1].set_xlabel("curvature of the update direction on held-out text")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("KL from base per unit ρ²")
    axes[0].legend(fontsize=7)
    finish(fig, "f5_subspace", {"base": base, "methods": out})
