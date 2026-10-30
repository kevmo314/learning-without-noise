"""Figure 8: is the picture specific to Adam? Seed decomposition (signal, drift, jitter energies and their curvature)
for AdamW and for SGD with momentum at learning rates giving similar task accuracy. 4 seeds."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.analysis import add, curvature, energies, mean
from lwn.fig import COLORS, context, deltas, finish
from lwn.runs import run

SEEDS = (0, 1, 2, 3)
ARMS = {"AdamW 3e-4": {"lr": 3e-4}, "AdamW 1e-3": {"lr": 1e-3}, "SGD 0.003": {"lr": 0.003, "optimizer": "sgd"},
        "SGD 0.01": {"lr": 0.01, "optimizer": "sgd"}, "SGD 0.03": {"lr": 0.03, "optimizer": "sgd"}}

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    rows = []
    for name, kw in ARMS.items():
        runs = [run(ctx, seed=s, steps=1000, snapshots=(1000,), **kw) for s in SEEDS]
        raws = [r["snaps"][1000]["raw"] for r in runs]
        emas = [r["snaps"][1000]["ema"] for r in runs]
        e = energies(raws, emas, ctx.w0_sq)
        sig = mean(emas)
        m = [deltas(ctx.eval_delta(d), base) for d in raws]
        rows.append({"arm": name, **{k: float(v) for k, v in e.items()},
                     "task": float(np.mean([x["task"] for x in m])), "bio_nll": float(np.mean([x["bio_nll"] for x in m])),
                     "kl": float(np.mean([x["kl"] for x in m])),
                     "curv_jitter": float(np.mean([curvature(ctx, add((1, r), (-1, q))) for r, q in zip(raws, emas)])),
                     "curv_drift": float(np.mean([curvature(ctx, add((1, q), (-1, sig))) for q in emas])),
                     "curv_signal": curvature(ctx, sig)})
        print(rows[-1], flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(rows))
    for i, (k, c) in enumerate((("signal", COLORS[2]), ("drift", COLORS[0]), ("jitter", COLORS[1]))):
        axes[0].bar(x + (i - 1) * 0.27, [r[k] for r in rows], 0.27, color=c, label=k)
        axes[1].bar(x + (i - 1) * 0.27, [r[f"curv_{k}"] for r in rows], 0.27, color=c, label=k)
    for ax, t in zip(axes, ("Energy of each part (ρ² units)", "Curvature on held-out text")):
        ax.set_xticks(x, [r["arm"] for r in rows], rotation=20)
        ax.set_yscale("log")
        ax.set_title(t)
    axes[0].legend()
    finish(fig, "f8_sgd", {"base": base, "rows": rows})
