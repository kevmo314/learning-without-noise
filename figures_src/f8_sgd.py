"""Figure 8: is the picture specific to Adam? Seed decomposition (signal, drift, jitter energies and their curvature)
for AdamW and for SGD with momentum at learning rates giving similar task accuracy. 4 seeds."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.analysis import add, curvature, energies, mean
from lwn.fig import load, COLORS, context, deltas, finish
from lwn.runs import run

SEEDS = (0, 1, 2, 3)
ARMS = {"AdamW 3e-4": {"lr": 3e-4}, "AdamW 1e-3": {"lr": 1e-3}, "SGD 0.003": {"lr": 0.003, "optimizer": "sgd"},
        "SGD 0.01": {"lr": 0.01, "optimizer": "sgd"}, "SGD 0.03": {"lr": 0.03, "optimizer": "sgd"}}

def compute():
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
    return {"base": base, "rows": rows}


def plot(data):
    rows = data["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    x = np.arange(len(rows))
    for i, (k, c) in enumerate((("signal", COLORS[2]), ("drift", COLORS[0]), ("jitter", COLORS[1]))):
        axes[0].bar(x + (i - 1) * 0.27, [r[k] for r in rows], 0.27, color=c, label=k)
        axes[1].bar(x + (i - 1) * 0.27, [r[f"curv_{k}"] for r in rows], 0.27, color=c, label=k)
    for ax, t in zip(axes[:2], ("Energy of each part (ρ² units)", "Curvature of each part on held-out text")):
        ax.set_xticks(x, [r["arm"] for r in rows], rotation=20)
        ax.set_yscale("log")
        ax.set_title(t)
    axes[0].legend()
    # Does curvature-weighted energy (≈ 2 x predicted KL) line the optimizers up where plain distance does not?
    for r, c in zip(rows, [COLORS[0], COLORS[0], COLORS[1], COLORS[1], COLORS[1]]):
        plain = r["signal"] + r["drift"] + r["jitter"]
        weighted = sum(r[k] * r[f"curv_{k}"] for k in ("signal", "drift", "jitter")) / 2
        axes[2].scatter(plain, r["kl"], marker="o", facecolors="none", edgecolors=c, s=50)
        axes[2].scatter(weighted, r["kl"], marker="o", color=c, s=50)
        axes[2].annotate(r["arm"], (weighted, r["kl"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    axes[2].plot([1e-4, 1], [1e-4, 1], ":", color="#868e96", lw=1)
    axes[2].set_xlim(5e-5, 1)
    axes[2].set_ylim(5e-2, 1)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("open: plain energy ρ²   filled: curvature-weighted energy / 2")
    axes[2].set_ylabel("measured KL from base")
    axes[2].set_title("Blue AdamW, orange SGD")
    finish(fig, "f8_sgd", data)


if __name__ == "__main__":
    data = load("f8_sgd")
    if data is None:
        data = compute()
    plot(data)
