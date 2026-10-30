"""Figure 11: putting it together. The noise fixes (lower LR, EMA export), steering (GPM on the pretraining mixture)
and replay (an LM loss on pretraining text each step) alone and combined, against the baseline. 6 seeds."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.fig import COLORS, SEEDS, context, deltas, finish, load, mean_ci
from lwn.runs import run

GPM = {"method": "gpm", "gpm_energy": 0.9, "gpm_source": "mix"}
ARMS = {
    "baseline (LR 1e-3, raw)": ({"lr": 1e-3}, "raw"),
    "lower LR (3e-4) + EMA": ({"lr": 3e-4}, "ema"),
    "GPM (pretraining mix)": ({"lr": 1e-3, **GPM}, "ema"),
    "replay 0.3": ({"lr": 1e-3, "replay": 0.3}, "ema"),
    "replay 1.0": ({"lr": 1e-3, "replay": 1.0}, "ema"),
    "lower LR + EMA + GPM": ({"lr": 3e-4, **GPM}, "ema"),
    "lower LR + EMA + GPM + replay 0.3": ({"lr": 3e-4, "replay": 0.3, **GPM}, "ema"),
}


def compute():
    ctx = context()
    base = ctx.base_metrics()
    rows = {}
    for name, (kw, kind) in ARMS.items():
        rows[name] = [deltas(ctx.eval_delta(run(ctx, steps=1000, snapshots=(1000,), seed=s, **kw)["snaps"][1000][kind]),
                             base) for s in SEEDS]
    return {"base": base, "rows": rows}


def plot(data):
    rows = data["rows"]
    names = list(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    y = np.arange(len(names))
    colors = [COLORS[5]] + [COLORS[0]] * 4 + [COLORS[2]] * 2
    for ax, k, title in ((axes[0], "bio_nll", "Knowledge loss increase (NLL)"),
                         (axes[1], "lm", "Language loss increase"), (axes[2], "task", "Task accuracy")):
        st = [mean_ci([r[k] for r in rows[n]]) for n in names]
        ax.barh(y, [m for m, _ in st], xerr=[c for _, c in st], color=colors, error_kw={"capsize": 2})
        ax.axvline(0, color="#495057", lw=1)
        ax.set_title(title)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[2].set_xlim(0.8, 0.95)
    finish(fig, "f11_combined", data)


if __name__ == "__main__":
    data = load("f11_combined")
    if data is None:
        data = compute()
    plot(data)
