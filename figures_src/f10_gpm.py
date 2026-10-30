"""Figure 10: protect the directions pretraining actually uses. GPM keeps the adapter's inputs out of the subspace
that pretraining activations occupy. Estimated from stories only it protects language but not the biographies; this
compares that with a basis estimated on the pretraining mixture (biographies included), at two energy thresholds.
6 seeds, mean +/- 95% CI."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.fig import COLORS, SEEDS, context, deltas, finish, load, mean_ci
from lwn.runs import run

ARMS = {"LoRA": {}, "GPM 0.9, stories": {"method": "gpm", "gpm_energy": 0.9},
        "GPM 0.9, pretraining mix": {"method": "gpm", "gpm_energy": 0.9, "gpm_source": "mix"},
        "GPM 0.99, stories": {"method": "gpm", "gpm_energy": 0.99},
        "GPM 0.99, pretraining mix": {"method": "gpm", "gpm_energy": 0.99, "gpm_source": "mix"}}


def compute():
    ctx = context()
    base = ctx.base_metrics()
    rows = {}
    for name, kw in ARMS.items():
        rows[name] = [deltas(ctx.eval_delta(run(ctx, lr=1e-3, steps=1000, snapshots=(1000,), seed=s, **kw)
                                            ["snaps"][1000]["ema"]), base) for s in SEEDS]
    return {"base": base, "rows": rows}


def plot(data):
    rows = data["rows"]
    names = list(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    y = np.arange(len(names))
    colors = [COLORS[5] if n == "LoRA" else COLORS[0] if "stories" in n else COLORS[2] for n in names]
    for ax, k, title in ((axes[0], "bio_nll", "Knowledge loss increase (NLL)"),
                         (axes[1], "lm", "Language loss increase"), (axes[2], "task", "Task accuracy")):
        st = [mean_ci([r[k] for r in rows[n]]) for n in names]
        ax.barh(y, [m for m, _ in st], xerr=[c for _, c in st], color=colors, error_kw={"capsize": 2})
        ax.set_title(title)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[2].set_xlim(0.8, 0.95)
    finish(fig, "f10_gpm", data)


if __name__ == "__main__":
    data = load("f10_gpm")
    if data is None:
        data = compute()
    plot(data)
