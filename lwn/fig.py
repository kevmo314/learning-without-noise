"""Shared setup for the presentation figures: a large evaluation set, many seeds, and mean +/- 95% CI plots.

Every figure uses the harder four-way story-feature task and the large evaluation set (12k knowledge items, 512
held-out text windows), reports graded metrics (knowledge NLL, language loss, KL) rather than accuracy where
possible, and averages over SEEDS independent fine-tunes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import Context

ROOT = Path(__file__).resolve().parent.parent
SEEDS = tuple(range(6))
BIO_ATTRS = ("birth_city", "university", "major", "company", "company_city")
# Colors: a fixed categorical order (never cycled), validated for colorblind separation.
COLORS = ["#2a6fdb", "#e8590c", "#2b9348", "#c2255c", "#7048e8", "#868e96"]


def context():
    return Context(task="features", n_bio_people=3000, bio_attrs=BIO_ATTRS, n_val=512)


def deltas(m, base):
    """Changes from the base model in the graded metrics, plus task accuracy."""
    return {"task": m["task"]["acc"], "bio_nll": m["bio"]["nll"] - base["bio"]["nll"],
            "bio": m["bio"]["acc"] - base["bio"]["acc"], "lm": m["lm"] - base["lm"], "kl": m.get("kl", 0.0)}


def mean_ci(xs):
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return float(xs.mean()), 0.0
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / np.sqrt(len(xs)))


def band(ax, x, ys, label, color, style="-"):
    """Plot the mean over seeds (ys: [seed][point]) with a shaded 95% CI."""
    ys = np.asarray(ys, dtype=float)
    m = ys.mean(0)
    ci = 1.96 * ys.std(0, ddof=1) / np.sqrt(len(ys)) if len(ys) > 1 else np.zeros_like(m)
    ax.plot(x, m, style, color=color, label=label, lw=2)
    ax.fill_between(x, m - ci, m + ci, color=color, alpha=0.18, lw=0)


def finish(fig, name, data):
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    for ax in fig.axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=160)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
    plt.close(fig)
