"""Figure 3: removing noise removes forgetting. Each arm sees the same examples from the same starting point; points
are mean +/- 95% CI over 6 seeds (seed averages use disjoint seed groups where possible)."""

import matplotlib.pyplot as plt
import numpy as np

from lwn.analysis import mean
from lwn.fig import load, COLORS, SEEDS, context, deltas, finish, mean_ci
from lwn.runs import run

LR, STEPS = 1e-3, 1000
ARMS = {
    "baseline": {},
    "lower LR (÷3)": {"lr": LR / 3},
    "batch ×4": {"batch": 128, "steps": STEPS // 4, "snapshots": (STEPS // 4,)},
    "cosine decay": {"schedule": "cosine"},
    "cap 2%": {"cap": 0.02},
    "adaptive WD": {"awd": 0.05},
    "GPM 0.99": {"method": "gpm", "gpm_energy": 0.99},
}

def compute():
    ctx = context()
    base = ctx.base_metrics()
    rows = {}
    for name, kw in ARMS.items():
        kw = {"lr": LR, "steps": STEPS, "snapshots": (STEPS,), **kw}
        runs = [run(ctx, seed=s, **kw) for s in SEEDS]
        snaps = [r["snaps"][max(r["snaps"])] for r in runs]
        rows[name] = [deltas(ctx.eval_delta(s["raw"]), base) for s in snaps]
        if name == "baseline":
            rows["EMA export"] = [deltas(ctx.eval_delta(s["ema"]), base) for s in snaps]
            cd = [run(ctx, seed=s, lr=LR, steps=STEPS, snapshots=(STEPS,), cooldown_from=(int(STEPS / 1.2),)) for s in SEEDS]
            rows["cooldown (last 17%)"] = [deltas(ctx.eval_delta(next(iter(r["branches"].values()))["raw"]), base) for r in cd]
            emas = [s["ema"] for s in snaps]
            rows["avg of 2 seeds"] = [deltas(ctx.eval_delta(mean(emas[i:i + 2])), base) for i in (0, 2, 4)]
            rows["avg of 3 seeds"] = [deltas(ctx.eval_delta(mean(emas[i:i + 3])), base) for i in (0, 3)]
            rows["avg of 6 seeds"] = [deltas(ctx.eval_delta(mean(emas)), base)]
    return {"base": base, "rows": rows}

# What each arm acts on, for colouring: removing fast jitter, averaging noise away, bounding drift, steering the update.
GROUPS = {"baseline": ("none", "#868e96"), "EMA export": ("removes jitter", COLORS[0]),
          "cooldown (last 17%)": ("removes jitter", COLORS[0]), "lower LR (÷3)": ("less noise per step", COLORS[2]),
          "batch ×4": ("less noise per step", COLORS[2]), "cosine decay": ("less noise per step", COLORS[2]),
          "avg of 2 seeds": ("averages noise away", COLORS[1]), "avg of 3 seeds": ("averages noise away", COLORS[1]),
          "avg of 6 seeds": ("averages noise away", COLORS[1]), "cap 2%": ("bounds the move", COLORS[3]),
          "adaptive WD": ("bounds the move", COLORS[3]), "GPM 0.99": ("steers away from used directions", COLORS[4])}
ORDER = ["baseline", "EMA export", "cooldown (last 17%)", "lower LR (÷3)", "batch ×4", "cosine decay", "avg of 2 seeds",
         "avg of 3 seeds", "avg of 6 seeds", "cap 2%", "adaptive WD", "GPM 0.99"]


def plot(data):
    rows = data["rows"]
    names = [n for n in ORDER if n in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    y = np.arange(len(names))
    for ax, k, title in ((axes[0], "bio_nll", "Knowledge loss increase (NLL)"),
                         (axes[1], "lm", "Language loss increase"), (axes[2], "task", "Task accuracy (held out)")):
        stats = [mean_ci([r[k] for r in rows[n]]) for n in names]
        ax.barh(y, [m for m, _ in stats], xerr=[c for _, c in stats], color=[GROUPS[n][1] for n in names],
                error_kw={"lw": 1.2, "capsize": 2})
        ax.set_title(title)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[2].set_xlim(0.8, 0.95)
    axes[2].axvline(mean_ci([r["task"] for r in rows["baseline"]])[0], color="#495057", lw=1, ls=":")
    seen = {}
    for n in names:
        seen.setdefault(GROUPS[n][0], GROUPS[n][1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
    axes[2].legend(handles, list(seen), fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    finish(fig, "f3_mitigations", data)


if __name__ == "__main__":
    data = load("f3_mitigations")
    if data is None:
        data = compute()
    plot(data)
