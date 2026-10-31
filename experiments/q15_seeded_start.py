"""Does a nonzero (PiSSA-style) start speed up early learning inside GPM's subspace, and does renormalizing after
constraining help? LoRA and GPM start with B = 0, so A gets no gradient at first. PiSSA-GPM starts from the top
singular directions of W0 restricted to GPM's allowed inputs, at the restricted matrix's singular values;
PiSSA-GPM-norm uses the same directions at W0's own (larger) singular values. Raw weights are evaluated at ~20
log-spaced steps to measure learning speed, then forgetting at the end. 6 seeds, two learning rates."""

import numpy as np

from lwn.fig import SEEDS, context, curve, deltas, log_steps, mean_ci, tag
from lwn.runs import run, save, table

GPM = {"gpm_energy": 0.9, "gpm_source": "mix"}
METHODS = {"LoRA": {"method": "lora"}, "GPM": {"method": "gpm", **GPM},
           "PiSSA-GPM": {"method": "pissa-gpm", **GPM}, "PiSSA-GPM-norm": {"method": "pissa-gpm-norm", **GPM}}
LRS = (1e-3, 1e-4)
EVALS = log_steps(1000, first=4, n=20)

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    rows, out = [], {}
    for lr in LRS:
        for name, kw in METHODS.items():
            runs = [run(ctx, lr=lr, steps=1000, snapshots=(1000,), evals=EVALS, seed=s, **kw) for s in SEEDS]
            curves = [[deltas(m, base) for _, m in curve(r)] for r in runs]
            end = [deltas(ctx.eval_delta(r["snaps"][1000]["ema"]), base) for r in runs]
            out[f"{name}, LR {lr:g}"] = {"steps": [s for s, _ in curve(runs[0])], "curves": curves, "end_ema": end,
                                         "loss": [[c["loss"] for c in r["curve"]] for r in runs]}
            acc = np.array([[c["task"] for c in cs] for cs in curves])
            steps = np.array([s for s, _ in curve(runs[0])])
            row = {"arm": f"{name}, LR {lr:g}"}
            for t in (16, 64, 256):
                i = int(np.argmin(abs(steps - t)))
                row[f"task@{steps[i]}"] = "%.3f±%.3f" % mean_ci(acc[:, i])
            for k in ("task", "bio_nll", "lm"):
                row[f"end {k}"] = "%.4f±%.4f" % mean_ci([e[k] for e in end])
            rows.append(row)
            table(rows, list(rows[0]))
    save(tag("q15_seeded_start"), {"base": base, "evals": EVALS, "rows": out})
