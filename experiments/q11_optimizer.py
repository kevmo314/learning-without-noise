"""Q11: does the jitter/drift picture depend on Adam?

Reviewers point out the dynamics here are clipped AdamW, whose per-coordinate preconditioning reshapes gradient
noise, not plain SGD (for which the Hessian-aligned noise theory was derived). We repeat the seed decomposition
with SGD + momentum at learning rates giving similar task accuracy, and compare noise energies, their costs and the
curvature of each direction.
"""

import numpy as np

from lwn.analysis import add, curvature, energies, mean
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

ARMS = {"adamw 1e-3": {"lr": 1e-3}, "sgd 0.3": {"lr": 0.3, "optimizer": "sgd"}, "sgd 1.0": {"lr": 1.0, "optimizer": "sgd"}}
K = 4

if __name__ == "__main__":
    ctx = Context()
    rows = []
    for name, kw in ARMS.items():
        runs = [run(ctx, seed=s, **kw) for s in range(K)]
        raws = [r["snaps"][1000]["raw"] for r in runs]
        emas = [r["snaps"][1000]["ema"] for r in runs]
        e = energies(raws, emas, ctx.w0_sq)
        sig = mean(emas)
        m_raw = [summary(ctx.eval_delta(d)) for d in raws]
        rows.append({"arm": name, "rho%": round(100 * rho(raws[0], ctx.w0_sq), 3),
                     "task": round(float(np.mean([m["task"] for m in m_raw])), 4),
                     "bio_raw": round(float(np.mean([m["bio"] for m in m_raw])), 4),
                     "bio_avg_ema": summary(ctx.eval_delta(sig))["bio"],
                     **{k: float(f"{v:.3g}") for k, v in e.items()},
                     "curv_jitter": round(curvature(ctx, add((1, raws[0]), (-1, emas[0]))), 1),
                     "curv_drift": round(curvature(ctx, add((1, emas[0]), (-1, sig))), 1),
                     "curv_signal": round(curvature(ctx, sig), 1)})
    table(rows, list(rows[0]))
    save("q11_optimizer", {"rows": rows})
