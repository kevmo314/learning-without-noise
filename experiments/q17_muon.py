"""Muon vs AdamW vs SGD, with and without GPM, and with a PiSSA start: forgetting at matched task accuracy.

Muon orthogonalises each update (U V^T of the momentum), which gives every singular direction of the gradient the same
size. Relative to SGD and AdamW, that turns down the dominant input directions, the ones ordinary text uses most, so
the prediction is SGD > AdamW > Muon > GPM in forgetting at equal task accuracy. If damage is the activation-weighted
change tr(dW^T C dW), rho_C should put every optimizer on one curve where plain rho does not. Each optimizer is swept
over learning rates (Muon's on AdamW's scale), 6 seeds, EMA at step 1000.
"""

import numpy as np

from experiments.q16_weighted_rho import covariances, weighted
from lwn.fig import SEEDS, context, deltas, mean_ci, tag
from lwn.runs import run, save, table

GPM = {"method": "gpm", "gpm_energy": 0.9, "gpm_source": "mix"}
ARMS = {
    "AdamW": ({}, (1e-4, 3e-4, 1e-3, 3e-3)),
    "Muon": ({"optimizer": "muon"}, (1e-3, 3e-3, 1e-2, 3e-2)),  # learns slower at equal LR: its first steps are smaller
    "SGD": ({"optimizer": "sgd"}, (0.03, 0.1, 0.3, 1.0)),
    "GPM + AdamW": (GPM, (1e-4, 3e-4, 1e-3, 3e-3)),
    "GPM + Muon": ({**GPM, "optimizer": "muon"}, (1e-3, 3e-3, 1e-2, 3e-2)),
    "PiSSA + AdamW": ({"method": "pissa"}, (3e-5, 1e-4, 3e-4, 1e-3)),
    "PiSSA + Muon": ({"method": "pissa", "optimizer": "muon"}, (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)),
    "PiSSA-GPM + AdamW": ({**GPM, "method": "pissa-gpm"}, (3e-5, 1e-4, 3e-4, 1e-3)),
}

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    C = covariances(ctx)
    W0 = {n: ctx.base_state[n + ".weight"].cuda() for n in ctx.names}
    w0 = sum(float(w.double().square().sum()) for w in W0.values())
    rows, out = [], []
    for name, (kw, lrs) in ARMS.items():
        for lr in lrs:
            for s in SEEDS:
                dW = run(ctx, lr=lr, steps=1000, snapshots=(1000,), seed=s, **kw)["snaps"][1000]["ema"]
                out.append({"arm": name, "lr": lr, "seed": s, **deltas(ctx.eval_delta(dW), base),
                            "rho": (sum(float(d.double().square().sum()) for d in dW.values()) / w0) ** 0.5,
                            "rho_C": weighted(dW, W0, C["mix"]), "rho_task": weighted(dW, W0, C["task"])})
            g = [r for r in out if r["arm"] == name and r["lr"] == lr]
            rows.append({"arm": name, "lr": lr, **{k: "%.4g±%.2g" % mean_ci([r[k] for r in g])
                                                   for k in ("task", "bio_nll", "lm", "rho", "rho_C")}})
            table(rows, list(rows[0]))
    save(tag("q17_muon"), {"base": base, "rows": out})
