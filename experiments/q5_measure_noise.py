"""Q5: how can a single training run measure its own noise, and does that predict how far it can safely move?

For each run we compute signals available online, without extra seeds:
  jitter_share   ||raw - EMA||^2 / ||dW||^2   (free given an EMA)
  kl             KL(base || model) on held-out pretraining text
  progress       ||dW_t - dW_t'|| / ||dW_t|| between consecutive snapshots: how much the update is still turning over
and, as an oracle the run cannot see, its break-even rho: the largest post-hoc rescaling of the update that still
keeps knowledge (bio NLL) at the base model's level. Leave-one-run-out regression says which signal predicts it.
"""

import numpy as np

from lwn.analysis import add, dot, scale_sweep
from lwn.runs import run, save, table
from lwn.train import Context

SCALES = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
ARMS = [dict(lr=lr, batch=b, label_noise=q) for lr in (3e-4, 1e-3, 3e-3) for b in (32,) for q in (0.0, 0.3)]
ARMS += [dict(lr=1e-3, batch=128, steps=250, snapshots=(62, 125, 250)), dict(lr=1e-3, schedule="isqrt"),
         dict(lr=1e-3, schedule="cosine")]


def break_even(ctx, delta, base_nll):
    pts = scale_sweep(ctx, delta, SCALES)
    nll = [p["metrics"]["bio"]["nll"] for p in pts]
    rhos = [p["rho"] for p in pts]
    ok = [i for i, v in enumerate(nll) if v <= base_nll]
    if not ok or ok[-1] == len(pts) - 1:
        return None
    i = ok[-1]  # interpolate to the crossing between scale i and i+1
    return rhos[i] + (base_nll - nll[i]) / (nll[i + 1] - nll[i]) * (rhos[i + 1] - rhos[i])


if __name__ == "__main__":
    ctx = Context()
    base_nll = ctx.base_metrics()["bio"]["nll"]
    rows = []
    for kw in ARMS:
        r = run(ctx, **kw)
        snaps = sorted(r["snaps"])
        for s in snaps[1:]:
            raw, ema = r["snaps"][s]["raw"], r["snaps"][s]["ema"]
            prev = r["snaps"][snaps[snaps.index(s) - 1]]["raw"]
            step = add((1, raw), (-1, prev))
            jit = add((1, raw), (-1, ema))
            be = break_even(ctx, raw, base_nll)
            rows.append({"run": ",".join(f"{k}={v}" for k, v in kw.items() if k != "snapshots"), "step": s,
                         "rho%": round(100 * (dot(raw, raw) / ctx.w0_sq) ** 0.5, 3),
                         "jitter_share": round(dot(jit, jit) / dot(raw, raw), 4),
                         "kl": round(ctx.eval_delta(raw)["kl"], 5),
                         "progress": round(dot(step, step) ** 0.5 / (dot(raw, raw) ** 0.5 + 1e-12), 4),
                         "break_even%": None if be is None else round(100 * be, 3)})
    table(rows, ["run", "step", "rho%", "jitter_share", "kl", "progress", "break_even%"])
    R = [r for r in rows if r["break_even%"]]
    y = np.log([r["break_even%"] for r in R])
    runs = [r["run"] for r in R]
    feats = {"constant": [], "log step": [np.log([r["step"] for r in R])],
             "log jitter share": [np.log([r["jitter_share"] for r in R])],
             "log kl": [np.log([r["kl"] for r in R])],
             "log jitter share + log step": [np.log([r["jitter_share"] for r in R]), np.log([r["step"] for r in R])]}
    errs = {}
    for name, fs in feats.items():
        X = np.stack([np.ones(len(R)), *fs], 1)
        e = []
        for held in set(runs):
            tr = [i for i, x in enumerate(runs) if x != held]
            te = [i for i, x in enumerate(runs) if x == held]
            c, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            e += list(X[te] @ c - y[te])
        errs[name] = float(np.sqrt(np.mean(np.square(e))))
        print(f"leave-one-run-out RMS log error, {name}: {errs[name]:.3f}")
    save("q5_measure_noise", {"rows": rows, "loro_error": errs})
