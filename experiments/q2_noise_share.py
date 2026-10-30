"""Q2: how much of forgetting is noise, and how much is the update the data actually asks for?

K seeds of the same fine-tune differ only in adapter init and batch order. Their shared part is the systematic
update; what differs is noise, split into fast jitter (raw - EMA) and slow drift (EMA minus the shared part).
Averaging seeds and taking EMAs gives versions of the update with the same signal and different amounts of each
noise, so regressing retention on the noise energies (one intercept per group) prices each kind, and the
intercept is what the signal alone would do. The curvature of each direction on held-out pretraining text
(a Fisher Rayleigh quotient) checks the proposed mechanism: jitter should sit in sharper directions.
"""

import numpy as np

from lwn.analysis import add, curvature, energies, mean
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

LRS = (3e-4, 1e-3)
STEPS = (250, 1000)
K = 4

if __name__ == "__main__":
    ctx = Context()
    base = summary(ctx.base_metrics())
    rows, groups, X, Y = [], [], [], {k: [] for k in ("bio_nll", "bio", "lm", "task", "kl")}
    for lr in LRS:
        runs = [run(ctx, lr=lr, seed=s) for s in range(K)]
        for step in STEPS:
            g = f"{lr:g}@{step}"
            raws = [r["snaps"][step]["raw"] for r in runs]
            emas = [r["snaps"][step]["ema"] for r in runs]
            e = energies(raws, emas, ctx.w0_sq)
            jit = [add((1, r), (-1, m)) for r, m in zip(raws, emas)]
            sig = mean(emas)
            drift = [add((1, m), (-1, sig)) for m in emas]
            curv = {
                "signal": curvature(ctx, sig),
                "drift": float(np.mean([curvature(ctx, d) for d in drift])),
                "jitter": float(np.mean([curvature(ctx, j) for j in jit])),
            }
            groups.append({"group": g, **{k: float(v) for k, v in e.items()}, **{f"curv_{k}": v for k, v in curv.items()}})
            J, D = e["jitter"], e["drift"]
            variants = {f"raw{i}": (raws[i], J, D) for i in range(K)}
            variants |= {f"ema{i}": (emas[i], 0.0, D) for i in range(K)}
            variants |= {"avg_raw": (mean(raws), J / K, D / K), "avg_ema": (sig, 0.0, D / K),
                         "avg2_ema": (mean(emas[:2]), 0.0, D / 2)}
            for name, (d, j, dd) in variants.items():
                m = summary(ctx.eval_delta(d))
                rows.append({"group": g, "variant": name, "rho%": round(100 * rho(d, ctx.w0_sq), 3),
                             "jitter": j, "drift": dd, **m})
                X.append((g, j, dd))
                for k in Y:
                    Y[k].append(m[k])
    names = [g["group"] for g in groups]
    A = np.array([[1.0 if g == n else 0.0 for n in names] + [-j, -d] for g, j, d in X])
    fits = {}
    for k, y in Y.items():
        sign = -1 if k in ("bio_nll", "lm", "kl") else 1  # fit damage as a positive cost
        c, *_ = np.linalg.lstsq(A, sign * np.array(y), rcond=None)
        fits[k] = {"jitter_cost": c[-2], "drift_cost": c[-1], "ratio": c[-2] / c[-1] if c[-1] else None,
                   "signal_only": {n: sign * c[i] for i, n in enumerate(names)}, "base": base[k]}
    table(groups, ["group", "signal", "drift", "jitter", "cross_seed_cos", "curv_signal", "curv_drift", "curv_jitter"])
    print()
    table(rows, ["group", "variant", "rho%", "task", "bio", "bio_nll", "lm", "kl"])
    print()
    for k, f in fits.items():
        print(k, f"jitter cost {f['jitter_cost']:.3g} drift cost {f['drift_cost']:.3g} ratio {f['ratio']:.3g}",
              "signal-only:", {n: round(v, 4) for n, v in f["signal_only"].items()}, "base", f["base"])
    save("q2_noise_share", {"base": base, "groups": groups, "rows": rows, "fits": fits})
