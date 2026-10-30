"""Q4: what mitigates noise, and what does each mitigation cost the task?

Every arm fine-tunes the same task for the same number of examples from the same starting point; the table is a
retention-vs-learning frontier. Arms fall into groups by the mechanism they act through:
  remove jitter       EMA export, cooldown branch, lower LR, larger batch, cosine / isqrt decay
  bound drift         radial cap, decay toward the base (fixed and adaptive), post-hoc shrink
  average out noise   average of seeds (2 and 4)
  steer the update    EWC (Fisher-weighted penalty), GPM (stay out of pretraining's activation subspace), replay
"""

from lwn.analysis import mean
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

LR, STEPS = 1e-3, 1000
ARMS = {
    "baseline (const, raw)": {},
    "lower LR (x1/3)": {"lr": LR / 3},
    "batch x4 (same examples)": {"batch": 128, "steps": STEPS // 4, "snapshots": (STEPS // 4,)},
    "batch x4, LR x2": {"batch": 128, "steps": STEPS // 4, "snapshots": (STEPS // 4,), "lr": 2 * LR},
    "cosine": {"schedule": "cosine"},
    "isqrt": {"schedule": "isqrt"},
    "cap 1%": {"cap": 0.01},
    "cap 2%": {"cap": 0.02},
    "wd to base": {"wd": 1.0},
    "adaptive wd to base": {"awd": 0.05},
    "EWC": {"ewc": 100.0},
    "GPM 0.9": {"method": "gpm", "gpm_energy": 0.9},
    "GPM 0.99": {"method": "gpm", "gpm_energy": 0.99},
    "replay": {"replay": 0.5},
    "no grad clip": {"clip": None},
}

if __name__ == "__main__":
    ctx = Context()
    rows = [{"arm": "base model", **summary(ctx.base_metrics())}]
    for name, kw in ARMS.items():
        kw = {"lr": LR, **kw}
        r = run(ctx, **kw)
        snap = r["snaps"][max(r["snaps"])]
        rows.append({"arm": name, "rho%": round(100 * rho(snap["raw"], ctx.w0_sq), 3), **summary(ctx.eval_delta(snap["raw"]))})
        if name == "baseline (const, raw)":
            rows.append({"arm": "EMA export", "rho%": round(100 * rho(snap["ema"], ctx.w0_sq), 3),
                         **summary(ctx.eval_delta(snap["ema"]))})
            for s in (0.5, 0.75):
                d = {n: s * x for n, x in snap["ema"].items()}
                rows.append({"arm": f"EMA, shrunk x{s}", "rho%": round(100 * rho(d, ctx.w0_sq), 3), **summary(ctx.eval_delta(d))})
            cd = run(ctx, lr=LR, cooldown_from=(int(STEPS / 1.2),))
            br = next(iter(cd["branches"].values()))["raw"]
            rows.append({"arm": "cooldown (last ~17%)", "rho%": round(100 * rho(br, ctx.w0_sq), 3), **summary(ctx.eval_delta(br))})
            seeds = [run(ctx, lr=LR, seed=s)["snaps"][STEPS]["ema"] for s in range(4)]
            for k in (2, 4):
                d = mean(seeds[:k])
                rows.append({"arm": f"avg of {k} seeds (EMA)", "rho%": round(100 * rho(d, ctx.w0_sq), 3), **summary(ctx.eval_delta(d))})
    table(rows, ["arm", "rho%", "task", "bio", "bio_nll", "lm", "kl", "acc_30_300", "acc_300_3000"])
    save("q4_mitigations", {"rows": rows})
