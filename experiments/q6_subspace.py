"""Q6: does where the adapter lives decide how much its noise hurts?

Adapter parametrisations put the update, and so its noise, in different subspaces:
  lora     a random input subspace (Kaiming A, zero B)
  pissa    the base matrix's top singular directions        olora   an orthonormal QR basis of the base matrix
  milora   the base matrix's bottom singular directions     gpm     orthogonal to pretraining's activation subspace
  full     every direction (full fine-tuning)
For each we train two runs from the same init with different batch orders, which isolates order-driven drift
(cross-order cosine), and compare retention at the same distance moved by rescaling each update post hoc.
"""

from lwn.analysis import dot, scale_sweep
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

METHODS = {"lora": {}, "pissa": {}, "milora": {}, "olora": {}, "gpm": {"gpm_energy": 0.9},
           "gpm99": {"method": "gpm", "gpm_energy": 0.99}, "full": {"lr": 1e-4}}
LR = 1e-3

if __name__ == "__main__":
    ctx = Context()
    rows, sweeps = [], []
    for name, kw in METHODS.items():
        kw = {"method": kw.get("method", name), "lr": kw.get("lr", LR), **{k: v for k, v in kw.items() if k not in ("method", "lr")}}
        a = run(ctx, **kw, seed=0, data_seed=0)
        b = run(ctx, **kw, seed=0, data_seed=1)
        da, db = a["snaps"][1000]["ema"], b["snaps"][1000]["ema"]
        cos = dot(da, db) / (dot(da, da) * dot(db, db)) ** 0.5
        rows.append({"method": name, "rho%": round(100 * rho(da, ctx.w0_sq), 3), "cross_order_cos": round(cos, 4),
                     **summary(ctx.eval_delta(da))})
        for p in scale_sweep(ctx, da, (0.25, 0.5, 1.0, 1.5, 2.0)):
            sweeps.append({"method": name, "scale": p["scale"], "rho%": round(100 * p["rho"], 3), **summary(p["metrics"])})
    table(rows, ["method", "rho%", "cross_order_cos", "task", "bio", "bio_nll", "lm", "kl"])
    print()
    table(sweeps, ["method", "scale", "rho%", "task", "bio", "bio_nll", "lm", "kl"])
    save("q6_subspace", {"rows": rows, "sweeps": sweeps})
