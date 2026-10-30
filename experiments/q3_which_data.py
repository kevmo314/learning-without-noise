"""Q3: which knowledge does noise destroy first?

No learning at all here: we add pure noise to the base weights and watch what breaks. Four kinds of noise, each
scaled to the same distances rho from the base:
  isotropic   Gaussian noise on every adapted matrix (full rank)
  lowrank     a random rank-16 product B A, the shape LoRA updates have
  jitter      the measured fast-jitter directions (raw - EMA) of real fine-tunes, rescaled
  drift       the measured slow-drift directions (EMA minus the cross-seed mean), rescaled
Knowledge is split by how often each person appeared in pretraining, so we can see whether rare, barely-learned
facts break before well-rehearsed ones, and how general language (held-out loss) compares.
"""

import torch

from lwn.analysis import add, dot, mean
from lwn.runs import run, save, summary, table
from lwn.train import Context

RHOS = (0.005, 0.01, 0.02, 0.04, 0.08)


def scaled(ctx, d, r):
    s = r * (ctx.w0_sq / dot(d, d)) ** 0.5
    return {n: s * x for n, x in d.items()}


if __name__ == "__main__":
    ctx = Context()
    base = summary(ctx.base_metrics())
    g = torch.Generator(device="cuda").manual_seed(0)
    W = {n: ctx.base_state[n + ".weight"] for n in ctx.names}
    iso = {n: torch.randn(w.shape, generator=g, device="cuda") * w.std() for n, w in W.items()}
    low = {n: (torch.randn(w.shape[0], 16, generator=g, device="cuda") @ torch.randn(16, w.shape[1], generator=g, device="cuda"))
           for n, w in W.items()}
    runs = [run(ctx, lr=1e-3, seed=s) for s in range(4)]
    emas = [r["snaps"][1000]["ema"] for r in runs]
    raws = [r["snaps"][1000]["raw"] for r in runs]
    jitter = add((1, raws[0]), (-1, emas[0]))
    drift = add((1, emas[0]), (-1, mean(emas)))
    rows = []
    for kind, d in {"isotropic": iso, "lowrank": low, "jitter": jitter, "drift": drift}.items():
        for r in RHOS:
            m = summary(ctx.eval_delta(scaled(ctx, d, r)))
            rows.append({"noise": kind, "rho%": 100 * r, **{k: round(m[k] - base[k], 4) for k in m}})
    cols = ["noise", "rho%", "bio", "acc_0_30", "acc_30_300", "acc_300_3000", "acc_3000_10000", "lm", "kl", "task"]
    print("changes from the base model")
    table(rows, cols)
    save("q3_which_data", {"base": base, "rows": rows})
