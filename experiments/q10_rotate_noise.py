"""Q10: is it *where* noise points, not how big it is? Rotate measured noise between subspaces at fixed norm.

Take the measured fast jitter and slow drift of real fine-tunes and split each matrix's noise into the part acting on
the input subspace that pretraining activations occupy (90% of their energy) and the part acting on its complement.
Each part is rescaled to the same distance from the base. If forgetting is noise weighted by pretraining
sensitivity, noise confined to the pretraining-active subspace should do far more damage than the same amount
outside it, for jitter and drift alike. Random Gaussian noise is the control.
"""

import torch

from lwn.analysis import add, dot, mean
from lwn.runs import run, save, summary, table
from lwn.train import Context

RHOS = (0.01, 0.02, 0.04)


def scaled(ctx, d, r):
    s = r * (ctx.w0_sq / dot(d, d)) ** 0.5
    return {n: s * x for n, x in d.items()}


if __name__ == "__main__":
    ctx = Context()
    base = summary(ctx.base_metrics())
    U = ctx.activation_basis(0.9)
    runs = [run(ctx, lr=1e-3, seed=s) for s in range(4)]
    emas = [r["snaps"][1000]["ema"] for r in runs]
    raw = runs[0]["snaps"][1000]["raw"]
    g = torch.Generator(device="cuda").manual_seed(0)
    noise = {"jitter": add((1, raw), (-1, emas[0])), "drift": add((1, emas[0]), (-1, mean(emas))),
             "random": {n: torch.randn(d.shape, generator=g, device="cuda") for n, d in raw.items()}}
    rows = []
    for kind, d in noise.items():
        inside = {n: U[n] @ (U[n].T @ x) for n, x in d.items()}  # dW is [in, out]: project the input side
        outside = add((1, d), (-1, inside))
        share = dot(inside, inside) / dot(d, d)
        for part, v in {"as measured": d, "in pretraining subspace": inside, "outside it": outside}.items():
            for r in RHOS:
                m = summary(ctx.eval_delta(scaled(ctx, v, r)))
                rows.append({"noise": kind, "part": part, "share_in": round(share, 3), "rho%": 100 * r,
                             **{k: round(m[k] - base[k], 4) for k in ("bio", "bio_nll", "lm", "kl", "task")}})
    print("changes from the base model")
    table(rows, ["noise", "part", "share_in", "rho%", "bio", "bio_nll", "lm", "kl", "task"])
    save("q10_rotate_noise", {"base": base, "rows": rows})
