"""Q9: is the forgetting caused by *stochastic* gradients at all? (the decisive test both reviewers asked for)

A skeptic's alternative: the seed-specific "drift" is not optimizer noise but the same deterministic task gradient
projected into each seed's random LoRA subspace, and forgetting would happen without any minibatch noise.
On a 512-example subset we compare exact full-batch AdamW (no sampling noise) with minibatch AdamW, crossing
adapter init with data order:
  * full-batch, two inits             -> disagreement here is purely init / subspace, not noise
  * minibatch, same init, two orders  -> disagreement here is purely sampling noise
  * minibatch, two inits, same order  -> init alone, under noise
Matching epochs (full-batch 300 steps = 300 epochs; minibatch 32 x 4800 steps = 300 epochs) and matching steps
(minibatch 300 steps) are both reported, with task accuracy so progress can be compared.
If full-batch training forgets about as much as minibatch at matched task progress, "forgetting is mostly noise" is
false for this setting.
"""

from lwn.analysis import dot
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

N = 512


def cos(a, b):
    return dot(a, b) / (dot(a, a) * dot(b, b)) ** 0.5


if __name__ == "__main__":
    ctx = Context()
    rows, pairs = [{"arm": "base", **summary(ctx.base_metrics())}], []
    arms = {}
    for lr in (3e-4, 1e-3):
        fb = {i: run(ctx, lr=lr, subset=N, batch=N, steps=300, snapshots=(100, 300), seed=i, ema_examples=None) for i in (0, 1)}
        mb = {(i, o): run(ctx, lr=lr, subset=N, batch=32, steps=4800, snapshots=(300, 1600, 4800), seed=i, data_seed=o,
                          ema_examples=None) for i, o in ((0, 0), (0, 1), (1, 0))}
        for i, r in fb.items():
            for s in (100, 300):
                d = r["snaps"][s]["raw"]
                rows.append({"arm": f"full-batch lr={lr:g} init{i}", "step": s, "epochs": s,
                             "rho%": round(100 * rho(d, ctx.w0_sq), 3), **summary(ctx.eval_delta(d))})
        for (i, o), r in mb.items():
            for s in (300, 1600, 4800):
                d = r["snaps"][s]["raw"]
                rows.append({"arm": f"minibatch lr={lr:g} init{i} order{o}", "step": s, "epochs": round(s * 32 / N, 1),
                             "rho%": round(100 * rho(d, ctx.w0_sq), 3), **summary(ctx.eval_delta(d))})
        pairs.append({"lr": lr, "pair": "full-batch, different init (300 ep)",
                      "cos": round(cos(fb[0]["snaps"][300]["raw"], fb[1]["snaps"][300]["raw"]), 4)})
        for s in (300, 4800):
            pairs.append({"lr": lr, "pair": f"minibatch, same init, different order ({s} steps)",
                          "cos": round(cos(mb[(0, 0)]["snaps"][s]["raw"], mb[(0, 1)]["snaps"][s]["raw"]), 4)})
            pairs.append({"lr": lr, "pair": f"minibatch, different init, same order ({s} steps)",
                          "cos": round(cos(mb[(0, 0)]["snaps"][s]["raw"], mb[(1, 0)]["snaps"][s]["raw"]), 4)})
    table(rows, ["arm", "step", "epochs", "rho%", "task", "bio", "bio_nll", "lm", "kl"])
    print()
    table(pairs, ["lr", "pair", "cos"])
    save("q9_deterministic", {"rows": rows, "pairs": pairs})
