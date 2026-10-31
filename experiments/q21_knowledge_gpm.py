"""Protect the subspace the knowledge actually uses. GPM's subspace measured on stories protects language but not the
biographies (figure 10), and on 2B a subspace measured on general prose left 1-4 MMLU-Pro points of loss. Here the
subspace comes from biographies of people the probe never asks about (knowledge-dense text with no leakage), alone
or added to the pretraining mix, against the mix alone. 6 seeds, two LRs, EMA at step 1000."""

from lwn.fig import SEEDS, context, deltas, mean_ci, tag
from lwn.runs import run, save, table

ARMS = {"LoRA": {}}
for e in (0.9, 0.95):
    for src in ("mix", "bio", "mix+bio"):
        ARMS[f"GPM {e} {src}"] = {"method": "gpm", "gpm_energy": e, "gpm_source": src}
LRS = (1e-3, 3e-4)

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    rows, out = [], []
    for lr in LRS:
        for name, kw in ARMS.items():
            ds = [deltas(ctx.eval_delta(run(ctx, lr=lr, steps=1000, snapshots=(1000,), seed=s, **kw)["snaps"][1000]
                                        ["ema"]), base) for s in SEEDS]
            out += [{"arm": name, "lr": lr, "seed": s, **d} for s, d in zip(SEEDS, ds)]
            rows.append({"arm": name, "lr": lr, **{k: "%.4f±%.4f" % mean_ci([d[k] for d in ds])
                                                    for k in ("task", "bio_nll", "bio", "lm")}})
            table(rows, list(rows[0]))
    save(tag("q21_knowledge_gpm"), {"base": base, "rows": out})
