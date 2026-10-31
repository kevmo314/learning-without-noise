"""Do the two GPM improvements stack? Input subspace from the pretraining mix plus held-out biographies (q21), and
output-side protection (q20), each alone and together, against plain GPM and LoRA. 6 seeds, two LRs, EMA at 1000."""

from lwn.fig import SEEDS, context, deltas, mean_ci, tag
from lwn.runs import run, save, table

ARMS = {"LoRA": {}, "GPM mix": {"method": "gpm", "gpm_energy": 0.95, "gpm_source": "mix"},
        "GPM mix+bio": {"method": "gpm", "gpm_energy": 0.95, "gpm_source": "mix+bio"},
        "GPM mix + out": {"method": "gpm", "gpm_energy": 0.95, "gpm_source": "mix", "gpm_out": 0.9},
        "GPM mix+bio + out": {"method": "gpm", "gpm_energy": 0.95, "gpm_source": "mix+bio", "gpm_out": 0.9}}
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
                                                    for k in ("task", "bio_nll", "lm", "kl")}})
            table(rows, list(rows[0]))
    save(tag("q22_best_gpm"), {"base": base, "rows": out})
