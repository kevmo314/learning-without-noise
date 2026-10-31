"""Two-sided GPM: does protecting output directions too cut knowledge loss?

GPM keeps the adapter's input directions (A) out of the subspace pretraining activations use. q16/q18 found that this
input-side picture explains language damage almost perfectly but knowledge damage only partly, and that the K-FAC
Fisher, which also weights the output side, predicts both. Here B is also kept out of the top directions of the
loss-gradient covariance at each layer's output (K-FAC's G factor on the pretraining mix): input side only, output
side only, and both, against plain LoRA. 6 seeds, two LRs, EMA at step 1000."""

from lwn.fig import SEEDS, context, deltas, mean_ci, tag
from lwn.runs import run, save, table

IN = {"method": "gpm", "gpm_energy": 0.9, "gpm_source": "mix"}
ARMS = {"LoRA": {}, "GPM in 0.9": IN, "GPM out 0.9": {"gpm_out": 0.9}, "GPM out 0.99": {"gpm_out": 0.99},
        "GPM in+out 0.9": {**IN, "gpm_out": 0.9}, "GPM in 0.9 + out 0.5": {**IN, "gpm_out": 0.5}}
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
    if any(k for k in ARMS.values() if k.get("gpm_out")):
        for e in sorted({k["gpm_out"] for k in ARMS.values() if k.get("gpm_out")}):
            V = ctx.output_basis(e)
            print(f"output energy {e}: protected output dims {sum(v.shape[1] for v in V.values())} of "
                  f"{sum(v.shape[0] for v in V.values())}")
    save(tag("q20_two_sided_gpm"), {"base": base, "rows": out})
