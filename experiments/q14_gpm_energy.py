"""How much of the input space does the adapter need? GPM at protected-energy thresholds from 50% to 99.9%: the
fraction of directions left to the adapter, and the forgetting and task accuracy that result. 6 seeds, EMA at 1000."""

from lwn.fig import SEEDS, context, deltas, mean_ci, tag
from lwn.runs import run, save, table

ENERGIES = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    rows, out = [], {}
    for e in (None, *ENERGIES):
        kw = {} if e is None else {"method": "gpm", "gpm_energy": e, "gpm_source": "mix"}
        ds = [deltas(ctx.eval_delta(run(ctx, lr=1e-3, steps=1000, snapshots=(1000,), seed=s, **kw)["snaps"][1000]["ema"]),
                     base) for s in SEEDS]
        free = 1.0 if e is None else sum(U.shape[0] - U.shape[1] for U in ctx.activation_basis(e, "mix").values()) / \
            sum(U.shape[0] for U in ctx.activation_basis(e, "mix").values())
        out[str(e)] = {"free": free, "runs": ds}
        row = {"energy": "LoRA" if e is None else e, "free dirs": f"{free:.0%}"}
        for k in ("task", "bio_nll", "lm"):
            m, c = mean_ci([d[k] for d in ds])
            row[k] = f"{m:.4f}±{c:.4f}"
        rows.append(row)
        table(rows, ["energy", "free dirs", "task", "bio_nll", "lm"])
    save(tag("q14_gpm_energy"), {"base": base, "rows": out})
