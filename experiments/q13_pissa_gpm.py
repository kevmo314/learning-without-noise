"""Does a nonzero start help inside GPM's protected subspace? PiSSA starts A and B nonzero (with a frozen cancelling
copy), so both factors get gradients from the first step; LoRA starts B at zero. pissa-gpm starts from the top
singular directions of W0 restricted to the inputs GPM allows, and keeps A there. Compared with LoRA, GPM and PiSSA at
the shared LR and at LR/10 (PiSSA's start makes each step larger), on the four-way task. 6 seeds, EMA at step 1000.
"""

from lwn.fig import SEEDS, context, deltas, mean_ci, tag
from lwn.runs import run, save, table

GPM = {"gpm_energy": 0.9, "gpm_source": "mix"}
ARMS = {"LoRA": {"method": "lora"}, "GPM": {"method": "gpm", **GPM},
        "PiSSA": {"method": "pissa"}, "PiSSA LR/10": {"method": "pissa", "lr": 1e-4},
        "PiSSA-GPM": {"method": "pissa-gpm", **GPM}, "PiSSA-GPM LR/10": {"method": "pissa-gpm", "lr": 1e-4, **GPM},
        "PiSSA-GPM LR/3": {"method": "pissa-gpm", "lr": 3e-4, **GPM}}

if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    rows, out = [], {}
    for name, kw in ARMS.items():
        kw = {"lr": 1e-3, **kw}
        ds = [deltas(ctx.eval_delta(run(ctx, steps=1000, snapshots=(1000,), seed=s, **kw)["snaps"][1000]["ema"]), base)
              for s in SEEDS]
        out[name] = ds
        row = {"arm": name}
        for k in ("task", "bio_nll", "lm", "kl"):
            m, c = mean_ci([d[k] for d in ds])
            row[k] = f"{m:.4f}±{c:.4f}"
        rows.append(row)
        table(rows, ["arm", "task", "bio_nll", "lm", "kl"])
    save(tag("q13_pissa_gpm"), {"base": base, "rows": out})
