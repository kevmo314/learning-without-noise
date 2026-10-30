"""Q6b: why do principal-subspace adapters forget more? Overlap with the inputs pretraining actually uses.

For each adapter type's final update, measure the share of its energy acting on the input subspace that held-out
pretraining activations occupy (90% and 99% of their energy), and its curvature on held-out text per unit size.
If damage tracks sensitivity, the methods that forget most per unit rho should put the most energy where the
pretraining inputs are.
"""

from lwn.analysis import curvature, dot
from lwn.runs import run, save, table
from lwn.train import Context

from experiments.q6_subspace import LR, METHODS

if __name__ == "__main__":
    ctx = Context()
    bases = {e: ctx.activation_basis(e) for e in (0.9, 0.99)}
    rows = []
    for name, kw in METHODS.items():
        kw = {"method": kw.get("method", name), "lr": kw.get("lr", LR), **{k: v for k, v in kw.items() if k not in ("method", "lr")}}
        d = run(ctx, **kw, seed=0, data_seed=0)["snaps"][1000]["ema"]
        row = {"method": name}
        for e, U in bases.items():
            inside = {n: U[n] @ (U[n].T @ x) for n, x in d.items()}
            row[f"share_in_{e}"] = round(dot(inside, inside) / dot(d, d), 4)
        row["curvature"] = round(curvature(ctx, d), 1)
        rows.append(row)
    table(rows, list(rows[0]))
    save("q6b_subspace_overlap", {"rows": rows})
