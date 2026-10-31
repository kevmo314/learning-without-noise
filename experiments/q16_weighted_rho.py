"""Is the best GPM threshold computable, and is there a distance that predicts forgetting across methods?

rho measures ||dW||_F as if every input direction mattered equally. What changes the model on ordinary text is
||dW x||, whose mean square is tr(dW^T C dW) with C the input covariance on that text; plain rho is the case C = I. For
each cached run (GPM at every threshold, LoRA at every rank) this compares, against the forgetting:
  rho        ||dW||_F / ||W0||_F
  rho_C      sqrt(sum tr(dW^T C dW) / sum tr(W0^T C W0)), C = pretraining-mix activations (activation-weighted rho)
  rho_task   the same with the task's input activations (how much the update moves the task)
and, from activations alone, the threshold minimising leakage / task room:
  (pretraining energy left in the free subspace) / (task energy left in the free subspace)."""

import numpy as np
import torch
from scipy.stats import spearmanr

from lwn.fig import SEEDS, context, deltas, tag
from lwn.model import targets
from lwn.runs import run, save

GPM = {"gpm_source": "mix"}
ENERGIES = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@torch.no_grad()
def covariances(ctx, n=32):
    """Uncentered input covariance of every adapted matrix on the pretraining mix and on task inputs (real tokens)."""
    model, names = ctx.eval_model, ctx.names
    mods = {k: m for k, m in model.named_modules() if k in names}
    out = {}
    for src in ("mix", "task"):
        acc, cnt, hooks, mask = {}, [0], [], {}

        def hook(n_):
            def f(mod, i, o):
                x = i[0].reshape(-1, i[0].shape[-1]).double()
                if "m" in mask:
                    x = x[mask["m"]]
                acc[n_] = acc.get(n_, 0) + x.T @ x
            return f

        hooks = [m.register_forward_hook(hook(k)) for k, m in mods.items()]
        if src == "mix":
            x = ctx.data.lm_batch(n, np.random.default_rng(123))[:, :-1].cuda()
            model(input_ids=x)
            tokens = x.numel()
        else:
            (ids, att), _ = ctx.data.task_batch(n * 2, np.random.default_rng(123))
            mask["m"] = att.reshape(-1).bool().cuda()
            pos = (att.cumsum(1) - 1).clamp_min(0)
            model.transformer(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda())
            tokens = int(att.sum())
        for h in hooks:
            h.remove()
        out[src] = {k: v / tokens for k, v in acc.items()}
    return out


def weighted(dW, W0, C):
    """sqrt(sum tr(dW^T C dW) / sum tr(W0^T C W0)); Conv1D layout [in, out], C is [in, in]."""
    num = sum(float(torch.einsum("io,ij,jo->", dW[n].double(), C[n], dW[n].double())) for n in dW)
    den = sum(float(torch.einsum("io,ij,jo->", W0[n].double(), C[n], W0[n].double())) for n in dW)
    return (num / den) ** 0.5


if __name__ == "__main__":
    ctx = context()
    base = ctx.base_metrics()
    C = covariances(ctx)
    W0 = {n: ctx.base_state[n + ".weight"].cuda() for n in ctx.names}
    w0 = sum(float(w.double().square().sum()) for w in W0.values())

    # Prediction from activations alone: for each threshold, pretraining and task energy left in the free subspace.
    pred = {}
    for e in ENERGIES:
        U = ctx.activation_basis(e, "mix")
        free = lambda Cs: sum(float(torch.trace(Cs[n]) - torch.trace(U[n].double().T @ Cs[n] @ U[n].double()))
                              for n in Cs) / sum(float(torch.trace(Cs[n])) for n in Cs)
        pred[e] = {"pre_free": free(C["mix"]), "task_free": free(C["task"])}
        pred[e]["ratio"] = pred[e]["pre_free"] / pred[e]["task_free"]
        print(f"energy {e}: pretraining energy free {pred[e]['pre_free']:.3f}, task energy free "
              f"{pred[e]['task_free']:.3f}, leakage/room {pred[e]['ratio']:.3f}", flush=True)

    arms = {f"GPM {e}": {"method": "gpm", "gpm_energy": e, **GPM} for e in ENERGIES}
    arms.update({f"LoRA r{r}": {"rank": r} for r in RANKS})
    rows = []
    for name, kw in arms.items():
        for s in SEEDS:
            dW = run(ctx, lr=1e-3, steps=1000, snapshots=(1000,), seed=s, **kw)["snaps"][1000]["ema"]
            m = deltas(ctx.eval_delta(dW), base)
            r = {"arm": name, "seed": s, "rho": (sum(float(d.double().square().sum()) for d in dW.values()) / w0) ** 0.5,
                 "rho_C": weighted(dW, W0, C["mix"]), "rho_task": weighted(dW, W0, C["task"]), **m}
            rows.append(r)
        g = [r for r in rows if r["arm"] == name]
        print(name, " ".join(f"{k} {np.mean([r[k] for r in g]):.4g}" for k in
                             ("rho", "rho_C", "rho_task", "task", "bio_nll", "lm")), flush=True)
    for k in ("bio_nll", "lm", "kl"):
        print(k, " ".join(f"spearman vs {x}: {spearmanr([r[x] for r in rows], [r[k] for r in rows])[0]:.3f}"
                          for x in ("rho", "rho_C", "rho_task")))
    save(tag("q16_weighted_rho"), {"pred": pred, "rows": rows})
