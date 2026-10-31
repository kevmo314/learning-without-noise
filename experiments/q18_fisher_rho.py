"""Does a Fisher on the knowledge itself predict knowledge loss, where activation energy (q16) does not?

GPM and the activation-weighted rho of q16 only see the input side of each matrix. Recalling a fact may depend on
narrow output directions too. The K-FAC approximation of the Fisher of a layer y = x W is A (x) G, with A the input
covariance and G the covariance of the loss gradient at the output; the predicted loss increase from dW is
1/2 tr(A dW G dW^T). Labels are sampled from the base model, so this is the true Fisher (a Gauss-Newton curvature),
computed before any fine-tuning. Two Fishers are compared on the same cached runs as q16 (GPM at six thresholds, LoRA
at nine ranks, 6 seeds each; metrics from q16):
  F_bio  on the knowledge probes: biography prompt + true answer, loss on the answer tokens
  F_lm   on ordinary held-out text
plus the diagonal Fisher on held-out text (Context.fisher) and the activation-weighted rhos from q16."""

import json

import numpy as np
import torch

from experiments.q16_weighted_rho import ENERGIES, RANKS, spearman
from lwn.fig import SEEDS, context, tag
from lwn.model import pad
from lwn.runs import ROOT, run, save


def kfac(ctx, batches):
    """Per adapted matrix, (A, G): input and output-gradient covariances under labels sampled from the base model.
    `batches` yields (ids, att, loss_mask): the loss is taken only where loss_mask is set."""
    model = ctx.fresh()
    for p in model.parameters():
        p.requires_grad_(False)
    mods = {n: m for n, m in model.named_modules() if n in ctx.names}
    A = {n: 0 for n in mods}
    G = {n: 0 for n in mods}
    saved, hooks, cnt = {}, [], [0, 0]

    def fwd(n):
        def f(mod, i, o):
            saved[n] = i[0].detach()
            if o.requires_grad:
                o.register_hook(lambda g, n=n: G.__setitem__(n, G[n] + (g.reshape(-1, g.shape[-1]).double().T @
                                                                         g.reshape(-1, g.shape[-1]).double())))
        return f

    for n, m in mods.items():
        hooks.append(m.register_forward_hook(fwd(n)))
    # Outputs need grad for the hooks to fire: make the embeddings require grad.
    model.transformer.wte.weight.requires_grad_(True)
    for ids, att, mask in batches:
        ids, att, mask = ids.cuda(), att.cuda(), mask.cuda()
        pos = (att.cumsum(1) - 1).clamp_min(0)
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos).logits[:, :-1].float()
        m = mask[:, 1:].bool()
        y = torch.distributions.Categorical(logits=logits[m]).sample()
        torch.nn.functional.cross_entropy(logits[m], y, reduction="sum").backward()
        real = att.reshape(-1).bool()
        for n in mods:
            x = saved[n].reshape(-1, saved[n].shape[-1])[real].double()
            A[n] = A[n] + x.T @ x
        cnt[0] += int(real.sum())
        cnt[1] += int(m.sum())
        model.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    return {n: (A[n] / cnt[0], G[n] / cnt[1]) for n in mods}


def bio_batches(ctx, n=512, bs=64):
    rng = np.random.default_rng(7)
    items = rng.choice(len(ctx.data.bio), n, replace=False)
    for i in range(0, n, bs):
        seqs, masks = [], []
        for j in items[i : i + bs]:
            it = ctx.data.bio[j]
            ans = it["cands"][0]  # the true answer
            seqs.append(it["prompt"] + ans)
            masks.append([0] * len(it["prompt"]) + [1] * len(ans))
        (ids, att), (mk, _) = pad(seqs), pad(masks)
        yield ids, att, mk


def lm_batches(ctx, n=64, bs=16):
    x = ctx.data.val[:n]
    for i in range(0, n, bs):
        ids = x[i : i + bs, :-1]  # windows carry one extra target token beyond the context length
        yield ids, torch.ones_like(ids), torch.ones_like(ids)


def fisher_dist(dW, F):
    return sum(float(torch.einsum("ij,jo,ok,ik->", F[n][0], dW[n].double(), F[n][1], dW[n].double())) for n in dW)


if __name__ == "__main__":
    ctx = context()
    F = {"F_bio": kfac(ctx, bio_batches(ctx)), "F_lm": kfac(ctx, lm_batches(ctx))}
    diag = {n: f.double() for n, f in ctx.fisher().items()}
    q16 = json.loads((ROOT / "results" / f"{tag('q16_weighted_rho')}.json").read_text())["rows"]
    GPM = {"gpm_source": "mix"}
    arms = {f"GPM {e}": {"method": "gpm", "gpm_energy": e, **GPM} for e in ENERGIES}
    arms.update({f"LoRA r{r}": {"rank": r} for r in RANKS})
    rows = []
    for row in q16:
        dW = {k: v.double() for k, v in run(ctx, lr=1e-3, steps=1000, snapshots=(1000,), seed=row["seed"],
                                               **arms[row["arm"]])["snaps"][1000]["ema"].items()}
        rows.append({**row, **{k: fisher_dist(dW, f) for k, f in F.items()},
                     "F_diag_lm": sum(float((diag[n] * dW[n].square()).sum()) for n in dW)})
    for name in arms:
        g = [r for r in rows if r["arm"] == name]
        print(name, " ".join(f"{k} {np.mean([r[k] for r in g]):.4g}" for k in
                             ("F_bio", "F_lm", "F_diag_lm", "bio_nll", "lm")), flush=True)
    for k in ("bio_nll", "lm", "kl"):
        print(k, " ".join(f"{x}: {spearman([r[x] for r in rows], [r[k] for r in rows]):.3f}"
                          for x in ("rho", "rho_C", "rho_bio", "F_bio", "F_lm", "F_diag_lm")))
    save(tag("q18_fisher_rho"), {"rows": rows})
