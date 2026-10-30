"""Splitting dW into signal and noise, and measuring how much each part costs.

With K runs that differ only in seed, each update is dW_k = S + D_k + J_k:
  S    the shared (systematic) part every seed agrees on: estimated by mean pairwise inner products
  D_k  slow seed-specific drift: what survives an EMA but differs between seeds
  J_k  fast jitter: raw minus EMA
Energies are reported in rho^2 units (divided by ||W0||^2), so sqrt(energy) reads as a relative norm.
"""

import itertools

import numpy as np
import torch


def dot(a, b):
    return sum(float((a[n].double() * b[n].double()).sum()) for n in a)


def add(*terms):
    """add((w1, dW1), (w2, dW2), ...) -> sum_k w_k dW_k."""
    names = terms[0][1].keys()
    return {n: sum(w * d[n] for w, d in terms) for n in names}


def mean(deltas):
    return add(*[(1 / len(deltas), d) for d in deltas])


def energies(raws, emas, w0_sq):
    """Signal / drift / jitter energies from K seeds' raw and EMA dW at one step."""
    K = len(raws)
    pairs = list(itertools.combinations(range(K), 2))
    S2 = np.mean([dot(emas[i], emas[j]) for i, j in pairs]) / w0_sq
    J2 = np.mean([dot(add((1, r), (-1, e)), add((1, r), (-1, e))) for r, e in zip(raws, emas)]) / w0_sq
    E2 = np.mean([dot(e, e) for e in emas]) / w0_sq
    cos = np.mean([dot(emas[i], emas[j]) / np.sqrt(dot(emas[i], emas[i]) * dot(emas[j], emas[j])) for i, j in pairs])
    return {"signal": S2, "drift": E2 - S2, "jitter": J2, "cross_seed_cos": cos}


@torch.no_grad()
def curvature(ctx, direction, at=None, eps=1e-3):
    """Average pretraining curvature along a dW direction: 2 KL(p_x || p_{x+eps v}) / eps^2 for unit v (rho units).

    This is the Fisher Rayleigh quotient v^T F v on the probe mixture, by finite differences, where x is the base
    (or base + `at`), on held-out text. eps is relative to ||W0||, matching the rho scale.
    """
    norm = np.sqrt(dot(direction, direction))
    scale = eps * np.sqrt(ctx.w0_sq) / norm
    m = ctx.eval_model
    m.load_state_dict(ctx.base_state)
    mods = dict(m.named_modules())
    if at:
        for n, d in at.items():
            mods[n].weight += d.to(mods[n].weight)
    from .model import val_logp

    p = val_logp(m, ctx.data)
    for n, d in direction.items():
        mods[n].weight += scale * d.to(mods[n].weight)
    q = val_logp(m, ctx.data)
    m.load_state_dict(ctx.base_state)
    kl = float((p.exp() * (p - q)).sum(-1).mean())
    return 2 * kl / eps**2


def scale_sweep(ctx, delta, scales=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
    """Post-hoc shrink/stretch of one dW along its own direction: the forgetting-vs-rho curve of a single update."""
    out = []
    for s in scales:
        m = ctx.eval_delta({n: s * d for n, d in delta.items()})
        out.append({"scale": s, "rho": s * np.sqrt(dot(delta, delta) / ctx.w0_sq), "metrics": m})
    return out
