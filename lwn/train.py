"""Pretraining the base model, and one configurable fine-tuning run that records dW snapshots for analysis."""

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .adapters import Adapter, rho
from .model import MODEL, Data, evaluate, lm_loss, make_model, targets, task_loss, val_logp

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / ("base.pt" if MODEL == "small" else f"base-{MODEL}.pt")


def pretrain(steps=20_000, batch=64, lr=2e-3, seed=0, log_every=1000):
    data = Data()
    model = make_model(data.vocab, seed=seed).cuda()
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:  # warmup, then cosine to 10%
            g["lr"] = lr * min(1, (step + 1) / 500) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, data.lm_batch(batch, rng))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            m = evaluate(model, data)
            bio = {k: round(v, 3) for k, v in m["bio"].items() if k.startswith("acc")}
            print(step + 1, f"{time.time() - t0:.0f}s train {loss.item():.3f} val {m['lm']:.3f}",
                  f"task {m['task']['acc']:.3f}", bio, flush=True)
    BASE.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), BASE)
    return model


@dataclass
class FT:
    """One fine-tuning run. Defaults mirror the LoRA recipe studied in the main project."""

    method: str = "lora"  # lora | pissa | milora | olora | gpm | full
    rank: int = 16
    lr: float = 1e-3
    schedule: str = "const"  # const | cosine | isqrt | linear (each with a 30-step warmup)
    steps: int = 1000
    batch: int = 32
    seed: int = 0  # adapter init
    data_seed: int | None = None  # batch order; defaults to seed
    cap: float | None = None  # per-matrix radial cap on ||dW|| / ||W0||
    ema_examples: int | None = 1024  # EMA half-life in examples (None: no EMA)
    clip: float | None = 1.0
    spike: tuple[int, float] | None = None  # (step, LR multiplier) for one step
    cooldown_from: tuple[int, ...] = ()  # branch here and decay LR linearly to 0 over 20% more steps
    replay: float = 0.0  # weight of an LM loss on pretraining text added to every step
    label_noise: float = 0.0  # fraction of fine-tuning labels replaced by a random answer
    ewc: float = 0.0  # weight of a diagonal-Fisher penalty on dW (Fisher of the base on held-out text)
    wd: float = 0.0  # decoupled decay of dW toward the base: dW <- (1 - lr wd) dW each step
    awd: float = 0.0  # adaptive decay (Ghiasi et al. 2023) toward the base: wd_t = awd * ||grad|| / ||dW||
    gpm_energy: float = 0.9  # gpm: fraction of pretraining activation energy whose subspace the adapter avoids
    gpm_source: str = "val"  # gpm: estimate that subspace from held-out stories ("val") or the pretraining mix ("mix")
    subset: int | None = None  # train on only the first `subset` task examples; batch >= subset means full-batch GD
    optimizer: str = "adamw"  # adamw | sgd (momentum 0.9)
    snapshots: tuple[int, ...] = (125, 250, 500, 1000)
    eval_every: int = 0  # 0: evaluate at snapshots only
    extra: dict = field(default_factory=dict)


def lr_at(cfg, step):
    warm = min(1.0, (step + 1) / 30)
    t = step / max(1, cfg.steps)
    f = {
        "const": 1.0,
        "cosine": 0.5 * (1 + math.cos(math.pi * t)),
        "linear": 1 - t,
        "isqrt": min(1.0, math.sqrt(30 / max(step, 1))),
    }[cfg.schedule]
    lr = cfg.lr * warm * f
    if cfg.spike and step == cfg.spike[0]:
        lr *= cfg.spike[1]
    return lr


class Context:
    """Shared, read-only objects for many runs: data, base weights and the base's held-out log-probs."""

    def __init__(self, task="badending", **data_kw):
        self.task = task
        self.data = Data(task=task, **data_kw)
        base = make_model(self.data.vocab).cuda()
        base.load_state_dict(torch.load(BASE, map_location="cuda"))
        self.base_state = copy.deepcopy(base.state_dict())
        self.eval_model = base
        self.names = list(targets(base))
        self.w0_sq = sum(float(self.base_state[n + ".weight"].square().sum()) for n in self.names)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.base_logp = val_logp(base, self.data)
        self._fisher = None
        self._basis = {}

    def fresh(self):
        model = make_model(self.data.vocab).cuda()
        model.load_state_dict(self.base_state)
        return model

    @torch.no_grad()
    def eval_delta(self, deltas):
        """Evaluate base + dW (any parametrisation) on every metric, including KL on held-out text."""
        m = self.eval_model
        mods = dict(m.named_modules())
        for n, d in deltas.items():
            mods[n].weight += d.to(mods[n].weight)
        out = evaluate(m, self.data, self.base_logp)
        m.load_state_dict(self.base_state)
        return out

    def base_metrics(self):
        return self.eval_delta({})

    def fisher(self):
        """Diagonal Fisher of the base on held-out text (labels sampled from the model), per adapted matrix."""
        if self._fisher is None:
            model = self.fresh()
            mods = {n: m for n, m in model.named_modules() if n in self.names}
            fis = {n: torch.zeros_like(m.weight) for n, m in mods.items()}
            x = self.data.val.cuda()
            for i in range(0, len(x), 16):
                model.zero_grad(set_to_none=True)
                logits = model(input_ids=x[i : i + 16, :-1]).logits.float()
                y = torch.distributions.Categorical(logits=logits).sample()
                torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                                  reduction="sum").backward()
                for n, m in mods.items():
                    fis[n] += m.weight.grad.square()
            tokens = x.shape[0] * (x.shape[1] - 1)
            self._fisher = {n: f / tokens for n, f in fis.items()}
        return self._fisher

    @torch.no_grad()
    def activation_basis(self, energy, source="val"):
        """Per adapted matrix, the top input directions holding `energy` of activations' squared norm, on held-out
        stories ("val") or on a fixed sample of the pretraining mixture, biographies included ("mix")."""
        if (energy, source) not in self._basis:
            acts, hooks = {}, []
            mods = {n: m for n, m in self.eval_model.named_modules() if n in self.names}
            for n, m in mods.items():
                hooks.append(m.register_forward_hook(
                    lambda mod, i, o, n=n: acts.setdefault(n, []).append(i[0].reshape(-1, i[0].shape[-1]))))
            x = self.data.val[:32] if source == "val" else self.data.lm_batch(32, np.random.default_rng(123))
            self.eval_model(input_ids=x[:, :-1].cuda())
            for h in hooks:
                h.remove()
            out = {}
            for n, xs in acts.items():
                U, S, _ = torch.linalg.svd(torch.cat(xs).T.double(), full_matrices=False)
                frac = S.square().cumsum(0) / S.square().sum()
                out[n] = U[:, : int((frac < energy).sum()) + 1].float()
            self._basis[(energy, source)] = out
        return self._basis[(energy, source)]


def finetune(ctx, cfg):
    """Run one fine-tune. Returns per-snapshot dW (raw, EMA, cooldown branches) and metrics along the way."""
    data = ctx.data
    model = ctx.fresh()
    basis = ctx.activation_basis(cfg.gpm_energy, cfg.gpm_source) if cfg.method == "gpm" else None
    ad = Adapter(model, cfg.method, cfg.rank, seed=cfg.seed, basis=basis)
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(ad.params(), lr=cfg.lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(ad.params(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
    dseed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    rng = np.random.default_rng(dseed)
    ema = ad.state() if cfg.ema_examples else None
    decay = 0.5 ** (cfg.batch / cfg.ema_examples) if cfg.ema_examples else None
    fisher = ctx.fisher() if cfg.ewc else None
    out = {"cfg": dict(cfg.__dict__), "snaps": {}, "curve": [], "branches": {}, "w0_sq": ctx.w0_sq}
    saved = {}

    def step_once(lr, r):
        for g in opt.param_groups:
            g["lr"] = lr
        if cfg.subset and cfg.batch >= cfg.subset:  # deterministic full-batch gradient
            batch, labels = data.task_batch(0, r, idx=list(range(cfg.subset)))
        elif cfg.subset:
            batch, labels = data.task_batch(0, r, idx=list(r.integers(cfg.subset, size=cfg.batch)))
        else:
            batch, labels = data.task_batch(cfg.batch, r)
        if cfg.label_noise:
            flip = torch.from_numpy(r.random(len(labels)) < cfg.label_noise)
            choices = torch.tensor(data.choices)
            wrong = choices[torch.from_numpy(r.integers(len(choices), size=len(labels)))]
            labels = torch.where(flip, wrong, labels)  # a random answer (possibly the right one, as with real noise)
        opt.zero_grad(set_to_none=True)
        (ids, att), n, total = batch, len(labels), 0.0
        for i in range(0, n, 128):  # gradient accumulation keeps full-batch runs within memory
            with torch.autocast("cuda", dtype=torch.bfloat16):
                part = task_loss(model, (ids[i : i + 128], att[i : i + 128]), labels[i : i + 128]) * min(128, n - i) / n
            part.backward()
            total += float(part.detach())
        loss = torch.tensor(total)
        extra = 0.0
        if cfg.replay:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                extra = cfg.replay * lm_loss(model, data.lm_batch(8, r))
        if fisher is not None:
            extra = extra + cfg.ewc * sum((fisher[n] * d.square()).sum() for n, d in ad.deltas().items())
        if torch.is_tensor(extra):
            extra.backward()
            loss = loss + float(extra.detach())
        gn = float(torch.nn.utils.clip_grad_norm_(ad.params(), cfg.clip if cfg.clip else float("inf")))
        opt.step()
        ad.project()
        lam = cfg.wd
        if cfg.awd:
            lam += cfg.awd * gn / max(math.sqrt(sum(float(d.detach().square().sum()) for d in ad.deltas().values())), 1e-12)
        if lam:
            ad.shrink(max(0.0, 1 - lr * lam))
        if cfg.cap:
            ad.cap(cfg.cap)
        return float(loss.detach()), gn

    for step in range(cfg.steps):
        if step in cfg.cooldown_from:
            saved[step] = (ad.state(), copy.deepcopy(opt.state_dict()), copy.deepcopy(rng))
        lr = lr_at(cfg, step)
        loss, gn = step_once(lr, rng)
        if ema is not None:
            with torch.no_grad():
                for e, p in zip(ema, ad.params()):
                    e.lerp_(p.detach(), 1 - decay)
        s = step + 1
        rec = {"step": s, "loss": loss, "grad_norm": gn, "lr": lr}
        if (cfg.eval_every and s % cfg.eval_every == 0) or s in cfg.snapshots:
            d = {n: x.detach() for n, x in ad.deltas().items()}
            rec["rho"] = rho(d, ctx.w0_sq)
            rec["metrics"] = ctx.eval_delta(d)
        out["curve"].append(rec)
        if s in cfg.snapshots:
            snap = {"raw": {n: d.detach().clone() for n, d in ad.deltas().items()}}
            if ema is not None:
                snap["ema"] = ad.deltas_of(ema)
            out["snaps"][s] = snap
    for b, (st, os, r) in saved.items():  # cooldown branches continue the same data stream from step b
        ad.load(st)
        opt.load_state_dict(os)
        K = max(1, int(0.2 * b))
        base_lr = lr_at(cfg, b)
        for k in range(K):
            step_once(base_lr * (K - k) / K, r)
        out["branches"][b] = {"steps": K, "raw": {n: d.detach().clone() for n, d in ad.deltas().items()}}
    return out
