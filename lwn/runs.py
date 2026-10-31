"""Cached fine-tuning runs and small reporting helpers shared by the experiment entrypoints."""

import hashlib
import json
from pathlib import Path

import torch

from .model import MODEL
from .train import FT, finetune

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / ("runs" if MODEL == "small" else f"runs-{MODEL}")
LATER = {"subset": None, "optimizer": "adamw", "gpm_source": "val", "evals": (), "gpm_out": 0.0}


def run(ctx, **kw):
    """finetune(FT(**kw)), cached on disk by its config so experiments can share and resume runs."""
    cfg = FT(**kw)
    # Fields added after the first runs are left out of the key at their defaults, so older cached runs stay valid.
    fields = {k: v for k, v in cfg.__dict__.items() if not (k in LATER and v == LATER[k])}
    if ctx.task != "badending":
        fields["task"] = ctx.task
    key = hashlib.sha1(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = CACHE / f"{key}.pt"
    if path.exists():
        return torch.load(path, map_location="cuda", weights_only=False)
    out = finetune(ctx, cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def summary(m):
    """The handful of numbers every table reports, from one ctx.eval_delta / curve metrics dict."""
    b = m["bio"]
    return {
        "task": round(m["task"]["acc"], 4),
        "bio": round(b["acc"], 4),
        "bio_nll": round(b["nll"], 4),
        "lm": round(m["lm"], 4),
        "kl": round(m.get("kl", 0.0), 5),
        **{k: round(v, 4) for k, v in b.items() if k.startswith("acc_")},
    }


def save(name, obj):
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float) + "\n")


def table(rows, cols):
    widths = [max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(f"{r.get(c, '')}".ljust(w) for c, w in zip(cols, widths)))
