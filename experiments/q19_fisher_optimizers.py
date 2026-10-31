"""q18's Fisher distances on q17's runs: does the K-FAC Fisher put every optimizer and start (AdamW, Muon, SGD, with
and without GPM or a PiSSA start) on one forgetting curve, where plain and activation-weighted rho do not?"""

import json

import numpy as np

from experiments.q16_weighted_rho import spearman
from experiments.q17_muon import ARMS
from experiments.q18_fisher_rho import bio_batches, fisher_dist, kfac, lm_batches
from lwn.fig import context, tag
from lwn.runs import ROOT, run, save

if __name__ == "__main__":
    ctx = context()
    F = {"F_bio": kfac(ctx, bio_batches(ctx)), "F_lm": kfac(ctx, lm_batches(ctx))}
    q17 = json.loads((ROOT / "results" / f"{tag('q17_muon')}.json").read_text())["rows"]
    rows = []
    for row in q17:
        kw = ARMS[row["arm"]][0]
        dW = {k: v.double() for k, v in run(ctx, lr=row["lr"], steps=1000, snapshots=(1000,), seed=row["seed"],
                                               **kw)["snaps"][1000]["ema"].items()}
        rows.append({**row, **{k: fisher_dist(dW, f) for k, f in F.items()}})
    ok = [r for r in rows if r["task"] > 0.85]  # leave out diverged runs, whose losses saturate
    for k in ("bio_nll", "lm", "kl"):
        print(k, "(task > 85%)", " ".join(f"{x}: {spearman([r[x] for r in ok], [r[k] for r in ok]):.3f}"
                                         for x in ("rho", "rho_C", "rho_task", "F_bio", "F_lm")))
    for name in ARMS:
        for lr in sorted({r["lr"] for r in rows if r["arm"] == name}):
            g = [r for r in rows if r["arm"] == name and r["lr"] == lr]
            print(f"{name:18s} {lr:<7g}", " ".join(f"{k} {np.mean([r[k] for r in g]):.4g}" for k in
                                                   ("task", "bio_nll", "lm", "F_bio", "F_lm", "rho_C")))
    save(tag("q19_fisher_optimizers"), {"rows": rows})
