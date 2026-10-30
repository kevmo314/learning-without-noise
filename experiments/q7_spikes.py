"""Q7: are loss spikes catastrophic for retention even after the loss recovers?

One step at k times the LR, midway through a run; everything else (init, data order) matches the control, so the
control is an exact counterfactual. If forgetting is noise in directions that pretraining is sensitive to but the
task is not, the task loss should recover (the task pulls those directions back) while knowledge does not.
"""

from lwn.runs import run, save, summary, table
from lwn.train import Context

FACTORS = (1, 10, 30, 100)
AT = 500

if __name__ == "__main__":
    ctx = Context()
    rows = []
    for k in FACTORS:
        r = run(ctx, lr=1e-3, spike=None if k == 1 else (AT, float(k)), eval_every=50,
                snapshots=(250, 500, 1000), ema_examples=None)
        for c in r["curve"]:
            if "metrics" in c and c["step"] >= AT - 50:
                losses = [x["loss"] for x in r["curve"] if c["step"] - 50 < x["step"] <= c["step"]]
                rows.append({"spike": k, "step": c["step"], "train_loss": round(sum(losses) / len(losses), 4),
                             "rho%": round(100 * c["rho"], 3), **summary(c["metrics"])})
    table(rows, ["spike", "step", "train_loss", "rho%", "task", "bio", "bio_nll", "lm", "kl"])
    save("q7_spikes", {"rows": rows})
