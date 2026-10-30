"""Q1: does the small model reproduce what we saw fine-tuning Qwen3.5 with LoRA?

The claims under test:
  a. Forgetting grows with the distance moved from the base (rho), and small moves can even help.
  b. Higher learning rates forget more at the same number of steps.
  c. An EMA of the weights forgets less than the raw weights at nearly the same rho.
  d. Decaying the LR late (a cooldown branch) recovers forgetting without shrinking rho.
  e. Shrinking the final update after training (post-hoc) can improve both retention and the task.
"""

from lwn.analysis import scale_sweep
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

LRS = (1e-4, 3e-4, 1e-3, 3e-3)
SEEDS = (0, 1)

if __name__ == "__main__":
    ctx = Context()
    base = summary(ctx.base_metrics())
    print("base", base)
    rows, sweeps = [], []
    for lr in LRS:
        for seed in SEEDS:
            r = run(ctx, lr=lr, seed=seed, cooldown_from=(500,))
            for c in r["curve"]:
                if "metrics" in c:
                    rows.append({"lr": lr, "seed": seed, "step": c["step"], "kind": "raw",
                                 "rho%": round(100 * c["rho"], 3), **summary(c["metrics"])})
            for s, snap in r["snaps"].items():
                rows.append({"lr": lr, "seed": seed, "step": s, "kind": "ema",
                             "rho%": round(100 * rho(snap["ema"], ctx.w0_sq), 3),
                             **summary(ctx.eval_delta(snap["ema"]))})
            for b, br in r["branches"].items():
                rows.append({"lr": lr, "seed": seed, "step": b + br["steps"], "kind": f"cooldown@{b}",
                             "rho%": round(100 * rho(br["raw"], ctx.w0_sq), 3), **summary(ctx.eval_delta(br["raw"]))})
            if seed == 0:
                for p in scale_sweep(ctx, r["snaps"][1000]["raw"]):
                    sweeps.append({"lr": lr, "scale": p["scale"], "rho%": round(100 * p["rho"], 3),
                                   **summary(p["metrics"])})
    cols = ["lr", "seed", "step", "kind", "rho%", "task", "bio", "bio_nll", "lm", "kl"]
    table(sorted(rows, key=lambda r: (r["lr"], r["seed"], r["step"], r["kind"])), cols)
    print("\npost-hoc scaling of the step-1000 raw update (seed 0)")
    table(sweeps, ["lr", "scale", "rho%", "task", "bio", "bio_nll", "lm", "kl"])
    save("q1_reproduce", {"base": base, "rows": rows, "sweeps": sweeps})
