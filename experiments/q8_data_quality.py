"""Q8: does the safe distance depend on data quality, and does adaptive weight decay track it?

Fine-tuning labels are flipped at random at several rates. Noisier labels mean noisier gradients, so if forgetting
is noise the break-even distance should fall as label noise rises. We compare, at every noise level, a fixed cap,
a fixed decay toward the base and adaptive decay (Ghiasi et al. 2023: strength proportional to ||grad|| / ||dW||,
which rises automatically with gradient noise), all exported through an EMA.
"""

from experiments.q5_measure_noise import break_even
from lwn.adapters import rho
from lwn.runs import run, save, summary, table
from lwn.train import Context

NOISE = (0.0, 0.1, 0.2, 0.3, 0.4)
ARMS = {"none": {}, "cap 1%": {"cap": 0.01}, "cap 2%": {"cap": 0.02}, "wd 1": {"wd": 1.0},
        "awd 0.02": {"awd": 0.02}, "awd 0.05": {"awd": 0.05}}

if __name__ == "__main__":
    ctx = Context()
    base_nll = ctx.base_metrics()["bio"]["nll"]
    rows = []
    for q in NOISE:
        for name, kw in ARMS.items():
            r = run(ctx, lr=1e-3, label_noise=q, **kw)
            ema = r["snaps"][1000]["ema"]
            be = break_even(ctx, r["snaps"][1000]["raw"], base_nll) if name == "none" else None
            rows.append({"label_noise": q, "arm": name, "rho%": round(100 * rho(ema, ctx.w0_sq), 3),
                         "break_even%": None if be is None else round(100 * be, 3), **summary(ctx.eval_delta(ema))})
    table(rows, ["label_noise", "arm", "rho%", "break_even%", "task", "bio", "bio_nll", "lm", "kl"])
    save("q8_data_quality", {"rows": rows})
