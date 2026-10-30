"""Q12: is the measured forgetting lost knowledge, or just miscalibrated answers? (reviewer-requested audit)

A fine-tune can lower benchmark scores without erasing anything, e.g. by shifting the model's prior toward some
answers or changing its confidence. For each checkpoint we refit, on half of the people, a temperature and one bias
per candidate value (e.g. a prior for "Boston"), and score knowledge on the other half; and we refit a temperature on
held-out language loss. The base model gets the same refit. If forgetting were calibration, refitting would recover
most of the drop; if it is lost knowledge, the drop survives.
"""

import torch
import torch.nn.functional as F

from lwn.adapters import rho
from lwn.model import bio_matrix
from lwn.runs import run, save, table
from lwn.train import Context


def refit_bio(scores, keys, people_half):
    vocab = {k: i for i, k in enumerate(sorted({k for row in keys for k in row}))}
    idx = torch.tensor([[vocab[k] for k in row] for row in keys])
    T = torch.zeros(1, requires_grad=True)
    b = torch.zeros(len(vocab), requires_grad=True)
    fit, te = people_half == 0, people_half == 1
    f = lambda m: scores[m] * T.exp() + b[idx[m]]
    opt = torch.optim.LBFGS([T, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = -f(fit).log_softmax(1)[:, 0].mean() + 1e-3 * b.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float((scores[te].argmax(1) == 0).float().mean()), float((f(te).argmax(1) == 0).float().mean())


@torch.no_grad()
def lm_logits(model, data):
    x = data.val.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(input_ids=x[:, :-1]).logits.float(), x[:, 1:]


def refit_lm(logits, y):
    n = len(logits) // 2
    T = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([T], max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy((logits[:n] * T.exp()).reshape(-1, logits.shape[-1]), y[:n].reshape(-1))
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        raw = F.cross_entropy(logits[n:].reshape(-1, logits.shape[-1]), y[n:].reshape(-1))
        cal = F.cross_entropy((logits[n:] * T.exp()).reshape(-1, logits.shape[-1]), y[n:].reshape(-1))
    return float(raw), float(cal)


if __name__ == "__main__":
    ctx = Context()
    data = ctx.data
    keys = [[tuple(c) for c in it["cands"]] for it in data.bio]
    half = torch.tensor([it["rank"] % 2 for it in data.bio])

    def audit(name, deltas):
        m = ctx.eval_model
        mods = dict(m.named_modules())
        with torch.no_grad():
            for n, d in deltas.items():
                mods[n].weight += d.to(mods[n].weight)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = bio_matrix(m, data).float()
        b_raw, b_cal = refit_bio(scores, keys, half)
        l_raw, l_cal = refit_lm(*lm_logits(m, data))
        m.load_state_dict(ctx.base_state)
        return {"checkpoint": name, "rho%": round(100 * rho(deltas, ctx.w0_sq), 3) if deltas else 0.0,
                "bio_raw": round(b_raw, 4), "bio_cal": round(b_cal, 4), "lm_raw": round(l_raw, 4), "lm_cal": round(l_cal, 4)}

    rows = [audit("base", {})]
    for lr in (3e-4, 1e-3, 3e-3):
        r = run(ctx, lr=lr, seed=0, cooldown_from=(500,))
        for s in (250, 1000):
            for kind in ("raw", "ema"):
                rows.append(audit(f"lr={lr:g} step {s} {kind}", r["snaps"][s][kind]))
    b0, l0 = rows[0], rows[0]
    for r in rows:
        r["bio_drop_raw"] = round(b0["bio_raw"] - r["bio_raw"], 4)
        r["bio_drop_cal"] = round(b0["bio_cal"] - r["bio_cal"], 4)
        r["lm_rise_raw"] = round(r["lm_raw"] - l0["lm_raw"], 4)
        r["lm_rise_cal"] = round(r["lm_cal"] - l0["lm_cal"], 4)
    table(rows, ["checkpoint", "rho%", "bio_raw", "bio_cal", "bio_drop_raw", "bio_drop_cal", "lm_rise_raw", "lm_rise_cal"])
    save("q12_calibration", {"rows": rows})
