"""The ~7M-parameter GPT-2 (transformers), the data views every experiment uses, and evaluation.

Retention is measured three ways, all graded rather than saturated:
  lm      next-token loss on held-out TinyStories (general language)
  bio     knowledge recall, scored like a likelihood-based multiple-choice benchmark: for "Alice B. Smith was born in",
          the true city must out-score 7 random alternatives. Reported overall and by pretraining exposure bucket.
  kl      KL(base || model) on held-out text: how far the output distribution moved
The fine-tuning task is "Does this story have a bad ending?" answered " yes" or " no".
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

from .corpus import OPTIONS, PROBE

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CTX = 256
TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
QUESTION = "\nQuestion: Does this story have a bad ending?\nAnswer:"
BUCKETS = [(0, 30), (30, 300), (300, 3000), (3000, 10_000)]  # exposure rank ranges, most to least frequent


def make_model(vocab, n_layer=8, n_embd=256, n_head=8, seed=0):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=vocab, n_positions=CTX, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=0, eos_token_id=0)
    return GPT2LMHeadModel(cfg)


def targets(model):
    return {n: m for n, m in model.named_modules() if n.endswith(TARGETS)}


class Data:
    def __init__(self, n_bio_people=1000, bio_attrs=("birth_city", "university", "company"), n_val=128, seed=0):
        self.tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
        self.vocab = self.tok.get_vocab_size()
        self.eot = self.tok.token_to_id("<|endoftext|>")
        self.train_ids = np.memmap(DATA / "pretrain.bin", dtype=np.uint16, mode="r")
        val = np.fromfile(DATA / "val.bin", dtype=np.uint16)
        self.val = torch.from_numpy(val[: n_val * (CTX + 1)].astype(np.int64)).view(n_val, CTX + 1)
        self.yes, self.no = self.tok.encode(" yes").ids[0], self.tok.encode(" no").ids[0]
        self.task_train = self._task(DATA / "task_train.jsonl")
        self.task_holdout = self._task(DATA / "task_holdout.jsonl")
        self.bio = self._bio(n_bio_people, bio_attrs, seed)

    # ---- pretraining stream
    def lm_batch(self, n, rng):
        starts = rng.integers(0, len(self.train_ids) - CTX - 1, size=n)
        x = np.stack([self.train_ids[s : s + CTX + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x)

    # ---- fine-tuning task
    def _task(self, path):
        rows = [json.loads(l) for l in open(path)]
        q = self.tok.encode(QUESTION).ids
        ids, labels = [], []
        for r in rows:
            s = self.tok.encode(r["story"]).ids[: CTX - len(q) - 1]
            ids.append(s + q)
            labels.append(self.yes if r["label"] else self.no)
        return ids, torch.tensor(labels)

    def task_batch(self, n, rng, split="train", idx=None):
        ids, labels = self.task_train if split == "train" else self.task_holdout
        if idx is None:
            idx = rng.integers(len(ids), size=n)
        return pad([ids[i] for i in idx]), labels[idx]

    # ---- knowledge probe
    def _bio(self, n_people, attrs, seed):
        pop = json.loads((DATA / "people.json").read_text())
        ranks = np.unique(np.round(np.logspace(0, np.log10(len(pop)), n_people * 2)).astype(int) - 1)
        rng = random.Random(seed)
        ranks = sorted(rng.sample(list(ranks), min(n_people, len(ranks))))
        items = []
        for r in ranks:
            p = pop[r]
            for a in attrs:
                cands = [p[a]] + rng.sample([o for o in OPTIONS[a] if o != p[a]], 7)
                prompt = self.tok.encode(PROBE[a].format(n=p["name"])).ids
                items.append({"rank": r, "attr": a, "prompt": prompt,
                              "cands": [self.tok.encode(" " + c).ids for c in cands]})
        return items


def pad(seqs, value=0):
    """Left-pad so every sequence ends at the last position; returns ids and an attention mask."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), value, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s) :] = torch.tensor(s)
        att[i, L - len(s) :] = 1
    return ids, att


def last_logits(model, batch):
    (ids, att) = batch
    pos = (att.cumsum(1) - 1).clamp_min(0)
    out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda())
    return out.logits[:, -1].float()


def task_loss(model, batch, labels):
    return F.cross_entropy(last_logits(model, batch), labels.cuda())


def lm_loss(model, x):
    x = x.cuda()
    logits = model(input_ids=x[:, :-1]).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))


@torch.no_grad()
def val_logp(model, data, positions=64):
    """Log-probs at the last `positions` positions of each held-out window (for KL to the base)."""
    x = data.val.cuda()
    logits = model(input_ids=x[:, :-1]).logits[:, -positions:]
    return logits.float().log_softmax(-1)


@torch.no_grad()
def evaluate(model, data, base_logp=None, task_n=1000):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = {"lm": float(lm_loss(model, data.val))}
        # task: two-way choice between " yes" and " no"
        ids, labels = data.task_holdout
        acc = nll = 0.0
        for i in range(0, min(task_n, len(ids)), 256):
            idx = list(range(i, min(i + 256, task_n, len(ids))))
            lg = last_logits(model, pad([ids[j] for j in idx]))[:, [data.no, data.yes]].log_softmax(-1)
            y = (labels[idx] == data.yes).long().cuda()
            acc += float((lg.argmax(-1) == y).sum())
            nll += float(-lg.gather(1, y[:, None]).sum())
        n = min(task_n, len(ids))
        out["task"] = {"acc": acc / n, "nll": nll / n}
        out["bio"] = bio_scores(model, data)
        if base_logp is not None:
            lp = val_logp(model, data)
            out["kl"] = float((base_logp.exp() * (base_logp - lp)).sum(-1).mean())
    model.train()
    return out


@torch.no_grad()
def bio_scores(model, data, chunk=2048):
    """Likelihood multiple choice over 8 candidates per (person, attribute)."""
    seqs, owners, spans = [], [], []
    for k, it in enumerate(data.bio):
        for c in it["cands"]:
            seqs.append(it["prompt"] + c)
            owners.append(k)
            spans.append(len(c))
    scores = torch.empty(len(seqs))
    for i in range(0, len(seqs), chunk):
        batch = seqs[i : i + chunk]
        ids, att = pad(batch)
        pos = (att.cumsum(1) - 1).clamp_min(0)
        lp = model(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda()).logits.float()
        lp = lp[:, :-1].log_softmax(-1).gather(2, ids[:, 1:, None].cuda())[..., 0]
        span = torch.tensor(spans[i : i + chunk], device=lp.device)
        keep = torch.arange(lp.shape[1], device=lp.device)[None] >= lp.shape[1] - span[:, None]
        scores[i : i + chunk] = (lp * keep).sum(1).cpu()
    scores = scores.view(len(data.bio), 8)
    correct = (scores.argmax(1) == 0).float()
    nll = -scores.log_softmax(1)[:, 0]
    ranks = torch.tensor([it["rank"] for it in data.bio])
    out = {"acc": float(correct.mean()), "nll": float(nll.mean())}
    for lo, hi in BUCKETS:
        m = (ranks >= lo) & (ranks < hi)
        out[f"acc_{lo}_{hi}"] = float(correct[m].mean())
        out[f"nll_{lo}_{hi}"] = float(nll[m].mean())
    return out
