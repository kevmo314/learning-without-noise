"""The pretraining corpus and the fine-tuning task, built from TinyStories plus synthetic biographies.

Pretraining text mixes
  * TinyStories (Eldan & Li 2023): simple English that a few-million-parameter model learns fluently
  * biographies of synthetic people in the style of Allen-Zhu & Li's bioS ("Physics of Language Models 3.1"):
    six attributes per person, each sentence drawn from several templates and the sentences shuffled. People are
    exposed Zipf-style (person i weighted 1/(i+1)), so the model knows frequent people well and rare people
    partly or not at all: knowledge that competes for capacity, as in a real pretrained model.
Fine-tuning is a narrow multiple-choice task on unseen stories: does the story have a bad ending? The labels
are TinyStoriesInstruct's own "BadEnding" feature, i.e. real, imperfect annotations.

Run `python -m lwn.corpus` once; it writes everything under data/.
"""

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tinystories"
OUT = ROOT / "data"
N_PEOPLE = 10_000
VOCAB = 4096

FIRST = """Aaron Abigail Adrian Alice Amelia Andrew Anna Arthur Audrey Ava Benjamin Bella Caleb Camila Carter Charlotte
Chloe Claire Daniel David Dylan Eleanor Elena Elijah Ella Emily Emma Ethan Evelyn Felix Fiona Gabriel Grace Hannah Harper
Henry Isaac Isabella Jack Jacob James Jasmine Julia Julian Kai Leah Leo Levi Liam Lily Lucas Lucy Luna Mason Maya Mia
Miles Naomi Nathan Nora Oliver Olivia Oscar Owen Paige Penelope Quinn Riley Rose Ruby Ryan Samuel Sarah Scarlett Sebastian
Sofia Sophie Stella Theo Thomas Violet Wesley William Willow Wyatt Xavier Zoe""".split()
LAST = """Adams Allen Anderson Bailey Baker Barnes Bell Bennett Brooks Brown Bryant Butler Campbell Carter Clark Coleman
Collins Cook Cooper Cox Davis Edwards Evans Fisher Flores Foster Garcia Gomez Gonzalez Gray Green Griffin Hall Harris
Hayes Henderson Hill Howard Hughes Jackson James Jenkins Johnson Jones Kelly King Lee Lewis Long Lopez Martin Martinez
Miller Mitchell Moore Morgan Morris Murphy Nelson Nguyen Parker Patterson Perez Perry Peterson Phillips Powell Price
Ramirez Reed Richardson Rivera Roberts Robinson Rogers Ross Russell Sanders Scott Simmons Smith Stewart Sullivan Taylor
Thomas Thompson Torres Turner Walker Ward Watson White Williams Wilson Wood Wright Young""".split()
CITIES = """Austin Boston Chicago Denver Dallas Seattle Portland Phoenix Miami Atlanta Detroit Houston Memphis Nashville Omaha
Tulsa Tampa Orlando Raleigh Richmond Buffalo Albany Hartford Newark Pittsburgh Cleveland Columbus Cincinnati Louisville
Lexington Madison Milwaukee Minneapolis Kansas Wichita Lincoln Boise Spokane Tacoma Fresno Sacramento Oakland Reno Tucson
Albuquerque Savannah Charleston Norfolk Baltimore Providence""".split()
UNIS = [f"{w} University" for w in """Ashford Bayside Briarwood Cedarcrest Clearwater Eastfield Elmhurst Fairmont Glenbrook
Greenvale Hawthorne Highland Kingsbridge Lakewood Maplewood Meadowbrook Northgate Oakridge Pinecrest Ravenwood Redstone
Riverside Rosewood Silverlake Southport Stonebridge Summit Westbrook Whitfield Willowdale""".split()]
MAJORS = """Biology Chemistry Physics Mathematics History Economics Psychology Sociology Philosophy Literature Music Art
Architecture Engineering Medicine Nursing Law Education Geography Astronomy Geology Linguistics Statistics Finance
Marketing Journalism Theater Dance Agriculture Forestry""".split()
COMPANIES = [f"{w} {s}" for w, s in zip("""Acme Apex Atlas Beacon Bluebird Cobalt Crescent Delta Ember Falcon Granite
Harbor Horizon Ironwood Juniper Keystone Lantern Meridian Nimbus Orchid Pioneer Quartz Radiant Sequoia Summit Tidewater
Topaz Vertex Willow Zenith""".split(), ["Labs", "Foods", "Motors", "Health", "Media", "Energy"] * 5)]
MONTHS = "January February March April May June July August September October November December".split()

ATTRS = ["birth_date", "birth_city", "university", "major", "company", "company_city"]
TEMPLATES = {
    "birth_date": ["{n} was born on {v}.", "{n} came into the world on {v}.", "{n}'s birthday is {v}.",
                   "{n} celebrates a birthday on {v}."],
    "birth_city": ["{n} was born in {v}.", "{n} grew up in {v}.", "{n} spent childhood years in {v}.",
                   "{n} is originally from {v}."],
    "university": ["{n} studied at {v}.", "{n} graduated from {v}.", "{n} went to college at {v}.",
                   "{n} earned a degree from {v}."],
    "major": ["{n} studied {v}.", "{n} majored in {v}.", "{n} chose to focus on {v}.", "{n} learned {v} in college."],
    "company": ["{n} works for {v}.", "{n} has a job at {v}.", "{n} is employed by {v}.", "{n} joined {v}."],
    "company_city": ["{n} works in {v}.", "{n} moved to {v} for work.", "{n} commutes to an office in {v}.",
                     "{n} has an office in {v}."],
}
# The recall probe: the first template's text up to the value, e.g. "Alice Smith was born in" -> " Austin".
PROBE = {a: t[0].split("{v}")[0].rstrip() for a, t in TEMPLATES.items()}
OPTIONS = {"birth_city": CITIES, "company_city": CITIES, "university": UNIS, "major": MAJORS, "company": COMPANIES}


def people(seed=0):
    rng = random.Random(seed)
    names, seen = [], set()
    while len(names) < N_PEOPLE:
        n = f"{rng.choice(FIRST)} {rng.choice('ABCDEFGHJKLMNPRSTW')}. {rng.choice(LAST)}"
        if n not in seen:
            seen.add(n)
            names.append(n)
    return [
        {
            "name": n,
            "rank": i,
            "birth_date": f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(1950, 2005)}",
            "birth_city": rng.choice(CITIES),
            "university": rng.choice(UNIS),
            "major": rng.choice(MAJORS),
            "company": rng.choice(COMPANIES),
            "company_city": rng.choice(CITIES),
        }
        for i, n in enumerate(names)
    ]


def render_bio(p, rng):
    sents = [rng.choice(TEMPLATES[a]).format(n=p["name"], v=p[a]) for a in ATTRS]
    rng.shuffle(sents)
    return " ".join(sents)


def load_stories(files):
    for f in files:
        for x in json.load(open(RAW / f"data{f:02d}.json")):
            yield x["story"].strip(), x.get("instruction", {}).get("features", [])


def build(pretrain_files=range(0, 20), task_files=(48, 49), bio_frac=0.2, seed=0):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    rng = random.Random(seed)
    pop = people(seed)
    (OUT / "people.json").write_text(json.dumps(pop))
    stories = [s for s, _ in load_stories(pretrain_files)]
    rng.shuffle(stories)
    val, stories = stories[:2000], stories[2000:]
    # Bios: as many as needed for bio_frac of documents, people drawn Zipf-style.
    n_bio = int(len(stories) * bio_frac / (1 - bio_frac))
    w = 1 / np.arange(1, N_PEOPLE + 1)
    who = np.random.default_rng(seed).choice(N_PEOPLE, size=n_bio, p=w / w.sum())
    bios = [render_bio(pop[i], rng) for i in who]
    docs = stories + bios
    rng.shuffle(docs)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(docs[:200_000] + bios[:20_000], trainer)
    tok.save(str(OUT / "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")

    def encode(texts, path):
        ids = []
        for i in range(0, len(texts), 10_000):
            for e in tok.encode_batch(texts[i : i + 10_000]):
                ids.extend(e.ids)
                ids.append(eot)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(path)
        return len(arr)

    n_train = encode(docs, OUT / "pretrain.bin")
    n_val = encode(val, OUT / "val.bin")

    # Fine-tuning task: balanced BadEnding yes/no on stories from files pretraining never read.
    task = [(s, "BadEnding" in f) for s, f in load_stories(task_files)]
    pos = [x for x in task if x[1]]
    neg = [x for x in task if not x[1]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = len(pos)
    rows = [{"story": s, "label": bool(y)} for s, y in pos + neg[:k]]
    rng.shuffle(rows)
    holdout, train = rows[:1000], rows[1000:]
    (OUT / "task_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (OUT / "task_holdout.jsonl").write_text("".join(json.dumps(r) + "\n" for r in holdout))
    meta = {"pretrain_tokens": n_train, "val_tokens": n_val, "stories": len(stories), "bios": len(bios),
            "task_train": len(train), "task_holdout": len(holdout), "vocab": tok.get_vocab_size()}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    build()
