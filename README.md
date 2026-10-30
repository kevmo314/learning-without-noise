# learning-without-noise

When you fine-tune a pretrained model on narrow data, it forgets some of what it knew. This repo tests a simple
explanation at small scale: **most of that forgetting is caused by optimizer noise, not by what the model is
being taught.** If that is right, the useful levers are the ones that remove noise (averaging, lower or decaying
learning rates, bigger batches) or keep it somewhere harmless (caps, decay toward the base, subspace constraints).

The idea comes from LoRA fine-tuning of Qwen3.5 (2B to 9B). There, about 97% of the adapter's change from the base
weights differed between random seeds, and the part both seeds agreed on did not hurt general knowledge at a
sensible learning rate. Here we rebuild the setting with a model small enough to run hundreds of ablations.

## The setup

- **Base model:** a 7.4M-parameter GPT-2 (Hugging Face `transformers`), pretrained from scratch on
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) mixed with synthetic biographies of 10,000
  people. The biographies follow Allen-Zhu & Li's bioS design: six attributes per person, several templates per
  attribute, shuffled sentences. People appear with Zipf-distributed frequency, so the model knows common people
  well and rare people partly or not at all. Its knowledge competes for capacity, as a real model's does.
- **Fine-tuning task:** "Does this story have a bad ending?", answered " yes" or " no", on stories the base never
  saw. The labels are TinyStories' own `BadEnding` tags, so they're real and imperfect.
- **What we measure:**
  - *Learning:* held-out task accuracy.
  - *Knowledge retention:* likelihood multiple choice over 8 options for prompts like "Alice B. Smith was born in".
    This is scored the way MMLU is scored with likelihoods, and split by how often each person appeared in
    pretraining.
  - *Language retention:* held-out TinyStories loss.
  - *Distribution shift:* KL from the base model on held-out text.
- **Distance moved:** ρ = ‖ΔW‖ / ‖W₀‖ over the adapted matrices.

## Questions, one entrypoint each

| script | question |
|---|---|
| `experiments/q0_pretrain.py` | Train the base model. |
| `experiments/q1_reproduce.py` | Does the small model reproduce the large-model observations (forgetting vs ρ, EMA, cooldown, post-hoc shrinking)? |
| `experiments/q2_noise_share.py` | How much of forgetting is noise? Splits the update into signal, slow drift and fast jitter across seeds, prices each, and measures the curvature of each direction. |
| `experiments/q3_which_data.py` | Which knowledge does pure noise destroy first? Compares rare and frequent facts with general language. |
| `experiments/q4_mitigations.py` | What mitigates noise, and at what cost to learning? |
| `experiments/q5_measure_noise.py` | Can a single run measure its own noise and predict how far it can safely move? |
| `experiments/q6_subspace.py` | Does the adapter's subspace matter? Compares LoRA, PiSSA, MiLoRA, OLoRA, GPM-style orthogonal and full fine-tuning. |
| `experiments/q7_spikes.py` | Are loss spikes catastrophic for retention even after the loss recovers? |
| `experiments/q8_data_quality.py` | Does the safe distance depend on label quality, and does adaptive weight decay track it? |

```bash
uv sync
uv run python -m lwn.corpus            # downloads TinyStories into data/ and builds the corpus (~10 min)
PYTHONPATH=. uv run python -m experiments.q0_pretrain
PYTHONPATH=. uv run python -m experiments.q1_reproduce   # and so on
```

Fine-tuning runs are cached in `results/runs/` by configuration, so experiments share runs. Each script prints
its tables and writes `results/<name>.json`.

## Findings

Results are added here as each question finishes.
