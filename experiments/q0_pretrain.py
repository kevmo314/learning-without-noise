"""Q0: pretrain the base model on TinyStories + Zipf-exposed biographies, so its knowledge is partly learned.
LWN_MODEL=large trains the 27M-parameter variant (lower peak LR) used to check the findings at a larger scale."""

from lwn.model import MODEL
from lwn.train import pretrain

if __name__ == "__main__":
    pretrain(lr=2e-3 if MODEL == "small" else 1e-3)
