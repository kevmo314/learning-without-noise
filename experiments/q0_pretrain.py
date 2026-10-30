"""Q0: pretrain the base model on TinyStories + Zipf-exposed biographies, so its knowledge is partly learned."""

from lwn.train import pretrain

if __name__ == "__main__":
    pretrain()
