"""Fine-tuning parametrisations with exact access to dW, the change from the pretrained weights.

lora    dW = s * B A, B=0 at init and A random (Kaiming): the standard adapter
pissa   A, B start at the top-r singular triplets of W0, whose product is subtracted from the frozen base
        (Meng et al. 2024), so dW = s * (B A - B0 A0) starts at exactly 0
milora  the same with the bottom-r singular triplets (Wang et al. 2024)
olora   the same with an orthonormal QR factor of W0: B0 = Q[:, :r], A0 = R[:r] / s (Buyukakyuz 2024)
gpm     lora whose input directions (rows of A) are kept orthogonal to the subspace that pretraining activations
        occupy at that layer, so dW x = 0 on pretraining-like inputs (the Gradient Projection Memory idea,
        Saha et al. 2021, applied to the pretraining distribution; pass `basis` from Context.activation_basis)
pissa-gpm  pissa conditioned on gpm: A, B start at the top-r singular triplets of W0 restricted to gpm's allowed
        inputs (W0^T (I - U U^T)), and A stays there, so dW x = 0 on pretraining-like inputs as for gpm
pissa-gpm-norm  pissa-gpm renormalized after constraining: the same directions, scaled to W0's own top-r singular
        values, so the start (and the first gradients) are as large as unconstrained pissa's
full    every adapted matrix is trained directly; dW = W - W0
All matrices use the GPT-2 Conv1D layout [in, out], so dW_m has the shape of W0_m.
"""

import math

import torch

from .model import targets


class Adapter:
    def __init__(self, model, method="lora", rank=8, alpha=None, seed=0, basis=None):
        self.model, self.method, self.rank = model, method, rank
        self.scale = (alpha or rank) / rank
        self.mods = targets(model)
        self.W0 = {n: m.weight.detach().clone() for n, m in self.mods.items()}
        for p in model.parameters():
            p.requires_grad_(False)
        self.A, self.B, self.A0, self.B0, self.hooks, self.proj = {}, {}, {}, {}, [], None
        if method == "full":
            for m in self.mods.values():
                m.weight.requires_grad_(True)
            return
        g = torch.Generator(device="cpu").manual_seed(seed)
        if method in ("gpm", "pissa-gpm", "pissa-gpm-norm"):  # I - U U^T on the input side of each matrix
            self.proj = {n: torch.eye(U.shape[0], device=U.device) - U @ U.T for n, U in basis.items()}
        for n, m in self.mods.items():
            nin, nout = m.weight.shape
            if method in ("lora", "gpm"):
                A = torch.empty(rank, nin)
                torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
                B = torch.zeros(nout, rank)
            elif method == "olora":
                Q, R = torch.linalg.qr(m.weight.detach().T.cpu().double())
                A, B = (R[:rank] / self.scale).float(), Q[:, :rank].float()
                with torch.no_grad():
                    m.weight -= (self.scale * (B @ A).T).to(m.weight)
            else:
                W = m.weight.detach().T
                if method.startswith("pissa-gpm"):
                    W = W @ self.proj[n]
                U, S, Vh = torch.linalg.svd(W.cpu().double(), full_matrices=False)
                k = slice(len(S) - rank, len(S)) if method == "milora" else slice(0, rank)
                if method == "pissa-gpm-norm":  # renormalize after constraining: W0's own top-r singular values
                    S = torch.linalg.svdvals(m.weight.detach().T.cpu().double())
                r = (S[k] / self.scale).sqrt()
                A, B = (r[:, None] * Vh[k]).float(), (U[:, k] * r).float()
                with torch.no_grad():  # frozen residual: W0 - s (B0 A0)^T
                    m.weight -= (self.scale * (B @ A).T).to(m.weight)
            dev = m.weight.device
            self.A0[n], self.B0[n] = A.to(dev), B.to(dev)
            self.A[n] = torch.nn.Parameter(A.clone().to(dev))
            self.B[n] = torch.nn.Parameter(B.clone().to(dev))
            self.hooks.append(m.register_forward_hook(self._hook(n)))
        if method == "gpm":
            self.project()

    def _hook(self, n):
        def fn(mod, inp, out):
            x = inp[0]
            return out + self.scale * (x @ self.A[n].T) @ self.B[n].T

        return fn

    def params(self):
        if self.method == "full":
            return [m.weight for m in self.mods.values()]
        return [p for n in self.mods for p in (self.A[n], self.B[n])]

    def delta(self, n):
        if self.method == "full":
            return self.mods[n].weight - self.W0[n]
        d = self.scale * (self.B[n] @ self.A[n]).T
        if self.method not in ("lora", "gpm"):
            d = d - self.scale * (self.B0[n] @ self.A0[n]).T
        return d

    def deltas(self):
        return {n: self.delta(n) for n in self.mods}

    @torch.no_grad()
    def state(self):
        return [p.detach().clone() for p in self.params()]

    @torch.no_grad()
    def load(self, state):
        for p, v in zip(self.params(), state):
            p.copy_(v)

    @torch.no_grad()
    def deltas_of(self, state):
        """dW for a saved parameter state (e.g. an EMA) without touching the live weights."""
        live = self.state()
        self.load(state)
        out = {n: d.detach().clone() for n, d in self.deltas().items()}
        self.load(live)
        return out

    @torch.no_grad()
    def project(self):
        if self.proj:
            for n, P in self.proj.items():
                self.A[n].copy_(self.A[n] @ P)

    @torch.no_grad()
    def shrink(self, c):
        """dW <- c dW for every matrix (decay toward the base)."""
        for n in self.mods:
            if self.method == "full":
                self.mods[n].weight.copy_(self.W0[n] + c * (self.mods[n].weight - self.W0[n]))
            elif self.method in ("lora", "gpm"):
                self.B[n].mul_(c)
            else:
                raise ValueError("decay toward the base is defined for lora, gpm and full fine-tuning only")

    @torch.no_grad()
    def cap(self, rho):
        """Radial projection of each dW_m onto ||dW_m|| <= rho ||W0_m|| (scales B, or W - W0 for full)."""
        for n in self.mods:
            norm, lim = float(self.delta(n).norm()), rho * float(self.W0[n].norm())
            if norm <= lim:
                continue
            c = lim / norm
            if self.method == "full":
                self.mods[n].weight.copy_(self.W0[n] + c * (self.mods[n].weight - self.W0[n]))
            elif self.method in ("lora", "gpm"):
                self.B[n].mul_(c)
            else:
                raise ValueError("the radial cap is defined for lora and full fine-tuning only")

    def w0_sq(self):
        return sum(float(w.square().sum()) for w in self.W0.values())


def rho(deltas, w0_sq):
    """Global relative change ||dW||_F / ||W0||_F over the adapted matrices."""
    return math.sqrt(sum(float(d.square().sum()) for d in deltas.values()) / w0_sq)
