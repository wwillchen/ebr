"""Optimizers: Muon for the hidden weight matrices, AdamW for embeddings, heads and vectors; linear LR warmup;
optional gradient clipping (relative to a running average, or adaptive to the recent norm distribution)."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # quintic Newton-Schulz iteration
NS_EPS = 1e-7
ADAMW_EPS = 1e-8
# Parameters whose name contains one of these go to AdamW (as do all non-matrix parameters).
ADAMW_PATTERNS = ("embed", "lm_head", "energy_head.output")
# Fused projections: name pattern -> the number of logical matrices stacked along the rows (muon_split_fused).
FUSED_BLOCKS = {"qkv_proj": 3, "gate_up_proj": 2}


def zeropower_via_newton_schulz(update: Tensor, ns_steps: int = 5) -> Tensor:
    """An approximate orthogonalization of the 2D ``update`` (U V^T of its SVD) by ``ns_steps`` Newton-Schulz steps."""
    if update.ndim != 2:
        raise ValueError(f"Muon requires a 2D update, got shape {tuple(update.shape)}")
    a, b, c = NS_COEFFICIENTS
    work_dtype = torch.bfloat16 if update.device.type == "cuda" else torch.float32
    x = update.to(dtype=work_dtype)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=NS_EPS)
    for _ in range(ns_steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Nesterov momentum, Newton-Schulz orthogonalization, decoupled weight decay. The step of each matrix is scaled
    by 0.2 * sqrt(max(rows, cols)) to match the update size of AdamW."""

    def __init__(self, params, lr: float, weight_decay: float, momentum: float = 0.95, ns_steps: int = 5) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Muon only supports 2D parameters, found shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
                buf.lerp_(g, 1 - momentum)
                # blocks > 1 (muon_split_fused): the matrix stacks that many logical matrices along its rows (a fused
                # q/k/v or SwiGLU gate/up projection); each is orthogonalized and scaled as if stored on its own.
                blocks = int(group.get("blocks", 1))
                nesterov = g.lerp(buf, momentum)
                parts = nesterov.chunk(blocks, dim=0) if blocks > 1 else (nesterov,)
                updates = [zeropower_via_newton_schulz(u, ns_steps=int(group["ns_steps"])) for u in parts]
                p.mul_(1 - lr * float(group["weight_decay"]))
                for target, update in zip(p.chunk(blocks, dim=0) if blocks > 1 else (p,), updates, strict=True):
                    scale = 0.2 * math.sqrt(max(target.shape[0], target.shape[1]))  # the update size of AdamW
                    target.add_(update.to(dtype=p.dtype), alpha=-lr * scale)
        return loss


def build_optimizers(model: nn.Module, config: Any) -> tuple[list[torch.optim.Optimizer], list[float], str]:
    """[Muon(hidden matrices), AdamW(the rest)], their base learning rates, and a one-line summary.

    With ``config.muon_split_fused`` the fused projections (``FUSED_BLOCKS``) get their own Muon group, which
    orthogonalizes each stacked matrix separately."""
    split = bool(getattr(config, "muon_split_fused", False))
    muon: dict[int, list[nn.Parameter]] = {}
    adamw = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim != 2 or any(pattern in name.lower() for pattern in ADAMW_PATTERNS):
            adamw.append(p)
            continue
        blocks = next((b for pattern, b in FUSED_BLOCKS.items() if pattern in name), 1) if split else 1
        if p.shape[0] % blocks:
            raise ValueError(f"{name}: {p.shape[0]} rows do not split into {blocks} matrices")
        muon.setdefault(blocks, []).append(p)
    if not muon:
        raise ValueError("no hidden 2D weights for Muon")
    wd = config.weight_decay
    groups = [{"params": ps, "weight_decay": wd, **({"blocks": b} if b > 1 else {})} for b, ps in sorted(muon.items())]
    optimizers: list[torch.optim.Optimizer] = [Muon(groups, lr=0, weight_decay=wd)]
    if adamw:
        optimizers.append(
            torch.optim.AdamW(
                [{"params": adamw, "weight_decay": wd}],
                lr=0,
                weight_decay=wd,
                betas=(config.beta1, config.beta2),
                eps=ADAMW_EPS,
            )
        )
    muon_params = [p for ps in muon.values() for p in ps]
    summary = (
        f"Muon: {len(muon_params)} tensors / {sum(p.numel() for p in muon_params):,} params"
        + (f" ({sum(len(ps) for b, ps in muon.items() if b > 1)} split)" if split else "")
        + f"; AdamW: {len(adamw)} tensors / {sum(p.numel() for p in adamw):,} params"
    )
    return optimizers, [config.lr] * len(optimizers), summary


CLIP_HALF_LIFE = 200  # steps: the half-life of RelativeGradClip's running average


class RelativeGradClip:
    """Clip the global gradient norm at ``factor`` times a running average of the recent norms.

    The average is exponential (half-life ``half_life`` steps, default ``CLIP_HALF_LIFE``) and is updated with the
    clipped norm, so a spike does not raise the limit for the next one. It is tracked from the first step; clipping
    starts after ``start_step`` (the LR warmup, whose first steps have norms far above the later ones). Everything stays
    on the device: no host sync.
    """

    def __init__(self, factor: float, *, start_step: int = 0, half_life: float | None = None) -> None:
        if not factor > 1.0:
            raise ValueError(f"the clip factor must be > 1, got {factor}")
        self.factor = float(factor)
        self.start_step = int(start_step)
        self.decay = 0.5 ** (1.0 / (half_life or CLIP_HALF_LIFE))
        self.avg: Tensor | None = None  # running average of the (clipped) global norm
        self.clipped: Tensor | None = None  # steps clipped since the last ``pop_metrics``

    def __call__(self, grads: list[Tensor], step: int) -> None:
        """Clip ``grads`` in place (the gradients of every parameter, after any cross-rank reduction)."""
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, dtype=torch.float32) for g in grads]))
        if self.avg is None:
            self.avg, self.clipped = norm.clone(), torch.zeros_like(norm)
            return
        if step > self.start_step:
            limit = self.factor * self.avg
            scale = (limit / norm.clamp_min(1e-12)).clamp(max=1.0)
            for g in grads:
                g.mul_(scale.to(g.dtype))
            self.clipped += (norm > limit).to(self.clipped.dtype)
            norm = torch.minimum(norm, limit)
        self.avg.lerp_(norm, 1.0 - self.decay)

    def pop_metrics(self) -> dict[str, float]:
        """The current limit and the number of steps clipped since the previous call (a host read)."""
        if self.avg is None:
            return {}
        out = {
            "train/grad_clip/limit": float(self.factor * self.avg),
            "train/grad_clip/clipped_steps": float(self.clipped),
        }
        self.clipped.zero_()
        return out

    def state_dict(self) -> dict[str, Tensor]:
        return {} if self.avg is None else {"avg": self.avg.detach().clone()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if "avg" in state:
            self.avg = state["avg"].detach().clone().to(torch.float32)
            self.clipped = torch.zeros_like(self.avg)


ADAPTIVE_CLIP_WINDOW = 200  # steps: the recent global norms AdaptiveGradClip takes its statistics from
ADAPTIVE_CLIP_MIN_HISTORY = 50  # steps recorded before AdaptiveGradClip starts clipping


class AdaptiveGradClip:
    """Clip the global gradient norm at ``exp(m + z * s)`` of the log norms of the last ``window`` steps.

    ``m`` is their median and ``s = (m - q25) / 0.6745`` a spread taken from the lower half only, so past spikes do not
    raise the limit (a robust z-score on log norms, as in AutoClip / ZClip). Every finite norm enters the window
    unclipped, so the limit follows a real change of gradient scale during training. Clipping starts once
    ``min_history`` norms are recorded; a non-finite norm is neither clipped nor recorded. Everything stays on the
    device: no host sync.
    """

    def __init__(self, z: float, *, window: int | None = None, min_history: int | None = None) -> None:
        window = int(window or ADAPTIVE_CLIP_WINDOW)
        min_history = int(min_history or ADAPTIVE_CLIP_MIN_HISTORY)
        if not z > 0:
            raise ValueError(f"the clip z must be > 0, got {z}")
        if not 2 <= min_history <= window:
            raise ValueError(f"need 2 <= min_history <= window, got {min_history} and {window}")
        self.z, self.window, self.min_history = float(z), window, min_history
        self.buf: Tensor | None = None  # [window] log norms, nan = empty
        self.count = 0  # norms recorded so far
        self.limit: Tensor | None = None
        self.clipped: Tensor | None = None  # steps clipped since the last ``pop_metrics``

    def __call__(self, grads: list[Tensor], step: int) -> None:
        """Clip ``grads`` in place (the gradients of every parameter, after any cross-rank reduction)."""
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, dtype=torch.float32) for g in grads]))
        if self.buf is None:
            self.buf = torch.full((self.window,), float("nan"), device=norm.device)
        if self.clipped is None:
            self.clipped = torch.zeros((), device=norm.device)
        if self.count >= self.min_history:
            m = torch.nanquantile(self.buf, 0.5)
            s = ((m - torch.nanquantile(self.buf, 0.25)) / 0.6745).clamp_min(1e-6)
            self.limit = torch.exp(m + self.z * s)
            finite = torch.isfinite(norm)
            scale = torch.where(finite, (self.limit / norm.clamp_min(1e-12)).clamp(max=1.0), torch.ones_like(norm))
            for g in grads:
                g.mul_(scale.to(g.dtype))
            self.clipped += (finite & (norm > self.limit)).to(self.clipped.dtype)
        log_norm = torch.log(norm)
        i = self.count % self.window
        self.buf[i] = torch.where(torch.isfinite(log_norm), log_norm, self.buf[i])
        self.count += 1

    def pop_metrics(self) -> dict[str, float]:
        """The current limit and the number of steps clipped since the previous call (a host read)."""
        out: dict[str, float] = {}
        if self.limit is not None:
            out["train/grad_clip/limit"] = float(self.limit)
        if self.clipped is not None:
            out["train/grad_clip/clipped_steps"] = float(self.clipped)
            self.clipped.zero_()
        return out

    def state_dict(self) -> dict[str, Tensor]:
        return {} if self.buf is None else {"buf": self.buf.detach().clone(), "count": torch.tensor(self.count)}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        if "buf" in state:
            if state["buf"].numel() != self.window:
                raise ValueError(f"saved clip window {state['buf'].numel()} != {self.window}")
            self.buf = state["buf"].detach().clone().to(torch.float32)
            self.count = int(state["count"])


def lr_with_warmup(
    step: int, *, base_lr: float, warmup_steps: int, total_steps: int | None = None, cooldown_steps: int | None = None
) -> float:
    """Linear warmup to ``base_lr``, then constant; with ``cooldown_steps``, a linear decay to 0 over the last
    ``cooldown_steps`` of ``total_steps``."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if cooldown_steps and total_steps is not None and step > total_steps - cooldown_steps:
        return base_lr * max(total_steps - step, 0) / cooldown_steps
    return base_lr


__all__ = [
    "FUSED_BLOCKS",
    "AdaptiveGradClip",
    "Muon",
    "RelativeGradClip",
    "build_optimizers",
    "lr_with_warmup",
    "zeropower_via_newton_schulz",
]
