"""Attention blocks of the energy network.

The training loss is differentiated through dE/dz, so the graded microsteps need attention with a second derivative
(``math``, or the ``fused`` Triton kernel). Microsteps without a graph may use any backend (``sdpa_auto``). A
thread-local flag (``second_order_window``) tells the attention which of the two it is running.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.layers import RMS_NORM_EPS, CastedLinear, RopeFused, SwiGLU, apply_rope, rms_norm

_SECOND_ORDER = threading.local()


def second_order_required() -> bool:
    return getattr(_SECOND_ORDER, "active", False)


@contextmanager
def second_order_window(active: bool) -> Iterator[None]:
    """Mark the attention calls inside as graded (``active``): they must support a second derivative."""
    previous = getattr(_SECOND_ORDER, "active", False)
    _SECOND_ORDER.active = bool(active)
    try:
        yield
    finally:
        _SECOND_ORDER.active = previous


class SecondOrderUnsafeError(RuntimeError):
    """A backend without a correct second derivative reached a graded microstep."""


# ----------------------------------------------------------------------------
# Backends. q, k, v and the output are (B, heads, N, D).
# ----------------------------------------------------------------------------


def math_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Plain softmax attention, differentiable to any order. Half-precision inputs get a float32 softmax."""
    out_dtype = q.dtype
    compute_dtype = torch.float32 if out_dtype in (torch.bfloat16, torch.float16) else out_dtype
    scores = torch.matmul(q.to(compute_dtype), k.to(compute_dtype).transpose(-1, -2)) * q.shape[-1] ** -0.5
    return torch.matmul(scores.softmax(dim=-1), v.to(compute_dtype)).to(out_dtype)


def _sdpa_auto(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


def _fused(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fused second-order attention (``models.fused_attn``): Triton on CUDA, dense reference on CPU."""
    from models.fused_attn import fused_attention

    return fused_attention(q, k, v, scale=q.shape[-1] ** -0.5)


@dataclass(frozen=True)
class AttentionBackend:
    name: str
    fn: Callable[..., torch.Tensor]
    second_order: bool


ATTENTION_BACKENDS: dict[str, AttentionBackend] = {
    "math": AttentionBackend("math", math_attention, True),
    "sdpa_auto": AttentionBackend("sdpa_auto", _sdpa_auto, False),  # stock SDPA, no double backward
    "fused": AttentionBackend("fused", _fused, True),
}


def get_backend(name: str) -> AttentionBackend:
    try:
        return ATTENTION_BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown attention backend '{name}'; available: {sorted(ATTENTION_BACKENDS)}") from None


def verify_second_order(
    backend: AttentionBackend,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    heads: int = 2,
    seq: int = 32,
    head_dim: int = 16,
    rtol: float = 5e-2,
    seed: int = 0,
) -> None:
    """Check numerically that ``backend`` has the same second derivatives as ``math_attention``, or raise.

    The probe has no residual path around the attention, so a dropped second-order term cannot hide. It reseeds the
    global RNG with ``seed`` and draws its inputs from it. By default the trainer runs it once before training under a
    forked RNG (``utils.checkpoint.prime_attention_self_test``); with ``attention_self_test=first_step`` on one GPU it
    runs at the first graded microstep with the run's seed, and training continues from that reseeded RNG.
    """
    torch.manual_seed(seed)
    b = 2
    d = heads * head_dim
    gen = torch.Generator(device="cpu").manual_seed(11)
    wq, wk, wv = (
        (torch.randn(d, d, generator=gen) * d**-0.5).to(device=device, dtype=dtype).requires_grad_(True)
        for _ in range(3)
    )
    wo = (torch.randn(d, 1, generator=gen) * d**-0.5).to(device=device, dtype=dtype).requires_grad_(True)
    params = [wq, wk, wv, wo]
    z0 = torch.randn(b, seq, d, device=device, dtype=dtype) * 0.5
    u = torch.randn_like(z0)

    def split(x: torch.Tensor) -> torch.Tensor:
        return x.view(b, seq, heads, head_dim).transpose(1, 2).contiguous()

    def energy(z: torch.Tensor, fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        y = fn(split(z @ wq), split(z @ wk), split(z @ wv))
        return (y.transpose(1, 2).reshape(b, seq, d) @ wo).float().sum()

    def probe(fn: Callable[..., torch.Tensor]):
        z = z0.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(energy(z, fn), z, create_graph=True)[0]
        s = (g * u).sum()
        hz = torch.autograd.grad(s, z, retain_graph=True, allow_unused=True)[0]
        ht = torch.autograd.grad(s, params, allow_unused=True)
        return g.detach().float(), hz, ht

    ref_g, ref_hz, ref_ht = probe(math_attention)
    try:
        got_g, got_hz, got_ht = probe(backend.fn)
    except Exception as exc:
        raise SecondOrderUnsafeError(
            f"attention backend '{backend.name}' failed the second-order self-test: {type(exc).__name__}: {exc}"
        ) from exc

    missing = [
        label
        for label, ref, got in (
            ("d2E/dz2", ref_hz, got_hz),
            *((f"d2E/dtheta dz [param {i}]", r, g) for i, (r, g) in enumerate(zip(ref_ht, got_ht))),
        )
        if ref is not None and got is None
    ]
    if missing:
        raise SecondOrderUnsafeError(
            f"attention backend '{backend.name}' produces NO second-order gradient for: " + ", ".join(missing)
        )

    def rel(a: torch.Tensor | None, ref: torch.Tensor | None) -> float:
        if ref is None:
            return 0.0
        if a is None:
            return float("inf")
        return ((a.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12)).item()

    errs = {
        "dE/dz": rel(got_g, ref_g),
        "d2E/dz2": rel(got_hz, ref_hz),
        "d2E/dtheta dz": max(rel(a, r) for a, r in zip(got_ht, ref_ht)),
    }
    if not all(v < rtol for v in errs.values()):
        raise SecondOrderUnsafeError(
            f"attention backend '{backend.name}' does NOT carry second order correctly: "
            + ", ".join(f"{k}={v:.3e}" for k, v in errs.items())
            + f" (tolerance {rtol})"
        )


# ----------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------


class EnergyAttention(nn.Module):
    """Multi-head attention with 2D RoPE, a learned per-head temperature, and a backend per tier (graded or not)."""

    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, *, backend: str, nograd_backend: str) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.output_size = head_dim * num_heads
        self.qkv_proj = CastedLinear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)
        # Log of the temperature, 0 at init: the standard 1 / sqrt(head_dim) scaling.
        self.log_logit_scale = nn.Parameter(torch.zeros(num_heads))
        self.graded_backend = get_backend(backend)
        if not self.graded_backend.second_order:
            raise SecondOrderUnsafeError(f"attention backend '{backend}' cannot be used in graded microsteps")
        self.nograd_backend = get_backend(nograd_backend)
        self._verified = False
        self.self_test_seed = 0  # the trainer sets the run's seed (attention_self_test=first_step)

    def _ensure_verified(self, device: torch.device, dtype: torch.dtype) -> None:
        """Run the second-order self-test once, at the first graded forward."""
        if self._verified:
            return
        verify_second_order(
            self.graded_backend,
            device=device,
            dtype=torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype,
            heads=min(2, self.num_heads),
            head_dim=min(16, self.head_dim),
            seed=self.self_test_seed,
        )
        self._verified = True

    def forward(self, cos_sin: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        b, n, _ = hidden_states.shape
        h, d = self.num_heads, self.head_dim
        graded = second_order_required()
        if graded:
            self._ensure_verified(hidden_states.device, hidden_states.dtype)
        backend = self.graded_backend if graded else self.nograd_backend

        temperature_applied = False
        if isinstance(cos_sin, RopeFused):
            # rope_impl=fused: split + 2D RoPE + head-major layout in one kernel. Without a graph the kernel also
            # applies the temperature; graded microsteps multiply below so log_logit_scale gets its gradient.
            from models.fused_attn.rope_split import rope_split

            qscale = None
            if not graded:
                qscale = self.log_logit_scale.detach().exp().to(hidden_states.dtype)
                temperature_applied = True
            q, k, v = rope_split(self.qkv_proj(hidden_states), cos_sin.cos, cos_sin.sin, h, qscale)
        else:
            q, k, v = self.qkv_proj(hidden_states).split([h * d, h * d, h * d], dim=-1)
            q, k, v = q.view(b, n, h, d), k.view(b, n, h, d), v.view(b, n, h, d)
            q, k = apply_rope(cos_sin, q, k)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if not temperature_applied:
            q = q * self.log_logit_scale.exp().to(q.dtype).view(1, -1, 1, 1)
        y = backend.fn(q, k, v)
        return self.o_proj(y.transpose(1, 2).reshape(b, n, self.output_size))

    def extra_repr(self) -> str:
        return f"heads={self.num_heads}, head_dim={self.head_dim}, graded={self.graded_backend.name}, nograd={self.nograd_backend.name}"


class EnergyReasoningBlock(nn.Module):
    """Attention, then SwiGLU, each with a residual connection followed by rms_norm (post-norm)."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError(f"hidden_size {config.hidden_size} is not divisible by num_heads {config.num_heads}")
        self.self_attn = EnergyAttention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            backend=config.attn_backend,
            nograd_backend=config.attn_nograd_backend,
        )
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)

    def forward(self, cos_sin: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), variance_epsilon=RMS_NORM_EPS)
        return rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=RMS_NORM_EPS)


__all__ = [
    "ATTENTION_BACKENDS",
    "AttentionBackend",
    "EnergyAttention",
    "EnergyReasoningBlock",
    "SecondOrderUnsafeError",
    "get_backend",
    "math_attention",
    "second_order_required",
    "second_order_window",
    "verify_second_order",
]
