"""Fused 2D-RoPE split / merge of the qkv projection (``rope_impl: fused``), with exact autograd links."""

# No ``from __future__ import annotations``: torch 2.6's custom_op schema inference needs real type objects.
from typing import Optional, Tuple

import torch

from models.fused_attn.loader import triton_module
from models.layers import quarter_swap

Tensor = torch.Tensor


def _triton():
    return triton_module("rope_split_triton", "RoPE split kernels")


def _kernel_ok(x: Tensor) -> bool:
    return x.is_cuda and x.dtype in (torch.bfloat16, torch.float16, torch.float32) and _triton() is not None


# ----------------------------------------------------------------------------------------------------------------------
# dense reference
# ----------------------------------------------------------------------------------------------------------------------


def _ct(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def ref_split(
    qkv: Tensor, cos: Tensor, sin: Tensor, heads: int, qscale: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    b, n, c3 = qkv.shape
    d = c3 // (3 * heads)
    ct = _ct(qkv)
    x = qkv.view(b, n, 3, heads, d)
    c, s = cos.to(ct)[None, :, None, :], sin.to(ct)[None, :, None, :]

    def rot(t):
        t = t.to(ct)
        return (t * c + quarter_swap(t) * s).to(qkv.dtype)

    q, k, v = rot(x[:, :, 0]), rot(x[:, :, 1]), x[:, :, 2]
    if qscale is not None and qscale.numel() > 0:
        q = (q.to(ct) * qscale.to(ct).view(1, 1, -1, 1)).to(qkv.dtype)
    return (q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())


def ref_merge(dq: Tensor, dk: Tensor, dv: Tensor, cos: Tensor, sin: Tensor, qscale: Optional[Tensor] = None) -> Tensor:
    b, h, n, d = dq.shape
    ct = _ct(dq)
    c, s = cos.to(ct)[None, None], sin.to(ct)[None, None]

    def adj(g):
        g = g.to(ct)
        return (g * c + quarter_swap(g * s)).to(dq.dtype)

    if qscale is not None and qscale.numel() > 0:
        dq = (dq.to(ct) * qscale.to(ct).view(1, -1, 1, 1)).to(dq.dtype)

    out = torch.stack([adj(dq), adj(dk), dv.to(dq.dtype)], dim=2)  # [B, H, 3, N, D]
    return out.permute(0, 3, 2, 1, 4).reshape(b, n, 3 * h * d)


# ----------------------------------------------------------------------------------------------------------------------
# custom ops (opaque to dynamo / inductor; fake kernels give shapes)
# ----------------------------------------------------------------------------------------------------------------------


@torch.library.custom_op("ebr_fused::rope_split", mutates_args=())
def rope_split_op(qkv: Tensor, cos: Tensor, sin: Tensor, qscale: Tensor, heads: int) -> Tuple[Tensor, Tensor, Tensor]:
    if _kernel_ok(qkv):
        return _triton().rope_split(qkv, cos, sin, heads, qscale)
    return ref_split(qkv, cos, sin, heads, qscale)


@rope_split_op.register_fake
def _(qkv, cos, sin, qscale, heads):
    b, n, c3 = qkv.shape
    d = c3 // (3 * heads)
    return tuple(qkv.new_empty((b, heads, n, d)) for _ in range(3))


@torch.library.custom_op("ebr_fused::rope_merge", mutates_args=())
def rope_merge_op(dq: Tensor, dk: Tensor, dv: Tensor, cos: Tensor, sin: Tensor, qscale: Tensor) -> Tensor:
    dk, dv = dk.to(dq.dtype), dv.to(dq.dtype)
    if _kernel_ok(dq):
        return _triton().rope_merge(dq, dk, dv, cos, sin, qscale)
    return ref_merge(dq, dk, dv, cos, sin, qscale)


@rope_merge_op.register_fake
def _(dq, dk, dv, cos, sin, qscale):
    b, h, n, d = dq.shape
    return dq.new_empty((b, n, 3 * h * d))


# ----------------------------------------------------------------------------------------------------------------------
# autograd: split and merge are each other's adjoint (exact to every order)
# ----------------------------------------------------------------------------------------------------------------------


class _Split(torch.autograd.Function):
    @staticmethod
    def forward(qkv, cos, sin, qscale, heads):
        return torch.ops.ebr_fused.rope_split(qkv, cos, sin, qscale, heads)

    @staticmethod
    def setup_context(ctx, inputs, output):
        qkv, cos, sin, qscale, heads = inputs
        ctx.save_for_backward(cos, sin, qscale)
        ctx.heads = heads
        ctx.shape = tuple(output[0].shape)

    @staticmethod
    def backward(ctx, gq, gk, gv):
        cos, sin, qscale = ctx.saved_tensors
        z = None
        if gq is None or gk is None or gv is None:
            ref = next(g for g in (gq, gk, gv) if g is not None)
            z = torch.zeros(ctx.shape, dtype=ref.dtype, device=ref.device)
        gq = z if gq is None else gq
        gk = z if gk is None else gk
        gv = z if gv is None else gv
        return _Merge.apply(gq, gk, gv, cos, sin, qscale), None, None, None, None


class _Merge(torch.autograd.Function):
    @staticmethod
    def forward(dq, dk, dv, cos, sin, qscale):
        return torch.ops.ebr_fused.rope_merge(dq, dk, dv, cos, sin, qscale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        dq, dk, dv, cos, sin, qscale = inputs
        ctx.save_for_backward(cos, sin, qscale)
        ctx.heads = int(dq.shape[1])

    @staticmethod
    def backward(ctx, g):
        cos, sin, qscale = ctx.saved_tensors
        gq, gk, gv = _Split.apply(g, cos, sin, qscale, ctx.heads)
        return gq, gk, gv, None, None, None


def _split(qkv: Tensor, cos: Tensor, sin: Tensor, qscale: Tensor, heads: int):
    return _Split.apply(qkv, cos, sin, qscale, heads)


# Keep the call one opaque node in dynamo graphs; AOTAutograd traces the Functions above with functorch (as the fused
# attention does, models.fused_attn.ops), so compiled forward / backward graphs contain the custom ops.
_split = torch._dynamo.allow_in_graph(_split)


def rope_split(
    qkv: Tensor, cos: Tensor, sin: Tensor, heads: int, qscale: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor, Tensor]:
    """``[B, N, 3*H*D]`` -> rotated ``q``, ``k`` and ``v``, each ``[B, H, N, D]``; differentiable to every order.

    ``qscale`` (optional, ``[H]``, no gradient): per-head temperature folded into q exactly as EnergyAttention applies it
    after RoPE (``q * exp(log_scale).to(q.dtype)``: pass that rounded value); then ``q = round(round(R q) * t)``, and the
    adjoint scales first. Only for the no-grad tier (the graded tier needs the temperature's gradient).
    """
    if qkv.dim() != 3 or qkv.shape[-1] % (3 * heads):
        raise ValueError(f"rope_split expects [B, N, 3*H*D], got {tuple(qkv.shape)} for {heads} heads")
    d = qkv.shape[-1] // (3 * heads)
    if d % 4 or tuple(cos.shape) != (qkv.shape[1], d) or tuple(sin.shape) != (qkv.shape[1], d):
        raise ValueError(f"rope_split: tables must be [N, D] with D % 4 == 0, got {tuple(cos.shape)} (D={d})")
    if qscale is None:
        qscale = qkv.new_zeros((0,), dtype=torch.float32)
    return _split(qkv, cos, sin, qscale.detach(), heads)


__all__ = ["quarter_swap", "ref_merge", "ref_split", "rope_split"]
