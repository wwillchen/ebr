"""Custom ops for fused second-order attention: attn_fwd -> attn_bwd -> attn_bwd_bwd (Triton on CUDA).

Set ``EBR_FUSED_ATTN_IMPL=triton`` to make a missing kernel an error, ``=reference`` to force the dense reference.
"""

# No ``from __future__ import annotations``: torch 2.6's custom_op schema inference
# needs real type objects, not strings.
from typing import Any, Callable, Optional, Tuple, Union

import torch

from models.fused_attn import reference
from models.fused_attn.loader import triton_module

Tensor = torch.Tensor


def _triton_module() -> Optional[Any]:
    """The Triton launchers, or None when unavailable (and not forced)."""
    return triton_module("triton_kernels", "Triton kernels", ("attn_fwd", "attn_bwd", "attn_bwd_bwd"))


def implementation(device: Union[torch.device, str] = "cuda") -> str:
    """'triton' or 'reference': what the ops dispatch to on ``device``."""
    if torch.device(device).type != "cuda":
        return "reference"
    return "triton" if _triton_module() is not None else "reference"


def _cuda_impl(name: str) -> Callable[..., Any]:
    ref = getattr(reference, name)

    def run(*args: Any) -> Any:
        tk = _triton_module()
        if tk is None:
            return ref(*args)
        kwargs = {}
        # fp32 inputs follow torch's matmul precision: "highest" gives IEEE fp32 dots (the compiled-dE/dz check uses
        # it), otherwise the kernel's TF32 default.
        if args[0].dtype == torch.float32 and torch.get_float32_matmul_precision() == "highest":
            kwargs["input_precision"] = "ieee"
        return getattr(tk, name)(*args, **kwargs)

    return run


# No input copies: the kernels take any batch / head / token strides (unit stride in D), so the q / k / v views of
# qkv_proj go in as they are.


def _check(q: Tensor, k: Tensor, v: Tensor) -> None:
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"fused attention expects q, k, v of identical shape [B, H, N, D]; got {q.shape}, {k.shape}, {v.shape}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"fused attention expects one dtype; got {q.dtype}, {k.dtype}, {v.dtype}")


# ----------------------------------------------------------------------------
# Ops
# ----------------------------------------------------------------------------


@torch.library.custom_op("ebr_fused::attn_fwd", mutates_args=())
def attn_fwd(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tuple[Tensor, Tensor]:
    _check(q, k, v)
    if q.is_cuda:
        return _cuda_impl("attn_fwd")(q, k, v, scale)
    return reference.attn_fwd(q, k, v, scale)


@attn_fwd.register_fake
def _(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tuple[Tensor, Tensor]:
    o = torch.empty_like(q, memory_format=torch.contiguous_format)
    lse = q.new_empty(q.shape[:-1], dtype=reference.lse_dtype_for(q))
    return o, lse


@torch.library.custom_op("ebr_fused::attn_bwd", mutates_args=())
def attn_bwd(
    q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, do: Tensor, scale: float
) -> Tuple[Tensor, Tensor, Tensor]:
    do = do.to(q.dtype)
    if q.is_cuda:
        return _cuda_impl("attn_bwd")(q, k, v, o, lse, do, scale)
    return reference.attn_bwd(q, k, v, o, lse, do, scale)


@attn_bwd.register_fake
def _(
    q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, do: Tensor, scale: float
) -> Tuple[Tensor, Tensor, Tensor]:
    f = lambda x: torch.empty_like(x, memory_format=torch.contiguous_format)
    return f(q), f(k), f(v)


@torch.library.custom_op("ebr_fused::attn_bwd_bwd", mutates_args=())
def attn_bwd_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    lse: Tensor,
    do: Tensor,
    dq: Tensor,
    gq: Tensor,
    gk: Tensor,
    gv: Tensor,
    scale: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if q.is_cuda:
        return _cuda_impl("attn_bwd_bwd")(q, k, v, o, lse, do, dq, gq, gk, gv, scale)
    return reference.attn_bwd_bwd(q, k, v, o, lse, do, dq, gq, gk, gv, scale)


@attn_bwd_bwd.register_fake
def _(q, k, v, o, lse, do, dq, gq, gk, gv, scale):
    f = lambda x: torch.empty_like(x, memory_format=torch.contiguous_format)
    return f(q), f(k), f(v), f(do)


# ----------------------------------------------------------------------------
# Autograd: fwd -> bwd -> bwd_bwd
# ----------------------------------------------------------------------------
#
# autograd.Functions rather than register_autograd on the ops: torch.func.grad (used by energy_compile) needs
# Functions with setup_context. The raw ops carry no autograd formula, so only fused_attention is differentiable.


class _FusedAttnBwd(torch.autograd.Function):
    """(q, k, v, o, lse, do) -> (dq, dk, dv); its backward is attn_bwd_bwd."""

    @staticmethod
    def forward(q, k, v, o, lse, do, scale):
        return torch.ops.ebr_fused.attn_bwd(q, k, v, o, lse, do, scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, o, lse, do, scale = inputs
        ctx.save_for_backward(q, k, v, o, lse, do, output[0])
        ctx.scale = scale

    @staticmethod
    def backward(ctx, gq, gk, gv):
        q, k, v, o, lse, do, dq = ctx.saved_tensors
        gq = torch.zeros_like(q) if gq is None else gq.to(q.dtype)
        gk = torch.zeros_like(k) if gk is None else gk.to(k.dtype)
        gv = torch.zeros_like(v) if gv is None else gv.to(v.dtype)
        hq, hk, hv, hdo = torch.ops.ebr_fused.attn_bwd_bwd(q, k, v, o, lse, do, dq, gq, gk, gv, ctx.scale)
        return hq, hk, hv, None, None, hdo, None


class _FusedAttn(torch.autograd.Function):
    """(q, k, v) -> o; its backward is _FusedAttnBwd (itself differentiable)."""

    @staticmethod
    def forward(q, k, v, scale):
        o, lse = torch.ops.ebr_fused.attn_fwd(q, k, v, scale)
        return o, lse

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, scale = inputs
        o, lse = output
        ctx.mark_non_differentiable(lse)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale

    @staticmethod
    def backward(ctx, do, dlse):
        del dlse  # lse is marked non-differentiable
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _FusedAttnBwd.apply(q, k, v, o, lse, do, ctx.scale)
        return dq, dk, dv, None


def _fused_attention_apply(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
    o, _ = _FusedAttn.apply(q, k, v, scale)
    return o


# Keep the call one opaque node for dynamo: inlined, its backward would not be differentiable under torch.func.grad.
# AOTAutograd then traces the Functions above, so the graphs hold attn_fwd / attn_bwd / attn_bwd_bwd as custom ops.
_fused_attention_apply = torch._dynamo.allow_in_graph(_fused_attention_apply)


def fused_attention(q: Tensor, k: Tensor, v: Tensor, scale: Optional[float] = None) -> Tensor:
    """Non-causal softmax attention, ``[B, H, N, D]`` in and out, differentiable to second order."""
    return _fused_attention_apply(q, k, v, reference.default_scale(q, scale))


__all__ = ["attn_bwd", "attn_bwd_bwd", "attn_fwd", "fused_attention", "implementation"]
