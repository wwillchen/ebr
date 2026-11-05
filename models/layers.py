"""Basic layers: casted linear and embedding, rms_norm, SwiGLU and 2D rotary position embeddings."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

RMS_NORM_EPS = 1e-5
ROPE_THETA = 10000.0


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    """Fill ``tensor`` in place with a normal truncated to [lower, upper] standard deviations, rescaled so that the
    result has standard deviation ``std`` (as JAX's truncated normal initializer)."""
    with torch.no_grad():
        if std == 0:
            return tensor.zero_()
        sqrt2 = math.sqrt(2.0)
        a = math.erf(lower / sqrt2)
        b = math.erf(upper / sqrt2)
        z = (b - a) / 2.0
        c = (2.0 * math.pi) ** -0.5
        pdf_u = c * math.exp(-0.5 * lower**2)
        pdf_l = c * math.exp(-0.5 * upper**2)
        comp_std = std / math.sqrt(1.0 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
        tensor.uniform_(a, b)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clip_(lower * comp_std, upper * comp_std)
        return tensor


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Normalize the last dimension to unit RMS (computed in float32, returned in the input dtype)."""
    dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + variance_epsilon)
    return hidden_states.to(dtype)


class CastedLinear(nn.Module):
    """Linear layer with float32 weights, cast to the input dtype at use."""

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5))
        )
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, self.weight.to(input.dtype), bias=bias)


class CastedEmbedding(nn.Module):
    """Embedding with float32 weights, cast to ``cast_to`` at use."""

    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype) -> None:
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float) -> None:
        super().__init__()
        intermediate = round(expansion * hidden_size * 2 / 3)
        intermediate = -(intermediate // -256) * 256  # round up to a multiple of 256
        self.gate_up_proj = CastedLinear(hidden_size, intermediate * 2, bias=False)
        self.down_proj = CastedLinear(intermediate, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ----------------------------------------------------------------------------
# 2D rotary position embeddings: the first half of each head's channels rotates with the cell's row, the second half
# with its column.
# ----------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding2D(nn.Module):
    """cos / sin tables of the row and column rotations of an ``height x width`` board."""

    def __init__(self, dim: int, height: int, width: int, base: float = ROPE_THETA) -> None:
        super().__init__()
        half = dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
        rows = torch.outer(torch.arange(height, dtype=torch.float32), inv_freq)
        cols = torch.outer(torch.arange(width, dtype=torch.float32), inv_freq)
        rows, cols = torch.cat((rows, rows), dim=-1), torch.cat((cols, cols), dim=-1)
        self.cos_cached_height = nn.Buffer(rows.cos(), persistent=False)
        self.sin_cached_height = nn.Buffer(rows.sin(), persistent=False)
        self.cos_cached_width = nn.Buffer(cols.cos(), persistent=False)
        self.sin_cached_width = nn.Buffer(cols.sin(), persistent=False)

    def forward(self) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return (self.cos_cached_height, self.cos_cached_width), (self.sin_cached_height, self.sin_cached_width)


def apply_rope(cos_sin, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2D RoPE on q and k of shape (B, N, heads, D), N = height x width, from :class:`RotaryEmbedding2D` tables."""
    (cos_h, cos_w), (sin_h, sin_w) = cos_sin
    height, width = cos_h.shape[0], cos_w.shape[0]
    cos_h, sin_h = cos_h[None, :, None, None, :], sin_h[None, :, None, None, :]
    cos_w, sin_w = cos_w[None, None, :, None, :], sin_w[None, None, :, None, :]

    def rotate(x: torch.Tensor) -> torch.Tensor:
        b, n, heads, d = x.shape
        grid = x.reshape(b, height, width, heads, d).to(cos_h.dtype)
        by_row, by_col = grid[..., : d // 2], grid[..., d // 2 :]
        by_row = by_row * cos_h + rotate_half(by_row) * sin_h
        by_col = by_col * cos_w + rotate_half(by_col) * sin_w
        return torch.cat([by_row, by_col], dim=-1).to(x.dtype).reshape(b, n, heads, d)

    return rotate(q), rotate(k)


class RopeFused(NamedTuple):
    """``rope_impl: fused``: the :class:`RotaryTable2D` tables, applied by the fused split kernel
    (``models.fused_attn.rope_split``)."""

    cos: torch.Tensor
    sin: torch.Tensor


def quarter_swap(x: torch.Tensor) -> torch.Tensor:
    """``rotate_half`` applied to each half of the last dim, without the sign: ``(a, b, c, d) -> (b, a, d, c)``."""
    shape = x.shape
    return x.reshape(*shape[:-1], 2, 2, shape[-1] // 4).flip(-2).reshape(shape)


class RotaryTable2D(nn.Module):
    """:class:`RotaryEmbedding2D` as full ``[height * width, D]`` tables for the fused RoPE split.

    Cell ``i * width + j`` has its row ``i`` frequencies in the first half of D and its column ``j`` frequencies in the
    second. The sign of rotate_half is folded into ``sin``, so ``q' = q * cos + quarter_swap(q) * sin``.
    """

    def __init__(self, dim: int, height: int, width: int, base: float = ROPE_THETA) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError(f"rope_impl=fused needs head_dim divisible by 4, got {dim}")
        (cos_h, cos_w), (sin_h, sin_w) = RotaryEmbedding2D(dim, height, width, base)()
        rows = torch.arange(height).repeat_interleave(width)
        cols = torch.arange(width).repeat(height)
        cos = torch.cat((cos_h[rows], cos_w[cols]), dim=-1)
        sin = torch.cat((sin_h[rows], sin_w[cols]), dim=-1)
        sign = torch.ones(dim)
        sign[: dim // 4] = -1.0
        sign[dim // 2 : dim // 2 + dim // 4] = -1.0
        self.cos_table = nn.Buffer(cos.to(torch.float32), persistent=False)
        self.sin_table = nn.Buffer((sin * sign).to(torch.float32), persistent=False)

    def forward(self) -> RopeFused:
        return RopeFused(self.cos_table, self.sin_table)


__all__ = [
    "RMS_NORM_EPS",
    "CastedEmbedding",
    "CastedLinear",
    "RotaryEmbedding2D",
    "RotaryTable2D",
    "RopeFused",
    "SwiGLU",
    "apply_rope",
    "quarter_swap",
    "rms_norm",
    "trunc_normal_init_",
]
