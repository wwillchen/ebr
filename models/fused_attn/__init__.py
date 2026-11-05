"""Fused second-order softmax attention: ``fused_attention(q, k, v, scale=None)`` on [B, H, N, D] tensors."""

from models.fused_attn.ops import fused_attention, implementation

__all__ = ["fused_attention", "implementation"]
