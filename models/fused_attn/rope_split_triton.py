"""Triton kernels for rope_split / rope_merge."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BT = 32
WARPS = 4

_TL = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


@triton.jit
def _perm(offs_d, Q: tl.constexpr):
    return ((offs_d // Q) ^ 1) * Q + (offs_d % Q)


@triton.jit
def _split_kernel(
    QKV,
    COS,
    SIN,
    QS,
    QO,
    KO,
    VO,
    T,
    N,
    H,
    s_b,
    s_n,
    HAS_QS: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    DT: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    b = (t // N).to(tl.int64)
    n = t % N
    offs_d = tl.arange(0, D)
    perm = _perm(offs_d, D // 4)
    m = tm[:, None]
    row = QKV + b[:, None] * s_b + n[:, None].to(tl.int64) * s_n + h * D
    c = tl.load(COS + n[:, None] * D + offs_d[None, :], mask=m, other=0.0)
    s = tl.load(SIN + n[:, None] * D + offs_d[None, :], mask=m, other=0.0)
    hd = H * D
    q = tl.load(row + offs_d[None, :], mask=m, other=0.0).to(tl.float32)
    qs = tl.load(row + perm[None, :], mask=m, other=0.0).to(tl.float32)
    k = tl.load(row + hd + offs_d[None, :], mask=m, other=0.0).to(tl.float32)
    ks = tl.load(row + hd + perm[None, :], mask=m, other=0.0).to(tl.float32)
    v = tl.load(row + 2 * hd + offs_d[None, :], mask=m, other=0.0)
    out = ((b * H + h) * N + n.to(tl.int64))[:, None] * D + offs_d[None, :]
    qr = (q * c + qs * s).to(DT)
    if HAS_QS:
        # per-head temperature: round(round(R q) * t), the stock bf16 multiply after RoPE
        qr = (qr.to(tl.float32) * tl.load(QS + h)).to(DT)
    tl.store(QO + out, qr, mask=m)
    tl.store(KO + out, (k * c + ks * s).to(DT), mask=m)
    tl.store(VO + out, v, mask=m)


@triton.jit
def _merge_kernel(
    DQ,
    DK,
    DV,
    COS,
    SIN,
    QS,
    G,
    T,
    N,
    H,
    q_b,
    q_h,
    q_n,
    k_b,
    k_h,
    k_n,
    v_b,
    v_h,
    v_n,
    HAS_QS: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    DT: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1).to(tl.int64)
    t = pid * BT + tl.arange(0, BT)
    tm = t < T
    b = (t // N).to(tl.int64)
    n = t % N
    nl = n.to(tl.int64)
    offs_d = tl.arange(0, D)
    perm = _perm(offs_d, D // 4)
    m = tm[:, None]
    c = tl.load(COS + n[:, None] * D + offs_d[None, :], mask=m, other=0.0)
    sp = tl.load(SIN + n[:, None] * D + perm[None, :], mask=m, other=0.0)
    qrow = DQ + (b * q_b + h * q_h + nl * q_n)[:, None]
    krow = DK + (b * k_b + h * k_h + nl * k_n)[:, None]
    vrow = DV + (b * v_b + h * v_h + nl * v_n)[:, None]
    gq = tl.load(qrow + offs_d[None, :], mask=m, other=0.0).to(tl.float32)
    gqs = tl.load(qrow + perm[None, :], mask=m, other=0.0).to(tl.float32)
    if HAS_QS:
        # adjoint of the temperature multiply first (rounded, as the stock bf16 mul backward), then R^T
        tq = tl.load(QS + h)
        gq = (gq * tq).to(DT).to(tl.float32)
        gqs = (gqs * tq).to(DT).to(tl.float32)
    gk = tl.load(krow + offs_d[None, :], mask=m, other=0.0).to(tl.float32)
    gks = tl.load(krow + perm[None, :], mask=m, other=0.0).to(tl.float32)
    gv = tl.load(vrow + offs_d[None, :], mask=m, other=0.0)
    hd = H * D
    out = G + (t.to(tl.int64) * (3 * hd) + h * D)[:, None] + offs_d[None, :]
    tl.store(out, (gq * c + gqs * sp).to(DT), mask=m)
    tl.store(out + hd, (gk * c + gks * sp).to(DT), mask=m)
    tl.store(out + 2 * hd, gv.to(DT), mask=m)


def _tables(cos, sin):
    return cos.to(torch.float32).contiguous(), sin.to(torch.float32).contiguous()


def rope_split(qkv, cos, sin, heads, qscale=None):
    B, N, C3 = qkv.shape
    D = C3 // (3 * heads)
    if qkv.stride(-1) != 1:
        qkv = qkv.contiguous()
    cos, sin = _tables(cos, sin)
    has_qs = qscale is not None and qscale.numel() > 0
    qs = qscale.to(torch.float32).contiguous() if has_qs else cos
    q, k, v = (torch.empty((B, heads, N, D), device=qkv.device, dtype=qkv.dtype) for _ in range(3))
    T = B * N
    _split_kernel[(triton.cdiv(T, BT), heads)](
        qkv,
        cos,
        sin,
        qs,
        q,
        k,
        v,
        T,
        N,
        heads,
        qkv.stride(0),
        qkv.stride(1),
        HAS_QS=has_qs,
        D=D,
        BT=BT,
        DT=_TL[qkv.dtype],
        num_warps=WARPS,
    )
    return q, k, v


def rope_merge(dq, dk, dv, cos, sin, qscale=None):
    B, H, N, D = dq.shape
    dq, dk, dv = (x if x.stride(-1) == 1 else x.contiguous() for x in (dq, dk, dv))
    cos, sin = _tables(cos, sin)
    has_qs = qscale is not None and qscale.numel() > 0
    qs = qscale.to(torch.float32).contiguous() if has_qs else cos
    g = torch.empty((B, N, 3 * H * D), device=dq.device, dtype=dq.dtype)
    T = B * N
    _merge_kernel[(triton.cdiv(T, BT), H)](
        dq,
        dk,
        dv,
        cos,
        sin,
        qs,
        g,
        T,
        N,
        H,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        HAS_QS=has_qs,
        D=D,
        BT=BT,
        DT=_TL[dq.dtype],
        num_warps=WARPS,
    )
    return g


__all__ = ["rope_merge", "rope_split"]
