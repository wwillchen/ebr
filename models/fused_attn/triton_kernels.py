"""Triton kernels: non-causal softmax attention, its backward and the backward of the backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Dot precision for fp32 inputs, overridable per call:
#   "tf32rn" (default) tf32 tensor cores with operands rounded to nearest, like cuBLAS under
#            float32_matmul_precision("high") (what the math backend uses). Error ~= math under TF32.
#   "tf32"   Triton's plain tf32 (operands truncated): about 2x the error of tf32rn, slightly faster.
#   "tf32x3" 3-pass split tf32, ~fp32 accuracy on tensor cores.
#   "ieee"   fp32 FMA dots, for gradcheck and the verifier.
DEFAULT_FP32_PRECISION = "tf32rn"
# For bf16/fp16 inputs: True = tensor-core dots in the input dtype (P, dS cast down), False = upcast to fp32 (tf32).
DEFAULT_LOWP_DOTS = True


# ----------------------------------------------------------------------------------------------------------------------
# jit helpers
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _rna_tf32(x):
    # Round-to-nearest fp32 -> tf32 (as cuBLAS does). Triton's tf32 dot otherwise drops the low 13 mantissa bits
    # (round toward zero), which doubles the error and biases it.
    return tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _mm(a, b, F64: tl.constexpr, PREC: tl.constexpr):
    """a @ b with fp32 (or fp64) result."""
    if F64:
        # Triton 3.2 has no fp64 tl.dot: exact broadcast-and-sum (gradcheck shapes only).
        return tl.sum(a[:, :, None] * b[None, :, :], axis=1)
    elif PREC == "tf32rn":
        if a.dtype == tl.float32:
            return tl.dot(_rna_tf32(a), _rna_tf32(b), input_precision="tf32")
        else:
            return tl.dot(a, b)
    else:
        # PREC is ignored by the backend for bf16/fp16 operands.
        return tl.dot(a, b, input_precision=PREC)


@triton.jit
def _exp(x, F64: tl.constexpr):
    # Scores are kept in natural units. x is always a small difference (score - row max, or score - lse), so folding
    # log2(e) in here rounds an O(1) number; the usual "scale * log2e" fold rounds O(|score|) numbers instead, which
    # shows up as error in P when the logits are large.
    if F64:
        return tl.exp(x)
    else:
        return tl.exp2(x * 1.4426950408889634)


@triton.jit
def _base(ptr, bh, H, sb, sh):
    """Start of the (b, h) slice of a [B, H, N, D] tensor with batch/head strides sb, sh."""
    b = bh // H
    return ptr + b * sb + (bh - b * H) * sh


@triton.jit
def _ld(ptr, offs_r, offs_d, N, D: tl.constexpr, sn):
    return tl.load(
        ptr + offs_r[:, None] * sn + offs_d[None, :],
        mask=(offs_r[:, None] < N) & (offs_d[None, :] < D),
        other=0.0,
    )


@triton.jit
def _st(ptr, val, offs_r, offs_d, N, D: tl.constexpr):
    tl.store(
        ptr + offs_r[:, None] * D + offs_d[None, :],
        val.to(ptr.dtype.element_ty),
        mask=(offs_r[:, None] < N) & (offs_d[None, :] < D),
    )


@triton.jit
def _probs(q, k, lse, offs_m, offs_n, N, qk_scale, F64: tl.constexpr, PREC: tl.constexpr):
    """P tile recomputed from Q, K and the natural-log row logsumexp; zero outside [0, N) x [0, N)."""
    # fma: scale * (q.k) - L rounded once.
    p = _exp(tl.fma(_mm(q, tl.trans(k), F64, PREC), qk_scale, -lse[:, None]), F64)
    valid = (offs_m[:, None] < N) & (offs_n[None, :] < N)
    return tl.where(valid, p, 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# forward
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    N,
    qk_scale,
    H,
    Q_sb,
    Q_sh,
    Q_sn,
    K_sb,
    K_sh,
    K_sn,
    V_sb,
    V_sh,
    V_sn,
    DO_sb,
    DO_sh,
    DO_sn,
    GQ_sb,
    GQ_sh,
    GQ_sn,
    GK_sb,
    GK_sh,
    GK_sn,
    GV_sb,
    GV_sh,
    GV_sn,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_T: tl.constexpr,
    TILE_T: tl.constexpr,
    ACC_T: tl.constexpr,
    F64: tl.constexpr,
    PREC: tl.constexpr,
):
    pid = tl.program_id(1)
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    q = _ld(_base(Q, bh, H, Q_sb, Q_sh), offs_m, offs_d, N, D, Q_sn).to(IN_T)
    m_i = tl.full([BM], float("-inf"), ACC_T)
    l_i = tl.zeros([BM], ACC_T)
    acc = tl.zeros([BM, BD], ACC_T)
    for start in range(0, N, BN):
        offs_n = start + tl.arange(0, BN)
        k = _ld(_base(K, bh, H, K_sb, K_sh), offs_n, offs_d, N, D, K_sn).to(IN_T)
        v = _ld(_base(V, bh, H, V_sb, V_sh), offs_n, offs_d, N, D, V_sn).to(IN_T)
        s = _mm(q, tl.trans(k), F64, PREC) * qk_scale
        s = tl.where((offs_n < N)[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = _exp(m_i - m_new, F64)
        p = _exp(s - m_new[:, None], F64)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + _mm(p.to(TILE_T), v.to(TILE_T), F64, PREC)
        m_i = m_new
    o = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    _st(O + base, o, offs_m, offs_d, N, D)
    tl.store(LSE + bh * N + offs_m, lse, mask=offs_m < N)


# ----------------------------------------------------------------------------------------------------------------------
# first backward
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _rowdot_kernel(
    DO, O, DI, N, H, DO_sb, DO_sh, DO_sn, D: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr, ACC_T: tl.constexpr
):
    """DI[b, h, i] = <dO_i, O_i>, computed once per call instead of once per tile."""
    bh = tl.program_id(0).to(tl.int64)
    offs_r = tl.program_id(1) * BR + tl.arange(0, BR)
    offs_d = tl.arange(0, BD)
    do = _ld(_base(DO, bh, H, DO_sb, DO_sh), offs_r, offs_d, N, D, DO_sn).to(ACC_T)
    o = _ld(O + bh * N * D, offs_r, offs_d, N, D, D).to(ACC_T)
    tl.store(DI + bh * N + offs_r, tl.sum(do * o, 1), mask=offs_r < N)


@triton.jit
def _bwd_col_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    DI,
    DO,
    DQ,
    DK,
    DV,
    N,
    qk_scale,
    sm_scale,
    H,
    Q_sb,
    Q_sh,
    Q_sn,
    K_sb,
    K_sh,
    K_sn,
    V_sb,
    V_sh,
    V_sn,
    DO_sb,
    DO_sh,
    DO_sn,
    GQ_sb,
    GQ_sh,
    GQ_sn,
    GK_sb,
    GK_sh,
    GK_sn,
    GV_sb,
    GV_sh,
    GV_sn,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_T: tl.constexpr,
    TILE_T: tl.constexpr,
    ACC_T: tl.constexpr,
    F64: tl.constexpr,
    PREC: tl.constexpr,
    SINGLE: tl.constexpr,
    ATOMIC_DQ: tl.constexpr,
):
    """Column-parallel dK, dV. With SINGLE (one column tile covers all N keys) it also writes dQ and computes the row
    scalar D itself. With ATOMIC_DQ it adds its dQ partial sums into the fp32 buffer DQ (FA2 style, nondeterministic
    summation order); otherwise dQ comes from _bwd_dq_kernel and DI holds the precomputed row scalars."""
    pid = tl.program_id(1)
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    offs_n = pid * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    k = _ld(_base(K, bh, H, K_sb, K_sh), offs_n, offs_d, N, D, K_sn).to(IN_T)
    v = _ld(_base(V, bh, H, V_sb, V_sh), offs_n, offs_d, N, D, V_sn).to(IN_T)
    dk = tl.zeros([BN, BD], ACC_T)
    dv = tl.zeros([BN, BD], ACC_T)
    for start in range(0, N, BM):
        offs_m = start + tl.arange(0, BM)
        q = _ld(_base(Q, bh, H, Q_sb, Q_sh), offs_m, offs_d, N, D, Q_sn).to(IN_T)
        do = _ld(_base(DO, bh, H, DO_sb, DO_sh), offs_m, offs_d, N, D, DO_sn)
        lse = tl.load(LSE + bh * N + offs_m, mask=offs_m < N, other=0.0).to(ACC_T)
        if SINGLE:
            o = _ld(O + base, offs_m, offs_d, N, D, D)
            di = tl.sum(do.to(ACC_T) * o.to(ACC_T), 1)
        else:
            di = tl.load(DI + bh * N + offs_m, mask=offs_m < N, other=0.0)
        do = do.to(IN_T)
        p = _probs(q, k, lse, offs_m, offs_n, N, qk_scale, F64, PREC)
        dv += _mm(tl.trans(p.to(TILE_T)), do.to(TILE_T), F64, PREC)
        dp = _mm(do, tl.trans(v), F64, PREC)
        ds = p * (dp - di[:, None])
        ds_c = ds.to(TILE_T)
        dk += _mm(tl.trans(ds_c), q.to(TILE_T), F64, PREC)
        if SINGLE:
            dq = _mm(ds_c, k.to(TILE_T), F64, PREC) * sm_scale
            _st(DQ + base, dq, offs_m, offs_d, N, D)
        if ATOMIC_DQ:
            dq = _mm(ds_c, k.to(TILE_T), F64, PREC) * sm_scale
            tl.atomic_add(
                DQ + base + offs_m[:, None] * D + offs_d[None, :],
                dq,
                mask=(offs_m[:, None] < N) & (offs_d[None, :] < D),
                sem="relaxed",
            )
    _st(DK + base, dk * sm_scale, offs_n, offs_d, N, D)
    _st(DV + base, dv, offs_n, offs_d, N, D)


@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    LSE,
    DI,
    DO,
    DQ,
    N,
    qk_scale,
    sm_scale,
    H,
    Q_sb,
    Q_sh,
    Q_sn,
    K_sb,
    K_sh,
    K_sn,
    V_sb,
    V_sh,
    V_sn,
    DO_sb,
    DO_sh,
    DO_sn,
    GQ_sb,
    GQ_sh,
    GQ_sn,
    GK_sb,
    GK_sh,
    GK_sn,
    GV_sb,
    GV_sh,
    GV_sn,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_T: tl.constexpr,
    TILE_T: tl.constexpr,
    ACC_T: tl.constexpr,
    F64: tl.constexpr,
    PREC: tl.constexpr,
):
    pid = tl.program_id(1)
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    q = _ld(_base(Q, bh, H, Q_sb, Q_sh), offs_m, offs_d, N, D, Q_sn).to(IN_T)
    do = _ld(_base(DO, bh, H, DO_sb, DO_sh), offs_m, offs_d, N, D, DO_sn).to(IN_T)
    lse = tl.load(LSE + bh * N + offs_m, mask=offs_m < N, other=0.0).to(ACC_T)
    di = tl.load(DI + bh * N + offs_m, mask=offs_m < N, other=0.0)
    dq = tl.zeros([BM, BD], ACC_T)
    for start in range(0, N, BN):
        offs_n = start + tl.arange(0, BN)
        k = _ld(_base(K, bh, H, K_sb, K_sh), offs_n, offs_d, N, D, K_sn).to(IN_T)
        v = _ld(_base(V, bh, H, V_sb, V_sh), offs_n, offs_d, N, D, V_sn).to(IN_T)
        p = _probs(q, k, lse, offs_m, offs_n, N, qk_scale, F64, PREC)
        dp = _mm(do, tl.trans(v), F64, PREC)
        ds = p * (dp - di[:, None])
        dq += _mm(ds.to(TILE_T), k.to(TILE_T), F64, PREC)
    _st(DQ + base, dq * sm_scale, offs_m, offs_d, N, D)


# ----------------------------------------------------------------------------------------------------------------------
# backward of the backward
# ----------------------------------------------------------------------------------------------------------------------


@triton.jit
def _bob_row_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    DO,
    GQ,
    GK,
    GV,
    DI,
    ES,
    TS,
    HQ,
    HDO,
    N,
    qk_scale,
    sm_scale,
    H,
    Q_sb,
    Q_sh,
    Q_sn,
    K_sb,
    K_sh,
    K_sn,
    V_sb,
    V_sh,
    V_sn,
    DO_sb,
    DO_sh,
    DO_sn,
    GQ_sb,
    GQ_sh,
    GQ_sn,
    GK_sb,
    GK_sh,
    GK_sn,
    GV_sb,
    GV_sh,
    GV_sn,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_T: tl.constexpr,
    TILE_T: tl.constexpr,
    ACC_T: tl.constexpr,
    F64: tl.constexpr,
    PREC: tl.constexpr,
):
    """Row-parallel BoB pass: hQ, hdO and the row scalars E, T (stored for the column pass)."""
    pid = tl.program_id(1)
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    offs_m = pid * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, BD)
    q = _ld(_base(Q, bh, H, Q_sb, Q_sh), offs_m, offs_d, N, D, Q_sn).to(IN_T)
    gq = _ld(_base(GQ, bh, H, GQ_sb, GQ_sh), offs_m, offs_d, N, D, GQ_sn).to(IN_T)
    do = _ld(_base(DO, bh, H, DO_sb, DO_sh), offs_m, offs_d, N, D, DO_sn)
    o = _ld(O + base, offs_m, offs_d, N, D, D).to(ACC_T)
    lse = tl.load(LSE + bh * N + offs_m, mask=offs_m < N, other=0.0).to(ACC_T)
    di = tl.sum(do.to(ACC_T) * o, 1)
    do = do.to(IN_T)
    e_i = tl.zeros([BM], ACC_T)
    t_i = tl.zeros([BM], ACC_T)
    x = tl.zeros([BM, BD], ACC_T)
    u = tl.zeros([BM, BD], ACC_T)
    z = tl.zeros([BM, BD], ACC_T)
    w = tl.zeros([BM, BD], ACC_T)
    for start in range(0, N, BN):
        offs_n = start + tl.arange(0, BN)
        k = _ld(_base(K, bh, H, K_sb, K_sh), offs_n, offs_d, N, D, K_sn).to(IN_T)
        v = _ld(_base(V, bh, H, V_sb, V_sh), offs_n, offs_d, N, D, V_sn).to(IN_T)
        gk = _ld(_base(GK, bh, H, GK_sb, GK_sh), offs_n, offs_d, N, D, GK_sn).to(IN_T)
        gv = _ld(_base(GV, bh, H, GV_sb, GV_sh), offs_n, offs_d, N, D, GV_sn).to(IN_T)
        p = _probs(q, k, lse, offs_m, offs_n, N, qk_scale, F64, PREC)
        a = (_mm(gq, tl.trans(k), F64, PREC) + _mm(q, tl.trans(gk), F64, PREC)) * sm_scale
        dp = _mm(do, tl.trans(v), F64, PREC)
        b = _mm(do, tl.trans(gv), F64, PREC)
        ds = p * (dp - di[:, None])
        pa = p * a
        m1 = ds * a + p * b
        e_i += tl.sum(pa, 1)
        t_i += tl.sum(m1, 1)
        ds_c = ds.to(TILE_T)
        p_c = p.to(TILE_T)
        k_t = k.to(TILE_T)
        x += _mm(m1.to(TILE_T), k_t, F64, PREC) + _mm(ds_c, gk.to(TILE_T), F64, PREC)
        u += _mm(ds_c, k_t, F64, PREC)
        z += _mm(p_c, k_t, F64, PREC)
        w += _mm(pa.to(TILE_T), v.to(TILE_T), F64, PREC) + _mm(p_c, gv.to(TILE_T), F64, PREC)
    hq = (x - e_i[:, None] * u - t_i[:, None] * z) * sm_scale
    hdo = w - e_i[:, None] * o
    _st(HQ + base, hq, offs_m, offs_d, N, D)
    _st(HDO + base, hdo, offs_m, offs_d, N, D)
    tl.store(DI + bh * N + offs_m, di, mask=offs_m < N)
    tl.store(ES + bh * N + offs_m, e_i, mask=offs_m < N)
    tl.store(TS + bh * N + offs_m, t_i, mask=offs_m < N)


@triton.jit
def _bob_col_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    DO,
    GQ,
    GK,
    GV,
    DI,
    ES,
    TS,
    HQ,
    HK,
    HV,
    HDO,
    N,
    qk_scale,
    sm_scale,
    H,
    Q_sb,
    Q_sh,
    Q_sn,
    K_sb,
    K_sh,
    K_sn,
    V_sb,
    V_sh,
    V_sn,
    DO_sb,
    DO_sh,
    DO_sn,
    GQ_sb,
    GQ_sh,
    GQ_sn,
    GK_sb,
    GK_sh,
    GK_sn,
    GV_sb,
    GV_sh,
    GV_sn,
    D: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IN_T: tl.constexpr,
    TILE_T: tl.constexpr,
    ACC_T: tl.constexpr,
    F64: tl.constexpr,
    PREC: tl.constexpr,
    SINGLE: tl.constexpr,
):
    """Column-parallel BoB pass: hK, hV. With SINGLE (one column tile covers all keys) the row scalars E, T are
    complete inside each row block, so this kernel also writes hQ, hdO and needs no scratch."""
    pid = tl.program_id(1)
    bh = tl.program_id(0).to(tl.int64)
    base = bh * N * D
    offs_n = pid * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, BD)
    k = _ld(_base(K, bh, H, K_sb, K_sh), offs_n, offs_d, N, D, K_sn).to(IN_T)
    v = _ld(_base(V, bh, H, V_sb, V_sh), offs_n, offs_d, N, D, V_sn).to(IN_T)
    gk = _ld(_base(GK, bh, H, GK_sb, GK_sh), offs_n, offs_d, N, D, GK_sn).to(IN_T)
    gv = _ld(_base(GV, bh, H, GV_sb, GV_sh), offs_n, offs_d, N, D, GV_sn).to(IN_T)
    hk = tl.zeros([BN, BD], ACC_T)
    hv = tl.zeros([BN, BD], ACC_T)
    for start in range(0, N, BM):
        offs_m = start + tl.arange(0, BM)
        q = _ld(_base(Q, bh, H, Q_sb, Q_sh), offs_m, offs_d, N, D, Q_sn).to(IN_T)
        gq = _ld(_base(GQ, bh, H, GQ_sb, GQ_sh), offs_m, offs_d, N, D, GQ_sn).to(IN_T)
        do = _ld(_base(DO, bh, H, DO_sb, DO_sh), offs_m, offs_d, N, D, DO_sn)
        lse = tl.load(LSE + bh * N + offs_m, mask=offs_m < N, other=0.0).to(ACC_T)
        if SINGLE:
            o = _ld(O + base, offs_m, offs_d, N, D, D)
            di = tl.sum(do.to(ACC_T) * o.to(ACC_T), 1)
        else:
            di = tl.load(DI + bh * N + offs_m, mask=offs_m < N, other=0.0)
        do = do.to(IN_T)
        p = _probs(q, k, lse, offs_m, offs_n, N, qk_scale, F64, PREC)
        a = (_mm(gq, tl.trans(k), F64, PREC) + _mm(q, tl.trans(gk), F64, PREC)) * sm_scale
        dp = _mm(do, tl.trans(v), F64, PREC)
        b = _mm(do, tl.trans(gv), F64, PREC)
        ds = p * (dp - di[:, None])
        pa = p * a
        m1 = ds * a + p * b
        if SINGLE:
            e_i = tl.sum(pa, 1)
            t_i = tl.sum(m1, 1)
        else:
            e_i = tl.load(ES + bh * N + offs_m, mask=offs_m < N, other=0.0)
            t_i = tl.load(TS + bh * N + offs_m, mask=offs_m < N, other=0.0)
        ds2 = (m1 - e_i[:, None] * ds - t_i[:, None] * p).to(TILE_T)
        pae = (pa - e_i[:, None] * p).to(TILE_T)
        ds_c = ds.to(TILE_T)
        hk += _mm(tl.trans(ds2), q.to(TILE_T), F64, PREC) + _mm(tl.trans(ds_c), gq.to(TILE_T), F64, PREC)
        hv += _mm(tl.trans(pae), do.to(TILE_T), F64, PREC)
        if SINGLE:
            hq = (_mm(ds2, k.to(TILE_T), F64, PREC) + _mm(ds_c, gk.to(TILE_T), F64, PREC)) * sm_scale
            hdo = _mm(pae, v.to(TILE_T), F64, PREC) + _mm(p.to(TILE_T), gv.to(TILE_T), F64, PREC)
            _st(HQ + base, hq, offs_m, offs_d, N, D)
            _st(HDO + base, hdo, offs_m, offs_d, N, D)
    _st(HK + base, hk * sm_scale, offs_n, offs_d, N, D)
    _st(HV + base, hv, offs_n, offs_d, N, D)


# ----------------------------------------------------------------------------------------------------------------------
# launchers
# ----------------------------------------------------------------------------------------------------------------------

_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float64: tl.float64,
}


def _cfg(q: torch.Tensor, input_precision: str | None, lowp_dots: bool | None):
    if q.dtype not in _TL_DTYPE:
        raise TypeError(f"fused attention: unsupported dtype {q.dtype}")
    if q.dim() != 4:
        raise ValueError(f"fused attention expects [B, H, N, D], got {tuple(q.shape)}")
    f64 = q.dtype == torch.float64
    prec = input_precision or DEFAULT_FP32_PRECISION
    if prec not in ("tf32", "tf32rn", "ieee", "tf32x3"):
        raise ValueError(f"input_precision must be tf32, tf32rn, tf32x3 or ieee, got {prec}")
    lowp = DEFAULT_LOWP_DOTS if lowp_dots is None else lowp_dots
    # in_t: dtype of the original operands (q, k, v, dO, gQ, gK, gV) in dots between two of them (S = QK^T,
    # dP = dO V^T, gQ K^T, Q gK^T, dO gV^T). For bf16/fp16 inputs this is the input dtype: exact products, fp32
    # accumulation, identical to upcasting and running a TF32 dot.
    # tile_t: dtype a derived N x N tile (P, dS, and the BoB tiles) is rounded to before it enters a dot; the other
    # operand is cast to match. lowp_dots=True: the input dtype (fast, bf16 tensor cores). lowp_dots=False: fp32
    # (rounded to tf32 by PREC), which is what the math backend does.
    if f64:
        in_t = tile_t = acc_t = tl.float64
    elif q.dtype in (torch.bfloat16, torch.float16):
        in_t, acc_t = _TL_DTYPE[q.dtype], tl.float32
        tile_t = in_t if lowp else tl.float32
    else:
        in_t = tile_t = acc_t = tl.float32
    B, H, N, D = q.shape
    bd = max(16, triton.next_power_of_2(D))
    return dict(f64=f64, prec=prec, in_t=in_t, tile_t=tile_t, acc_t=acc_t, B=B, H=H, N=N, D=D, bd=bd)


def _c(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _d1(t: torch.Tensor) -> torch.Tensor:
    """Inputs may have any batch/head/token strides (e.g. views of a fused qkv projection, or expanded zeros); only the
    head dim must be unit-stride."""
    return t if (t.stride(-1) == 1 or t.shape[-1] == 1) else t.contiguous()


def _strides(q, k=None, v=None, do=None, gq=None, gk=None, gv=None):
    out = [q.shape[1]]
    for t in (q, k, v, do, gq, gk, gv):
        t = q if t is None else t
        out += [t.stride(0), t.stride(1), t.stride(2)]
    return out


# Launch plans (BM, BN, num_warps, num_stages). PLAN_OVERRIDE[key] with key in
# {"fwd", "bwd_mode", "bwd_single", "bwd_col", "bwd_dq", "bob_mode", "bob_single", "bob_row", "bob_col"} forces a
# value. bwd_mode "atomic" is not run-to-run deterministic and is never a default.
PLAN_OVERRIDE: dict = {}


def _plans(c):
    """Default plans for this call's shape class."""
    N, bd = c["N"], c["bd"]
    np2 = max(16, triton.next_power_of_2(N))
    if c["f64"]:
        p = (16, 16, 4, 1)
        return dict(fwd=p, bwd_mode="two", bwd_col=p, bwd_dq=p, bob_mode="two", bob_row=p, bob_col=p)
    lowp = c["tile_t"] != tl.float32  # derived tiles go into bf16/fp16 tensor-core dots
    ieee = (not lowp) and c["prec"] in ("ieee", "tf32x3")
    if ieee:
        return dict(
            fwd=(32, 32, 4, 1),
            bwd_mode="two",
            bwd_col=(16, 32, 4, 1),
            bwd_dq=(32, 16, 4, 1),
            bob_mode="two",
            bob_row=(16, 16, 4, 1),
            bob_col=(16, 16, 4, 1),
        )
    small = N <= 128 and bd <= 64
    if lowp:
        return dict(
            fwd=(64, 32, 4, 3) if N <= 128 else (64, 64, 4, 3),
            bwd_mode="single" if small else "two",
            bwd_single=(32, np2, 4, 1),
            bwd_col=(64, 64, 4, 3),
            bwd_dq=(64, 32, 4, 3),
            bob_mode="single" if small else "two",
            bob_single=(16, np2, 4, 1),
            bob_row=(64, 32, 4, 3),
            bob_col=(64, 64, 4, 2),
        )
    # fp32 tiles (fp32 inputs, or bf16 inputs with lowp_dots=False): tf32 dots, twice the register/smem footprint, so
    # the single-tile path loses to the streaming kernels even at N = 81.
    if N <= 128:
        return dict(
            fwd=(128, 32, 8, 3),
            bwd_mode="two",
            bwd_col=(32, 32, 4, 3),
            bwd_dq=(32, 64, 4, 2),
            bob_mode="two",
            bob_row=(32, 32, 4, 3),
            bob_col=(32, 32, 4, 3),
        )
    return dict(
        fwd=(128, 32, 8, 3),
        bwd_mode="two",
        bwd_col=(64, 32, 4, 3),
        bwd_dq=(32, 64, 4, 3),
        bob_mode="two",
        bob_row=(64, 32, 4, 3),
        bob_col=(64, 32, 4, 3),
    )


def _plan(c, key):
    if key in PLAN_OVERRIDE:
        return PLAN_OVERRIDE[key]
    return _plans(c)[key]


def _rowdot(do, o, c):
    B, H, N, D = c["B"], c["H"], c["N"], c["D"]
    di = torch.empty((B, H, N), device=o.device, dtype=torch.float64 if c["f64"] else torch.float32)
    br = 64
    _rowdot_kernel[(B * H, triton.cdiv(N, br))](
        do, o, di, N, H, do.stride(0), do.stride(1), do.stride(2), D=D, BD=c["bd"], BR=br, ACC_T=c["acc_t"], num_warps=4
    )
    return di


def _scales(scale: float, f64: bool):
    # Triton passes python float scalars as fp32; round once here so every kernel uses the identical value. (The fp64
    # gradcheck path sees the same fp32-rounded scale.)
    s32 = float(torch.tensor(scale, dtype=torch.float32))
    return s32, s32


def attn_fwd(q, k, v, scale: float, *, input_precision: str | None = None, lowp_dots: bool | None = None):
    """O = softmax(scale Q K^T) V and the natural-log row logsumexp L (fp32; fp64 for fp64 inputs)."""
    q, k, v = _d1(q), _d1(k), _d1(v)
    c = _cfg(q, input_precision, lowp_dots)
    B, H, N, D = c["B"], c["H"], c["N"], c["D"]
    o = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)
    st = _strides(q, k, v)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float64 if c["f64"] else torch.float32)
    qk_scale, _ = _scales(scale, c["f64"])
    bm, bn, warps, stages = _plan(c, "fwd")
    grid = (B * H, triton.cdiv(N, bm))  # batch x heads on axis 0: up to 2^31 - 1 blocks (axis 1 stops at 65,535)
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        N,
        qk_scale,
        *st,
        D=D,
        BD=c["bd"],
        BM=bm,
        BN=bn,
        IN_T=c["in_t"],
        TILE_T=c["tile_t"],
        ACC_T=c["acc_t"],
        F64=c["f64"],
        PREC=c["prec"],
        num_warps=warps,
        num_stages=stages,
    )
    return o, lse


def attn_bwd(q, k, v, o, lse, do, scale: float, *, input_precision: str | None = None, lowp_dots: bool | None = None):
    """(dQ, dK, dV) of <dO, O>."""
    q, k, v, do = _d1(q), _d1(k), _d1(v), _d1(do)
    o, lse = _c(o), _c(lse)
    c = _cfg(q, input_precision, lowp_dots)
    B, H, N, D = c["B"], c["H"], c["N"], c["D"]
    dq, dk, dv = (torch.empty((B, H, N, D), device=q.device, dtype=q.dtype) for _ in range(3))
    st = _strides(q, k, v, do)
    qk_scale, sm_scale = _scales(scale, c["f64"])
    common = dict(D=D, BD=c["bd"], IN_T=c["in_t"], TILE_T=c["tile_t"], ACC_T=c["acc_t"], F64=c["f64"], PREC=c["prec"])
    mode = _plan(c, "bwd_mode")
    if mode == "single":
        bm, bn, warps, stages = _plan(c, "bwd_single")
        assert bn >= N
        _bwd_col_kernel[(B * H, 1)](
            q,
            k,
            v,
            o,
            lse,
            lse,
            do,
            dq,
            dk,
            dv,
            N,
            qk_scale,
            sm_scale,
            *st,
            BM=bm,
            BN=bn,
            SINGLE=True,
            ATOMIC_DQ=False,
            num_warps=warps,
            num_stages=stages,
            **common,
        )
        return dq, dk, dv
    di = _rowdot(do, o, c)
    bm, bn, warps, stages = _plan(c, "bwd_col")
    if mode == "atomic":
        dq32 = torch.zeros((B, H, N, D), device=q.device, dtype=torch.float64 if c["f64"] else torch.float32)
        _bwd_col_kernel[(B * H, triton.cdiv(N, bn))](
            q,
            k,
            v,
            o,
            lse,
            di,
            do,
            dq32,
            dk,
            dv,
            N,
            qk_scale,
            sm_scale,
            *st,
            BM=bm,
            BN=bn,
            SINGLE=False,
            ATOMIC_DQ=True,
            num_warps=warps,
            num_stages=stages,
            **common,
        )
        return dq32.to(q.dtype), dk, dv
    _bwd_col_kernel[(B * H, triton.cdiv(N, bn))](
        q,
        k,
        v,
        o,
        lse,
        di,
        do,
        dq,
        dk,
        dv,
        N,
        qk_scale,
        sm_scale,
        *st,
        BM=bm,
        BN=bn,
        SINGLE=False,
        ATOMIC_DQ=False,
        num_warps=warps,
        num_stages=stages,
        **common,
    )
    bm, bn, warps, stages = _plan(c, "bwd_dq")
    _bwd_dq_kernel[(B * H, triton.cdiv(N, bm))](
        q,
        k,
        v,
        lse,
        di,
        do,
        dq,
        N,
        qk_scale,
        sm_scale,
        *st,
        BM=bm,
        BN=bn,
        num_warps=warps,
        num_stages=stages,
        **common,
    )
    return dq, dk, dv


def attn_bwd_bwd(
    q,
    k,
    v,
    o,
    lse,
    do,
    dq,
    gq,
    gk,
    gv,
    scale: float,
    *,
    input_precision: str | None = None,
    lowp_dots: bool | None = None,
):
    """(hQ, hK, hV, hdO): total derivative of <gQ, dQ> + <gK, dK> + <gV, dV> w.r.t. (Q, K, V, dO).

    ``dq`` is accepted for the op contract and not read (the kernels recompute dS tile by tile). Any of gq, gk, gv may
    be None (treated as zeros).
    """
    del dq
    q, k, v, do = _d1(q), _d1(k), _d1(v), _d1(do)
    o, lse = _c(o), _c(lse)
    c = _cfg(q, input_precision, lowp_dots)
    B, H, N, D = c["B"], c["H"], c["N"], c["D"]
    # A missing cotangent is one zero row expanded to [B, H, N, D] (strides 0, 0, 0, 1: the kernels need unit stride
    # in D), so no N x D memory is allocated or cleared.
    zero = lambda: torch.zeros((D,), device=q.device, dtype=q.dtype).expand(B, H, N, D)
    gq = zero() if gq is None else _d1(gq.to(q.dtype))
    gk = zero() if gk is None else _d1(gk.to(q.dtype))
    gv = zero() if gv is None else _d1(gv.to(q.dtype))
    hq, hk, hv, hdo = (torch.empty((B, H, N, D), device=q.device, dtype=q.dtype) for _ in range(4))
    st = _strides(q, k, v, do, gq, gk, gv)
    qk_scale, sm_scale = _scales(scale, c["f64"])
    common = dict(D=D, BD=c["bd"], IN_T=c["in_t"], TILE_T=c["tile_t"], ACC_T=c["acc_t"], F64=c["f64"], PREC=c["prec"])
    if _plan(c, "bob_mode") == "single":
        bm, bn, warps, stages = _plan(c, "bob_single")
        assert bn >= N
        _bob_col_kernel[(B * H, 1)](
            q,
            k,
            v,
            o,
            lse,
            do,
            gq,
            gk,
            gv,
            lse,
            lse,
            lse,
            hq,
            hk,
            hv,
            hdo,
            N,
            qk_scale,
            sm_scale,
            *st,
            BM=bm,
            BN=bn,
            SINGLE=True,
            num_warps=warps,
            num_stages=stages,
            **common,
        )
        return hq, hk, hv, hdo
    stat_dtype = torch.float64 if c["f64"] else torch.float32
    rows = torch.empty((3, B, H, N), device=q.device, dtype=stat_dtype)  # D, E, T row scalars
    di, es, ts = rows[0], rows[1], rows[2]
    bm, bn, warps, stages = _plan(c, "bob_row")
    _bob_row_kernel[(B * H, triton.cdiv(N, bm))](
        q,
        k,
        v,
        o,
        lse,
        do,
        gq,
        gk,
        gv,
        di,
        es,
        ts,
        hq,
        hdo,
        N,
        qk_scale,
        sm_scale,
        *st,
        BM=bm,
        BN=bn,
        num_warps=warps,
        num_stages=stages,
        **common,
    )
    bm, bn, warps, stages = _plan(c, "bob_col")
    _bob_col_kernel[(B * H, triton.cdiv(N, bn))](
        q,
        k,
        v,
        o,
        lse,
        do,
        gq,
        gk,
        gv,
        di,
        es,
        ts,
        hq,
        hk,
        hv,
        hdo,
        N,
        qk_scale,
        sm_scale,
        *st,
        BM=bm,
        BN=bn,
        SINGLE=False,
        num_warps=warps,
        num_stages=stages,
        **common,
    )
    return hq, hk, hv, hdo


__all__ = ["attn_fwd", "attn_bwd", "attn_bwd_bwd", "DEFAULT_FP32_PRECISION", "DEFAULT_LOWP_DOTS", "PLAN_OVERRIDE"]
