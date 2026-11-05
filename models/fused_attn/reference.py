"""Dense PyTorch reference of the fused attention ops (CPU implementation and test oracle)."""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _compute_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def lse_dtype_for(x: Tensor) -> torch.dtype:
    """``lse`` is float32 per the contract; float64 inputs keep float64 so f64 checks stay exact."""
    return _compute_dtype(x)


def default_scale(q: Tensor, scale: float | None) -> float:
    return float(q.shape[-1] ** -0.5) if scale is None else float(scale)


def _probs(q: Tensor, k: Tensor, scale: float, lse: Tensor | None = None) -> tuple[Tensor, Tensor]:
    ct = _compute_dtype(q)
    s = torch.matmul(q.to(ct), k.to(ct).transpose(-1, -2)) * scale
    if lse is None:
        lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse.to(ct).unsqueeze(-1))
    return p, lse


def attn_fwd(q: Tensor, k: Tensor, v: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    ct = _compute_dtype(q)
    p, lse = _probs(q, k, scale)
    o = torch.matmul(p, v.to(ct))
    return o.to(q.dtype), lse.to(lse_dtype_for(q))


def _first_backward(q, k, v, o, lse, do, scale):
    ct = _compute_dtype(q)
    p, _ = _probs(q, k, scale, lse)
    dof, vf = do.to(ct), v.to(ct)
    dp = torch.matmul(dof, vf.transpose(-1, -2))
    # D = rowsum(dO * O). Using the stored O (as the kernels do) rather than rowsum(P * dP).
    d = (dof * o.to(ct)).sum(-1, keepdim=True)
    ds = p * (dp - d)
    return p, dp, d, ds


def attn_bwd(
    q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, do: Tensor, scale: float
) -> tuple[Tensor, Tensor, Tensor]:
    ct = _compute_dtype(q)
    p, _, _, ds = _first_backward(q, k, v, o, lse, do, scale)
    dq = torch.matmul(ds, k.to(ct)) * scale
    dk = torch.matmul(ds.transpose(-1, -2), q.to(ct)) * scale
    dv = torch.matmul(p.transpose(-1, -2), do.to(ct))
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    del dq  # passed only so kernels need not recompute it
    ct = _compute_dtype(q)
    qf, kf, vf, dof = q.to(ct), k.to(ct), v.to(ct), do.to(ct)
    gqf, gkf, gvf = gq.to(ct), gk.to(ct), gv.to(ct)
    p, dp, d, ds = _first_backward(q, k, v, o, lse, do, scale)

    g = (torch.matmul(gqf, kf.transpose(-1, -2)) + torch.matmul(qf, gkf.transpose(-1, -2))) * scale
    r = torch.matmul(dof, gvf.transpose(-1, -2))
    e = (p * g).sum(-1, keepdim=True)
    a = p * (g - e)
    gp = (dp - d) * g - e * dp + r
    gs = p * (gp - (p * gp).sum(-1, keepdim=True))

    hq = (torch.matmul(gs, kf) + torch.matmul(ds, gkf)) * scale
    hk = (torch.matmul(gs.transpose(-1, -2), qf) + torch.matmul(ds.transpose(-1, -2), gqf)) * scale
    hv = torch.matmul(a.transpose(-1, -2), dof)
    hdo = torch.matmul(a, vf) + torch.matmul(p, gvf)
    return hq.to(q.dtype), hk.to(k.dtype), hv.to(v.dtype), hdo.to(do.dtype)


__all__ = ["attn_bwd", "attn_bwd_bwd", "attn_fwd", "default_scale", "lse_dtype_for"]
