"""EBR: a latent state z refined by gradient descent on a learned scalar energy E(z, x).

One descent step (a microstep) is  z <- sphere(rms_norm(z) + noise - alpha * dE/dz), where rms_norm normalizes each
cell's vector and sphere rescales the whole board to unit RMS. A segment runs ``cycles`` microsteps. During training
only the last ``grad_cycles`` of them build a graph (``create_graph=True``), so the loss reaches the energy through
dE/dz. Segments are chained: a row that has halted gets a fresh random state and a new puzzle at the next segment.

A row halts when its descent has converged: its residual r, the mean relative move ||z_t - z_{t-1}|| / ||z_{t-1}|| of
the last ``convergence_window`` microsteps, falls below ``convergence_threshold``. There is no learned halting head;
the convergence loss shapes the energy through r.

Training also uses
  * forward SXM (``explore_k``): roll out K candidate trajectories, train and carry the one closest to the label;
  * reverse SXM (``late_nearest_ce_steps``, models/sxm.py): grade the last microsteps against the nearest solution;
  * noisy solution initialization (``noisy_solution_init``): start some rows from a noisy version of their solution.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn

from models.attention import EnergyReasoningBlock, get_backend, second_order_window
from models.layers import (
    RMS_NORM_EPS,
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding2D,
    RotaryTable2D,
    rms_norm,
    trunc_normal_init_,
)
from models.losses import IGNORE_LABEL_ID, stablemax_cross_entropy
from models.sxm import reverse_sxm_targets, solution_sets


class EnergyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Filled in by the trainer from the dataset metadata.
    seq_len: int
    vocab_size: int

    # Energy network: num_layers attention + SwiGLU blocks with 2D RoPE over a board_height x board_width board.
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    board_height: int
    board_width: int
    forward_dtype: str
    energy_cell_activation: str = "linear"  # linear | softplus (per-cell energy >= 0)
    # Graded microsteps need attention with a second derivative (math, fused); no-grad microsteps may use any.
    attn_backend: str
    attn_nograd_backend: str

    # Segments: a training segment runs U[randomize_min_cycles, cycles] microsteps (rows that just reset run
    # U[reset_cycles_low, reset_cycles_high]); the last grad_cycles carry a graph. Eval segments run all cycles.
    cycles: int
    grad_cycles: int
    randomize_min_cycles: int
    reset_cycles_low: int
    reset_cycles_high: int
    halt_max_steps: int  # segments per puzzle
    halt_exploration_prob: float

    # Descent: training draws a step size alpha ~ U[step_size_low, step_size_high] per row, eval uses step_size.
    step_size: float
    step_size_low: float
    step_size_high: float
    init_std: float
    # Noise per microstep: noise_cosine anneals sigma from noise_settle_hot at segment 0 to noise_scale at the last
    # segment; otherwise sigma = noise_scale. training_noise_scale replaces it with a constant during training.
    noise_scale: float
    noise_cosine: bool
    noise_settle_hot: float
    training_noise_scale: float | None = None
    # Interior loss: weight of the CE of every graded microstep, ramped by w_t = t^p / sum s^p.
    microstep_ce: float
    microstep_ce_ramp: float

    # Convergence halting: the convergence logit is (convergence_threshold - r) / convergence_temperature.
    convergence_threshold: float
    convergence_temperature: float
    convergence_window: int

    # Forward SXM: explore_k candidate rollouts per segment, the lowest-loss one is trained and carried.
    explore_k: int
    # Training only: noisy solution initialization. A reset row starts at a mix of noise and its solution embedding,
    # with a ~ U[0, 1] per row (a = 0 is the inference initialization, a = 1 decodes to the solution).
    noisy_solution_init: bool = False
    # Reverse SXM (models/sxm.py): the endpoint and the last late_nearest_ce_steps graded microsteps are graded
    # against the nearest solution of the same input in the batch. A solution is not reused until
    # late_nearest_pool_coverage of the input's solutions are taken.
    late_nearest_ce_steps: int = 0
    late_nearest_pool_coverage: float = 0.75
    explore_rank_max_completions: int = 64

    # Speed options. energy_compile compiles dE/dz (checked against eager); descent_wrapper_compile also compiles the
    # pointwise ops around it (changes GPU rounding); rope_impl fused applies the same RoPE values in a fused kernel;
    # explore_reuse_prefix and avoid_host_syncs skip redundant work.
    energy_compile: bool
    descent_wrapper_compile: bool = False
    rope_impl: str = "broadcast"
    explore_reuse_prefix: bool = False
    avoid_host_syncs: bool = False

    @model_validator(mode="after")
    def _validate(self) -> EnergyConfig:
        if not 1 <= self.grad_cycles <= self.randomize_min_cycles <= self.cycles:
            raise ValueError("need 1 <= grad_cycles <= randomize_min_cycles <= cycles")
        if not self.grad_cycles <= self.reset_cycles_low <= self.reset_cycles_high <= self.cycles:
            raise ValueError("need grad_cycles <= reset_cycles_low <= reset_cycles_high <= cycles")
        if not 0 < self.step_size_low < self.step_size_high:
            raise ValueError("need 0 < step_size_low < step_size_high")
        if self.energy_cell_activation not in ("linear", "softplus"):
            raise ValueError("energy_cell_activation must be linear or softplus")
        if self.rope_impl not in ("broadcast", "fused"):
            raise ValueError("rope_impl must be broadcast or fused")
        if not 1 <= self.convergence_window <= self.grad_cycles:
            raise ValueError("convergence_window must be in [1, grad_cycles]")
        if self.explore_k < 1 or (self.explore_reuse_prefix and self.explore_k == 1):
            raise ValueError("explore_k must be >= 1 (and > 1 for explore_reuse_prefix)")
        if self.late_nearest_ce_steps and not 0 < self.late_nearest_ce_steps <= self.grad_cycles:
            raise ValueError("late_nearest_ce_steps must be in [0, grad_cycles]")
        if not 0.0 < self.late_nearest_pool_coverage <= 1.0:
            raise ValueError("late_nearest_pool_coverage must be in (0, 1]")
        if not get_backend(self.attn_backend).second_order:
            raise ValueError(f"attn_backend={self.attn_backend!r} has no second derivative; use math or fused")
        get_backend(self.attn_nograd_backend)
        return self


# ----------------------------------------------------------------------------
# State carried between segments
# ----------------------------------------------------------------------------


@dataclass
class LatentCarry:
    z: torch.Tensor


@dataclass
class ModelCarry:
    inner_carry: LatentCarry
    steps: torch.Tensor  # segments run on the current puzzle
    halted: torch.Tensor  # rows that get a new puzzle and a fresh state at the next segment
    current_data: dict[str, torch.Tensor]  # each row's current puzzle


@dataclass
class SegmentPlan:
    """Per-segment settings of the descent, drawn by ``EnergyReasoner`` before the segment runs. None = default."""

    noise_scale: torch.Tensor | None = None  # [B, 1, 1] sigma per row (cosine schedule or the training override)
    step_size: torch.Tensor | None = None  # [B, 1, 1] alpha per row (training)
    row_cycles: torch.Tensor | None = None  # [B] microsteps of each row that just reset (0 = the segment's length)
    cycles: int | None = None  # segment length; None = draw it in ``segment_length``
    loop_plan: tuple[int, int, int] | None = None  # avoid_host_syncs: (cycles, loop length, min row cycles)

    def tiled(self, k: int) -> SegmentPlan:
        """The plan for ``k`` copies of the batch stacked along the rows."""
        tile = lambda t: None if t is None else torch.cat([t] * k, dim=0)
        return SegmentPlan(
            tile(self.noise_scale), tile(self.step_size), tile(self.row_cycles), self.cycles, self.loop_plan
        )


@dataclass
class Rollout:
    """What one segment produced besides its endpoint."""

    graded_states: list[torch.Tensor] = field(default_factory=list)  # outputs of the graded microsteps
    window_start: torch.Tensor | None = None  # the state before the graded microsteps (explore_reuse_prefix)


# ----------------------------------------------------------------------------
# Energy
# ----------------------------------------------------------------------------


class PerCellEnergyHead(nn.Module):
    def __init__(self, hidden_size: int, cell_activation: str) -> None:
        super().__init__()
        self.hidden = CastedLinear(hidden_size, hidden_size, bias=True)
        self.output = CastedLinear(hidden_size, 1, bias=False)
        self.cell_activation = cell_activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw = self.output(F.silu(self.hidden(hidden_states)))
        if self.cell_activation == "softplus":
            return F.softplus(raw.to(torch.float32))
        return raw


class ScalarEnergy(nn.Module):
    """E(z, x) = sum over cells of a per-cell scalar head on top of ``num_layers`` attention blocks over z + x."""

    def __init__(self, config: EnergyConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EnergyReasoningBlock(config) for _ in range(config.num_layers))
        self.energy_head = PerCellEnergyHead(config.hidden_size, config.energy_cell_activation)

    def forward(self, latent: torch.Tensor, puzzle: torch.Tensor, cos_sin: Any) -> torch.Tensor:
        hidden_states = latent + puzzle
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return self.energy_head(hidden_states).to(torch.float32).squeeze(-1).sum(dim=1)


def sphere(z: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """Rescale each row (the whole board) to unit RMS, in float32."""
    rms = z.to(torch.float32).square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
    return (z.to(torch.float32) / rms).to(z.dtype)


# descent_wrapper_compile: the pointwise ops before and after dE/dz, compiled as their own graphs.
def _noisy_point(z: torch.Tensor, noise: torch.Tensor, sigma: Any, eps: float) -> torch.Tensor:
    return rms_norm(z, variance_epsilon=eps) + noise * sigma


def _noisy_point_replayed(z: torch.Tensor, noise: torch.Tensor, eps: float) -> torch.Tensor:
    v = rms_norm(z, variance_epsilon=eps)
    return v + noise.to(dtype=v.dtype)


def _step_and_project(
    v: torch.Tensor, grad: torch.Tensor, alpha: Any, z: torch.Tensor, active: torch.Tensor | None, eps: float
) -> torch.Tensor:
    z_next = sphere(v - alpha * grad, eps)
    if active is not None:
        z_next = torch.where(active.to(device=z_next.device), z_next, z.to(dtype=z_next.dtype))
    return z_next


_WRAPPER_COMPILED: dict[str, Callable[..., torch.Tensor]] = {}
WRAPPER_COMPILE_BACKEND = "inductor"  # "aot_eager" works without a C++ toolchain


def _raise_recompile_limit() -> None:
    """One graph per (tier, shape): explore rollouts, winner recompute, eval. Give dynamo room for all of them."""
    dynamo_config = torch._dynamo.config
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(dynamo_config, attr) and getattr(dynamo_config, attr) < 64:
            setattr(dynamo_config, attr, 64)


def _compiled_wrapper() -> dict[str, Callable[..., torch.Tensor]]:
    if not _WRAPPER_COMPILED:
        _raise_recompile_limit()
        for fn in (_noisy_point, _noisy_point_replayed, _step_and_project):
            _WRAPPER_COMPILED[fn.__name__] = torch.compile(fn, backend=WRAPPER_COMPILE_BACKEND, dynamic=False)
    return _WRAPPER_COMPILED


class EnergyDescent(nn.Module):
    """One descent microstep ``z -> sphere(rms_norm(z) + noise - alpha dE/dz)``."""

    def __init__(self, config: EnergyConfig) -> None:
        super().__init__()
        self.energy = ScalarEnergy(config)
        self.config = config
        self.norm_eps = RMS_NORM_EPS
        # The compiled dE/dz is built lazily and kept out of copies (the EMA model is a deep copy).
        self._compiled_energy_grad: Callable[..., torch.Tensor] | None = None
        self._compiled_verified = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_compiled_energy_grad"] = None
        state["_compiled_verified"] = False
        return state

    def normalize_descent_state(self, latent: torch.Tensor) -> torch.Tensor:
        return rms_norm(latent, variance_epsilon=self.norm_eps)

    def forward(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        cos_sin: Any,
        *,
        noise: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One microstep from ``z``.

        ``noise``: a presampled noise tensor to add (it replays a candidate's noise); otherwise noise is drawn with
        per-row ``sigma``, or the config's noise_scale. ``alpha``: per-row step size (training). ``active``: rows that
        move; the others keep ``z``.
        """
        cfg = self.config
        graded = self.training and torch.is_grad_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            if z.is_inference():
                z = z.detach().clone()
            if sigma is not None and sigma.is_inference():
                sigma = sigma.clone()
            v = self._noisy_point(z, noise, sigma, graded)
            if not v.requires_grad:
                v = v.detach().requires_grad_(True)
            grad = self._energy_gradient(v, x, cos_sin, create_graph=graded)
            step = alpha.to(v.dtype) if self.training and alpha is not None else cfg.step_size
            if cfg.descent_wrapper_compile:
                with torch.set_grad_enabled(graded):
                    z_next = _compiled_wrapper()["_step_and_project"](v, grad, step, z, active, self.norm_eps)
            else:
                z_next = _step_and_project(v, grad, step, z, active, self.norm_eps)
        return z_next if graded else z_next.detach()

    def _noisy_point(
        self, z: torch.Tensor, noise: torch.Tensor | None, sigma: torch.Tensor | None, graded: bool
    ) -> torch.Tensor:
        """v = rms_norm(z) + noise, the point where dE/dz is taken."""
        if sigma is not None:
            sigma = sigma.to(device=z.device, dtype=z.dtype)
        elif self.config.noise_scale:
            sigma = self.config.noise_scale
        if self.config.descent_wrapper_compile:
            with torch.set_grad_enabled(graded):
                if noise is not None:
                    return _compiled_wrapper()["_noisy_point_replayed"](z, noise.to(device=z.device), self.norm_eps)
                if sigma is not None:  # the draw stays eager: same shape, dtype and device as below
                    return _compiled_wrapper()["_noisy_point"](z, torch.randn_like(z), sigma, self.norm_eps)
            return self.normalize_descent_state(z)
        v = self.normalize_descent_state(z)
        if noise is not None:
            return v + noise.to(device=v.device, dtype=v.dtype)
        if sigma is not None:
            return v + torch.randn_like(v) * sigma
        return v

    # -- dE/dz ---------------------------------------------------------------

    def _energy_gradient(self, v: torch.Tensor, x: torch.Tensor, cos_sin: Any, create_graph: bool) -> torch.Tensor:
        if not self.config.energy_compile:
            with second_order_window(create_graph):
                energy = self.energy(v, x, cos_sin)
            (grad,) = torch.autograd.grad(energy.sum(), v, create_graph=create_graph)
            return grad

        if self._compiled_energy_grad is None:
            self._compiled_energy_grad = self._build_compiled_energy_grad()
        compiled = self._compiled_energy_grad
        params = [p for _, p in self.energy.named_parameters()]
        if create_graph:
            # The attention second-order self-test must run eagerly, before dynamo traces the module.
            for module in self.energy.modules():
                if hasattr(module, "_ensure_verified"):
                    module._ensure_verified(v.device, v.dtype)
            if not self._compiled_verified:
                self._verify_compiled_energy_grad(compiled, v, x, cos_sin)
                self._compiled_verified = True
        # The attention block reads the tier flag at trace time, so each tier gets its own graph.
        with second_order_window(create_graph):
            if create_graph:
                return compiled(v, x, params, cos_sin)
            with torch.no_grad():
                return compiled(v, x, params, cos_sin)

    def _build_compiled_energy_grad(self) -> Callable[..., torch.Tensor]:
        energy = self.energy
        names = [n for n, _ in energy.named_parameters()]

        def energy_grad(z: torch.Tensor, x: torch.Tensor, params: list, cos_sin: Any) -> torch.Tensor:
            weights = dict(zip(names, params, strict=True))

            def total_energy(zz: torch.Tensor) -> torch.Tensor:
                return torch.func.functional_call(energy, weights, (zz, x), {"cos_sin": cos_sin}).sum()

            return torch.func.grad(total_energy)(z)

        _raise_recompile_limit()
        return torch.compile(energy_grad, backend="inductor", dynamic=False)

    def _verify_compiled_energy_grad(
        self, compiled: Callable[..., torch.Tensor], v: torch.Tensor, x: torch.Tensor, cos_sin: Any
    ) -> None:
        """Check that the compiled dE/dz matches eager to second order (float32, private RNG), or raise."""
        rows = min(2, v.shape[0])
        z0 = v.detach()[:rows].to(torch.float32)
        x0 = x.detach()[:rows].to(torch.float32)
        gen = torch.Generator(device=z0.device).manual_seed(1234)
        u = torch.randn(z0.shape, generator=gen, device=z0.device, dtype=z0.dtype)
        params = [p for _, p in self.energy.named_parameters()]

        def probe(grad_fn: Callable[[torch.Tensor], torch.Tensor]):
            z = z0.clone().requires_grad_(True)
            with torch.enable_grad(), second_order_window(True):
                g = grad_fn(z)
                hz, *ht = torch.autograd.grad((g * u).sum(), [z] + params, allow_unused=True)
            return g.detach(), hz, ht

        def eager_grad(z: torch.Tensor) -> torch.Tensor:
            return torch.autograd.grad(self.energy(z, x0, cos_sin).sum(), z, create_graph=True)[0]

        # Compare in true float32: TF32 eager and compiled matmuls differ at ~1e-3.
        prev_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            ref_g, ref_hz, ref_ht = probe(eager_grad)
            got_g, got_hz, got_ht = probe(lambda z: compiled(z, x0, params, cos_sin))
        finally:
            torch.set_float32_matmul_precision(prev_precision)

        def rel(a: torch.Tensor | None, r: torch.Tensor | None) -> float:
            if r is None:
                return 0.0 if a is None else float("inf")
            if a is None:
                return float("inf")
            return ((a.float() - r.float()).norm() / r.float().norm().clamp_min(1e-12)).item()

        errs = {
            "dE/dz": rel(got_g, ref_g),
            "d2E/dz2": rel(got_hz, ref_hz),
            "d2E/dtheta dz": max(rel(a, r) for a, r in zip(got_ht, ref_ht, strict=True)),
        }
        tol = 1e-3
        if not all(e < tol for e in errs.values()):
            raise RuntimeError(
                "energy_compile: compiled dE/dz does not match eager: "
                + ", ".join(f"{k}={e:.3e}" for k, e in errs.items())
                + f" (tolerance {tol}). Set arch.energy_compile=false to fall back to eager."
            )


# ----------------------------------------------------------------------------
# One segment
# ----------------------------------------------------------------------------


def _discard_init_draw(shape: tuple[int, ...], *, std: float, dtype: torch.dtype | None = None) -> None:
    """Advance the global RNG as initializing a ``shape`` tensor would, and drop the values."""
    trunc_normal_init_(torch.empty(shape, dtype=dtype), std=std)


class InnerNetwork(nn.Module):
    """Embeddings, decoder (lm_head) and the descent: runs one segment on a batch."""

    def __init__(self, config: EnergyConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        # Maze eval: draw fresh states for the whole batch at every reset and keep them on the reset rows.
        self.eval_full_batch_reset = False
        # Forward SXM: the K - 1 losing candidates of the last segment, the next segment's challengers. Kept out of
        # the state dict and saved with the training checkpoint (``explore_state_dict``).
        self._challenger_z: torch.Tensor | None = None

        # Module construction order fixes the parameter init draws; keep it.
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(
            config.vocab_size, config.hidden_size, init_std=1.0 / self.embed_scale, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        # Two random draws that initialize nothing (here and after ``self.level``); removing them changes the random
        # init of every parameter created after them.
        _discard_init_draw((2, config.hidden_size), std=1.0 / (config.hidden_size**0.5))
        head_dim = config.hidden_size // config.num_heads
        h, w = config.board_height, config.board_width
        if h * w != config.seq_len:
            raise ValueError(f"board {h}x{w} does not match seq_len {config.seq_len}")
        rope = RotaryTable2D if config.rope_impl == "fused" else RotaryEmbedding2D
        self.rotary_emb = rope(head_dim, h, w)
        self.level = EnergyDescent(config)
        _discard_init_draw((config.hidden_size,), std=config.init_std, dtype=self.forward_dtype)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_challenger_z"] = None  # the EMA copy never trains
        return state

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.lm_head(z)

    def cos_sin(self) -> Any:
        return self.rotary_emb()

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_scale * self.embed_tokens(input_ids.to(torch.int32))

    def empty_carry(self, batch_size: int, device: torch.device | None = None) -> LatentCarry:
        shape = (batch_size, self.config.seq_len, self.config.hidden_size)
        return LatentCarry(z=torch.empty(shape, dtype=self.forward_dtype, device=device))

    @torch.no_grad()
    def explore_state_dict(self) -> dict[str, Any] | None:
        return None if self._challenger_z is None else {"challenger_z": self._challenger_z.clone()}

    @torch.no_grad()
    def load_explore_state_dict(self, state: dict[str, Any] | None) -> None:
        if state and state.get("challenger_z") is not None:
            self._challenger_z = state["challenger_z"].clone()

    # -- initial states ---------------------------------------------------------

    def fresh_states(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        return trunc_normal_init_(torch.empty(shape, dtype=like.dtype, device=like.device), std=self.config.init_std)

    def reset_carry(self, reset: torch.Tensor, carry: LatentCarry) -> LatentCarry:
        """Give the rows in ``reset`` a fresh state drawn from TruncNormal(0, init_std)."""
        z = carry.z
        if self.eval_full_batch_reset and not self.training:
            fresh = trunc_normal_init_(torch.empty_like(z), std=self.config.init_std)
            return LatentCarry(z=torch.where(reset.to(device=z.device).view(-1, 1, 1), fresh, z))
        rows = reset.nonzero(as_tuple=False).flatten()
        if rows.numel():
            z[rows.to(device=z.device)] = self.fresh_states((rows.numel(),) + tuple(z.shape[1:]), z)
        return carry

    @torch.no_grad()
    def noisy_solution_init(self, z: torch.Tensor, labels: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Mix states ``z`` [n, S, H] with their solutions ``labels`` [n, S] by a [n] in [0, 1], on graded cells:

            z' = (I - P) z + sqrt(1 - a^2) P z + a s(y)

        P projects onto the directions the decoder (lm_head) reads and s(y) is a state that decodes to y, so a = 0
        keeps the noise and a = 1 decodes to the solution.
        """
        out_dtype = z.dtype
        z = z.to(torch.float32)
        w = self.lm_head.weight.detach().to(torch.float32)  # [V, H]
        pinv = torch.linalg.pinv(w)  # [H, V]
        project = pinv @ w
        rank = float(torch.linalg.matrix_rank(w).item())
        directions = pinv.T / pinv.T.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        directions = directions * (self.config.init_std * math.sqrt(rank))  # as long as P z of a fresh draw
        graded = labels != IGNORE_LABEL_ID
        a = a.view(-1, 1, 1) * graded.to(torch.float32).unsqueeze(-1)
        along = z @ project
        mixed = z - along + torch.sqrt(1.0 - a.square()) * along + a * directions[labels.clamp_min(0).long()]
        return mixed.to(out_dtype)

    @torch.no_grad()
    def solution_decode_accuracy(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Fraction of graded cells of each row that ``z`` decodes to its label."""
        graded = labels != IGNORE_LABEL_ID
        hits = ((self.decode(self.level.normalize_descent_state(z)).argmax(-1) == labels) & graded).sum(-1)
        return hits.to(torch.float32) / graded.sum(-1).clamp_min(1).to(torch.float32)

    # -- segment structure --------------------------------------------------------

    def segment_length(self, plan: SegmentPlan, device: torch.device) -> int:
        """Microsteps of a segment: all ``cycles`` at eval, U[randomize_min_cycles, cycles] in training."""
        if plan.cycles is not None:
            return plan.cycles
        cycles = self.config.cycles
        if not self.training or self.config.randomize_min_cycles >= cycles:
            return cycles
        return int(torch.randint(self.config.randomize_min_cycles, cycles + 1, (1,), device=device).item())

    def loop_length(self, cycles: int, plan: SegmentPlan, device: torch.device) -> tuple[int, torch.Tensor | None]:
        """Microsteps the loop runs, and each row's own count (rows that just reset run their own c_r)."""
        if not (self.training and plan.row_cycles is not None):
            return cycles, None
        row_cycles = plan.row_cycles.to(device=device, dtype=torch.long)
        row_cycles = torch.where(row_cycles > 0, row_cycles, torch.full_like(row_cycles, cycles))
        if plan.loop_plan is not None and plan.loop_plan[0] == cycles:
            return plan.loop_plan[1], row_cycles
        return max(cycles, int(row_cycles.max())), row_cycles

    def pinned_loop_plan(self, cycles: int, plan: SegmentPlan, device: torch.device) -> tuple[int, int, int] | None:
        """avoid_host_syncs: the loop length and the shortest row in one host read, reused by the whole segment."""
        if not (self.training and plan.row_cycles is not None):
            return None
        row_cycles = plan.row_cycles.to(device=device, dtype=torch.long)
        row_cycles = torch.where(row_cycles > 0, row_cycles, torch.full_like(row_cycles, cycles))
        longest, shortest = (int(v) for v in torch.stack([row_cycles.max(), row_cycles.min()]).tolist())
        loop = max(cycles, longest)
        return cycles, loop, min(max(shortest, 1), loop)

    def rollout(
        self,
        z: torch.Tensor,
        batch: dict[str, torch.Tensor],
        plan: SegmentPlan,
        *,
        noise: list[torch.Tensor] | None = None,
        resume_window_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Rollout]:
        """Run one segment from ``z``. Returns (endpoint, logits, convergence logit, rollout).

        A no-grad prefix of microsteps is followed by the graded window of ``grad_cycles``. A row that just reset with
        its own count c_r is frozen for the first loop - c_r microsteps (never inside the graded window).
        ``noise``: presampled noise, one tensor per microstep. ``resume_window_start``: skip the prefix and start the
        graded window from this state (the explore rollout already ran the same prefix with the same noise).
        """
        cos_sin = self.cos_sin()
        x = self.input_embeddings(batch["inputs"])
        loop, row_cycles = self.loop_length(self.segment_length(plan, z.device), plan, z.device)
        window = max(1, min(self.config.grad_cycles, loop))
        prefix = loop - window
        active = self._moving_rows(loop, row_cycles, plan, z.device)

        def step(state: torch.Tensor, t: int) -> torch.Tensor:
            return self.level(
                state,
                x,
                cos_sin,
                noise=None if noise is None else noise[t],
                sigma=plan.noise_scale,
                alpha=plan.step_size,
                active=active[t],
            )

        graded = self.training and torch.is_grad_enabled()
        if resume_window_start is not None:
            if tuple(resume_window_start.shape) != tuple(z.shape):
                raise RuntimeError("explore_reuse_prefix: resume state shape mismatch")
            if noise is None and (plan.noise_scale is not None or self.config.noise_scale):
                raise RuntimeError("explore_reuse_prefix: only replayed noise can be skipped")
            z = resume_window_start
        elif prefix:
            with torch.no_grad():
                for t in range(prefix):
                    z = step(z, t)
        result = Rollout(window_start=z.detach())
        moves = []
        for t in range(prefix, loop):
            z_next = step(z, t)
            if graded and self.config.microstep_ce:
                result.graded_states.append(z_next)
            moves.append((z, z_next) if graded else (z.detach(), z_next.detach()))
            z = z_next
        logits = self.decode(z).contiguous()
        return z.detach(), logits, self.convergence_logit(moves[-self.config.convergence_window :]), result

    @staticmethod
    def _moving_rows(
        loop: int, row_cycles: torch.Tensor | None, plan: SegmentPlan, device: torch.device
    ) -> list[torch.Tensor | None]:
        """Per microstep, the [B, 1, 1] mask of rows that move, or None when every row moves.

        A row with its own count c_r waits for the first loop - c_r microsteps, so all rows end together.
        """
        if row_cycles is None:
            return [None] * loop
        row_cycles = row_cycles.clamp(1, loop)
        if plan.loop_plan is not None and plan.loop_plan[1] == loop:
            shortest = plan.loop_plan[2]
        else:
            shortest = int(row_cycles.min())
        moving = torch.arange(loop, device=device).view(-1, 1) >= (loop - row_cycles).view(1, -1)
        return [moving[t].view(-1, 1, 1) if t < loop - shortest else None for t in range(loop)]

    def convergence_logit(self, moves: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """(convergence_threshold - r) / convergence_temperature, r the mean relative move of ``moves``. [B] float32."""
        relative = []
        for before, after in moves:
            b = before.to(torch.float32).flatten(1)
            a = after.to(torch.float32).flatten(1)
            # sqrt(. + eps) keeps the gradient finite for a move of exactly zero.
            relative.append(((a - b).square().sum(1) + 1e-12).sqrt() / (b.square().sum(1) + 1e-12).sqrt())
        r = torch.stack(relative).mean(0)
        return (self.config.convergence_threshold - r) / self.config.convergence_temperature

    # -- forward SXM ------------------------------------------------------------

    def explore(
        self,
        z: torch.Tensor,
        batch: dict[str, torch.Tensor],
        plan: SegmentPlan,
        reset: torch.Tensor,
        noisy_init_a: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Rollout, dict[str, torch.Tensor]]:
        """Forward SXM: roll out K candidates without a graph, keep each row's lowest-loss one, recompute it with a graph.

        Candidate 0 continues the row's state; candidates 1..K-1 continue the last segment's losers (fresh states
        for rows that just reset). All K run as one batch with presampled noise, and the winner's noise is replayed,
        so only one differentiable graph is built. The K - 1 losers' endpoints become the next challengers.
        """
        k = self.config.explore_k
        labels = batch["labels"]
        device = z.device
        rows = z.shape[0]
        shape = tuple(z.shape[1:])
        reset = reset.to(device=device)
        graded_cells = labels != IGNORE_LABEL_ID

        challengers = self._starting_challengers(z, labels, reset, noisy_init_a)
        candidates = torch.cat([z.detach()] + [challengers[i].clone() for i in range(k - 1)], dim=0)  # [K * B, S, H]

        cycles = self.segment_length(plan, device)
        plan = replace(plan, cycles=cycles)
        if self.config.avoid_host_syncs:
            plan.loop_plan = self.pinned_loop_plan(cycles, plan, device)
            loop = cycles if plan.loop_plan is None else plan.loop_plan[1]
        else:
            loop, _ = self.loop_length(cycles, plan, device)

        with torch.no_grad():
            sigma = plan.noise_scale
            sigma = torch.cat([sigma] * k, dim=0) if sigma is not None else (self.config.noise_scale or None)
            noise = None
            if sigma is not None:
                noise = [
                    torch.randn((k * rows,) + shape, device=device, dtype=torch.float32) * sigma for _ in range(loop)
                ]
            ends, logits, _, explored = self.rollout(
                candidates, {key: torch.cat([v] * k, dim=0) for key, v in batch.items()}, plan.tiled(k), noise=noise
            )
            per_cell = stablemax_cross_entropy(
                logits,
                torch.cat([labels] * k, dim=0),
                valid_mask=torch.cat([graded_cells] * k, dim=0),
            )
            denom = torch.cat([graded_cells.sum(-1).clamp_min(1)] * k, dim=0)
            losses = (per_cell.sum(-1) / denom).to(torch.float32).view(k, rows)
            winner = losses.min(dim=0).indices
            row = torch.arange(rows, device=device)
            pick = lambda t: t.view((k, rows) + shape)[winner, row]
            self._challenger_z = self._losers(ends.view((k, rows) + shape), losses)
            stats = self._explore_stats(losses, reset, rows)

        z_out, logits, convergence, rollout = self.rollout(
            pick(candidates),
            batch,
            plan,
            noise=None if noise is None else [pick(t) for t in noise],
            resume_window_start=pick(explored.window_start) if self.config.explore_reuse_prefix else None,
        )
        return z_out, logits, convergence, rollout, stats

    def _starting_challengers(
        self, z: torch.Tensor, labels: torch.Tensor, reset: torch.Tensor, noisy_init_a: torch.Tensor | None
    ) -> torch.Tensor:
        """Candidates 1..K-1 [K-1, B, S, H]: the benched losers, fresh states on the rows that just reset."""
        k = self.config.explore_k
        challengers = self._challenger_z
        if (
            challengers is None
            or tuple(challengers.shape) != (k - 1,) + tuple(z.shape)
            or challengers.device != z.device
        ):
            # Cold start (first segment, or a resume without saved challengers): copies of the incumbent.
            challengers = z.detach().unsqueeze(0).repeat((k - 1,) + (1,) * z.ndim).clone()
        else:
            challengers = challengers.to(dtype=z.dtype)
        rows = reset.nonzero(as_tuple=False).flatten()
        if rows.numel():
            fresh = self.fresh_states((k - 1, rows.numel()) + tuple(z.shape[1:]), z)
            if self.training and self.config.noisy_solution_init and noisy_init_a is not None:
                # The challengers of a reset row share its coefficient a.
                a = noisy_init_a.to(device=z.device)[rows]
                row_labels = labels.to(device=z.device)[rows]
                fresh = torch.stack([self.noisy_solution_init(f, row_labels, a) for f in fresh])
            challengers[:, rows] = fresh
        return challengers

    @staticmethod
    def _losers(ends: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
        """The K - 1 endpoints ``ends`` [K, B, ...] that did not win, in candidate order."""
        k, rows = losses.shape
        winner = losses.argmin(dim=0)
        is_winner = (torch.arange(k, device=losses.device).view(k, 1) == winner.view(1, -1)).to(torch.int8)
        keep = torch.argsort(is_winner, dim=0, stable=True)[: k - 1]
        index = keep.view(k - 1, rows, *([1] * (ends.dim() - 2))).expand((k - 1,) + tuple(ends.shape[1:]))
        return torch.gather(ends, 0, index).detach().clone()

    def _explore_stats(self, losses: torch.Tensor, reset: torch.Tensor, rows: int) -> dict[str, torch.Tensor]:
        """How often the carried candidate wins, and by how much the winner beats the runner-up (row sums)."""
        incumbent_wins = (losses.argmin(dim=0) == 0).to(torch.float32)
        continuing = ~reset
        if self.config.avoid_host_syncs:
            continuing_wins = (incumbent_wins * continuing.to(torch.float32)).sum()
        else:
            continuing_wins = incumbent_wins[continuing].sum()
        top2 = torch.topk(losses, 2, dim=0, largest=False).values
        return {
            "explore_incumbent_win_frac": incumbent_wins.mean() * rows,
            "explore_mid_incumbent_wins": continuing_wins,
            "explore_mid_rows": continuing.to(torch.float32).sum(),
            "explore_win_margin": (top2[1] - top2[0]).mean() * rows,
        }


# ----------------------------------------------------------------------------
# Segment wrapper
# ----------------------------------------------------------------------------


class EnergyReasoner(nn.Module):
    """Runs one segment on a batch: resets halted rows, descends, halts converged rows, and builds the interior loss.

    Returns the new carry and ``outputs``: logits, the convergence logit, and in training the interior loss
    (``aux_loss``), the reverse SXM endpoint targets (``graded_labels``) and telemetry (``aux_metrics``, row sums).
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = EnergyConfig(**config_dict)
        self.inner = InnerNetwork(self.config)

    def initial_carry(self, batch: dict[str, torch.Tensor]) -> ModelCarry:
        rows = batch["inputs"].shape[0]
        device = batch["inputs"].device
        return ModelCarry(
            inner_carry=self.inner.empty_carry(rows, device=device),
            steps=torch.zeros((rows,), dtype=torch.int32, device=device),
            halted=torch.ones((rows,), dtype=torch.bool, device=device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: ModelCarry, batch: dict[str, torch.Tensor]) -> tuple[ModelCarry, dict[str, Any]]:
        cfg = self.config
        inner = self.inner
        training = self.training
        device = batch["inputs"].device
        reset = carry.halted.to(device=device)
        metrics: dict[str, torch.Tensor] = {}

        # Draw the segment's settings (the order of the random draws below is part of the recipe).
        batch = self._with_solution_sets(carry, batch)
        plan = self._segment_plan(carry, batch, metrics)

        # New puzzles and fresh states on the reset rows.
        latent = inner.reset_carry(reset, LatentCarry(carry.inner_carry.z.to(device=device)))
        current_data = {k: v.to(device=batch[k].device) for k, v in carry.current_data.items()}
        reset_rows = reset.nonzero(as_tuple=False).flatten()
        if reset_rows.numel():
            for k, v in current_data.items():
                v[reset_rows] = batch[k][reset_rows]
        noisy_init_a = None
        if training and cfg.noisy_solution_init:
            noisy_init_a = self._noisy_solution_reset(latent.z, reset_rows, batch["labels"], metrics)
        steps = torch.where(reset, torch.zeros_like(carry.steps, device=device), carry.steps.to(device=device))

        if training and cfg.explore_k > 1:
            z, logits, convergence, rollout, explore_stats = inner.explore(
                latent.z, current_data, plan, carry.halted.detach().clone(), noisy_init_a
            )
            metrics = {**explore_stats, **metrics}
        else:
            z, logits, convergence, rollout = inner.rollout(latent.z, current_data, plan)

        with torch.no_grad():
            steps = steps + 1
            halted = self._halted(steps, convergence)
        new_carry = ModelCarry(LatentCarry(z), steps, halted, current_data)
        outputs: dict[str, Any] = {"logits": logits, "convergence_logit": convergence}

        if training:
            self._training_losses(outputs, metrics, logits, convergence, rollout.graded_states, current_data)
        if metrics:
            outputs["aux_metrics"] = metrics
        return new_carry, outputs

    def _with_solution_sets(self, carry: ModelCarry, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reverse SXM: carry each row's in-batch solution set (``rank_label_set``) next to its label.

        Outside reverse-SXM training a carry that already holds solution sets gets empty ones, so it keeps its layout.
        """
        cfg = self.config
        if self.training and cfg.late_nearest_ce_steps:
            sets = solution_sets(batch["inputs"], batch["labels"], cfg.explore_rank_max_completions)
            if "rank_label_set" not in carry.current_data:
                carry.current_data["rank_label_set"] = torch.full_like(sets, IGNORE_LABEL_ID)
            return {**batch, "rank_label_set": sets}
        if "rank_label_set" in carry.current_data and "rank_label_set" not in batch:
            return {**batch, "rank_label_set": torch.full_like(carry.current_data["rank_label_set"], IGNORE_LABEL_ID)}
        return batch

    def _halted(self, steps: torch.Tensor, convergence: torch.Tensor) -> torch.Tensor:
        """Rows that halt after this segment: at halt_max_steps, or (in training) once converged.

        In training a row runs at least a random number of segments in [2, halt_max_steps] with probability
        halt_exploration_prob.
        """
        cfg = self.config
        halted = steps >= cfg.halt_max_steps
        if self.training and cfg.halt_max_steps > 1:
            min_segments = (torch.rand_like(convergence) < cfg.halt_exploration_prob) * torch.randint_like(
                steps, low=2, high=cfg.halt_max_steps + 1
            )
            halted = (halted | (convergence > 0)) & (steps >= min_segments)
        return halted

    def _segment_plan(self, carry: ModelCarry, batch: dict[str, torch.Tensor], metrics: dict) -> SegmentPlan:
        """Noise level, step size and reset-row lengths of this segment."""
        cfg = self.config
        device = batch["inputs"].device
        rows = batch["inputs"].shape[0]
        plan = SegmentPlan()
        if self.training and cfg.training_noise_scale is not None:
            plan.noise_scale = torch.full((rows, 1, 1), cfg.training_noise_scale, device=device, dtype=torch.float32)
        elif cfg.noise_cosine:
            plan.noise_scale = self._cosine_noise_levels(carry, device)
        if self.training:
            draw = torch.rand((rows, 1, 1), device=device, dtype=torch.float32)
            plan.step_size = cfg.step_size_low + draw * (cfg.step_size_high - cfg.step_size_low)
            reset = carry.halted.to(device=device)
            draw = torch.randint(cfg.reset_cycles_low, cfg.reset_cycles_high + 1, (rows,), device=device)
            plan.row_cycles = torch.where(reset, draw, torch.zeros_like(draw))
            n_reset = reset.to(torch.float32).sum()
            mean_cycles = (draw.to(torch.float32) * reset.to(torch.float32)).sum() / n_reset.clamp_min(1.0)
            metrics.update(reset_cycles_frac=n_reset, reset_cycles_mean=mean_cycles * float(rows))
        return plan

    def _cosine_noise_levels(self, carry: ModelCarry, device: torch.device) -> torch.Tensor:
        """Per-row sigma: noise_settle_hot at segment 0, cosine down to noise_scale at the last segment. [B, 1, 1]"""
        steps = carry.steps.to(device=device)
        segment = torch.where(carry.halted.to(device=device), torch.zeros_like(steps), steps).to(torch.float32)
        total = self.config.halt_max_steps
        cold, hot = self.config.noise_scale, self.config.noise_settle_hot
        if total <= 1:
            ramp = torch.zeros_like(segment)
        else:
            ramp = 0.5 * (1.0 + torch.cos(math.pi * segment / float(total - 1)))
        return (cold + (hot - cold) * ramp).view(-1, 1, 1)

    @torch.no_grad()
    def _noisy_solution_reset(
        self, z: torch.Tensor, rows: torch.Tensor, labels: torch.Tensor, metrics: dict
    ) -> torch.Tensor:
        """Mix the reset rows of ``z`` (in place) with their solutions, a ~ U[0, 1] per row. Returns a [B]."""
        device = z.device
        a_all = torch.zeros((z.shape[0],), device=device, dtype=torch.float32)
        rows = rows.to(device=device)
        a_sum = acc_sum = torch.zeros((), device=device)
        if rows.numel():
            a = torch.rand((rows.numel(),), device=device, dtype=torch.float32)
            row_labels = labels.to(device=device)[rows]
            z[rows] = self.inner.noisy_solution_init(z[rows], row_labels, a)
            a_all[rows] = a
            a_sum = a.sum()
            acc_sum = self.inner.solution_decode_accuracy(z[rows], row_labels).sum()
        metrics.update(
            noisy_init_rows=torch.full((), float(rows.numel()), device=device),
            noisy_init_a_sum=a_sum,
            noisy_init_decode_acc=acc_sum,
        )
        return a_all

    def _training_losses(
        self,
        outputs: dict,
        metrics: dict,
        logits: torch.Tensor,
        convergence: torch.Tensor,
        graded_states: list[torch.Tensor],
        current_data: dict[str, torch.Tensor],
    ) -> None:
        """Reverse SXM endpoint targets, the interior CE of the graded microsteps (``aux_loss``), and telemetry."""
        cfg = self.config
        rows = logits.shape[0]
        labels = current_data["labels"].to(device=logits.device)

        sets = None
        has_set = None
        if cfg.late_nearest_ce_steps and "rank_label_set" in current_data:
            sets = current_data["rank_label_set"].to(device=logits.device)
            targets, has_set, moved = self._reverse_sxm(logits, sets, labels, current_data["inputs"])
            outputs["graded_labels"] = targets
            with torch.no_grad():
                metrics["explore_late_nearest_endpoint_changed"] = (
                    ((targets != labels).any(-1) & has_set).to(torch.float32).sum()
                )
                metrics["explore_late_nearest_endpoint_pool_displaced"] = (moved & has_set).to(torch.float32).sum()

        if cfg.microstep_ce and graded_states:
            per_step = self._graded_microstep_ce(
                graded_states, labels, sets, has_set, current_data["inputs"], rows, metrics
            )
            outputs["aux_loss"] = self._interior_loss(per_step, rows, metrics)

        with torch.no_grad():  # row sums: the residual r of the graded rollout and the converged rows
            c = convergence.detach().to(torch.float32)
            metrics["convergence_residual"] = (cfg.convergence_threshold - cfg.convergence_temperature * c).sum()
            metrics["converged_frac"] = (c > 0).to(torch.float32).sum()

    def _graded_microstep_ce(
        self,
        graded_states: list[torch.Tensor],
        labels: torch.Tensor,
        sets: torch.Tensor | None,
        has_set: torch.Tensor | None,
        inputs: torch.Tensor,
        rows: int,
        metrics: dict,
    ) -> list[torch.Tensor]:
        """The mean-cell CE of each graded microstep's decode. With reverse SXM (``sets``; ``has_set``: the rows that
        have one) the last late_nearest_ce_steps of them are graded against the nearest solution instead of the label.
        """
        n_late = min(self.config.late_nearest_ce_steps, len(graded_states)) if sets is not None else 0
        per_step, changed, moved_off = [], [], []
        for t, state in enumerate(graded_states):
            step_logits = self.inner.decode(state)
            step_labels = labels
            if t >= len(graded_states) - n_late:
                step_labels, step_has_set, moved = self._reverse_sxm(step_logits, sets, labels, inputs)
                changed.append(((step_labels != labels).any(-1) & step_has_set).to(torch.float32))
                moved_off.append((moved & step_has_set).to(torch.float32))
            per_step.append(self._mean_cell_ce(step_logits, step_labels))
        if sets is not None:
            with torch.no_grad():
                device = labels.device
                zero = torch.zeros((), device=device)
                metrics["explore_late_nearest_steps"] = torch.tensor(float(len(changed)), device=device) * rows
                metrics["explore_late_nearest_changed"] = torch.stack(changed).mean() * rows if changed else zero
                metrics["explore_late_nearest_known"] = has_set.to(torch.float32).mean() * rows
                metrics["explore_late_nearest_pool_displaced"] = (
                    torch.stack(moved_off).mean() * rows if moved_off else zero
                )
        return per_step

    def _reverse_sxm(
        self, logits: torch.Tensor, sets: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return reverse_sxm_targets(logits, sets, labels, inputs, self.config.late_nearest_pool_coverage)

    @staticmethod
    def _mean_cell_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """CE averaged over each row's graded cells, then over rows."""
        graded = labels != IGNORE_LABEL_ID
        per_cell = stablemax_cross_entropy(logits, labels, valid_mask=graded)
        return (per_cell / graded.sum(-1).clamp_min(1).unsqueeze(-1).to(torch.float32)).sum(-1).mean()

    def _interior_loss(self, per_step: list[torch.Tensor], rows: int, metrics: dict) -> torch.Tensor:
        """microstep_ce x the ramp-weighted mean of the graded microsteps' CE, as a row sum."""
        cfg = self.config
        stacked = torch.stack(per_step)
        mean = stacked.mean()
        if cfg.microstep_ce_ramp:
            w = torch.arange(1, len(per_step) + 1, dtype=torch.float32, device=stacked.device).pow(
                cfg.microstep_ce_ramp
            )
            weighted = (stacked * (w / w.sum())).sum()
        else:
            weighted = mean
        with torch.no_grad():
            metrics["microstep_ce_mean"] = mean.detach() * rows
            metrics["microstep_ce_first"] = per_step[0].detach() * rows
            metrics["microstep_ce_last"] = per_step[-1].detach() * rows
            if cfg.avoid_host_syncs:
                count = torch.full((), float(len(per_step)), dtype=torch.float32, device=stacked.device)
            else:
                count = torch.tensor(float(len(per_step)), device=stacked.device)
            metrics["microstep_ce_steps"] = count * rows
            if cfg.microstep_ce_ramp:
                metrics["microstep_ce_ramp_gap"] = (weighted.detach() - mean.detach()) * rows
            for t, ce in enumerate(per_step):
                metrics[f"microstep_ce_t{t + 1}"] = ce.detach() * rows
        return cfg.microstep_ce * weighted * rows


__all__ = [
    "EnergyConfig",
    "EnergyDescent",
    "EnergyReasoner",
    "InnerNetwork",
    "LatentCarry",
    "ModelCarry",
    "ScalarEnergy",
]
