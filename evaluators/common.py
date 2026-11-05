"""What the evaluation runners share: opening a checkpoint for evaluation (its config, the EMA weights), writing the
result files, and the segment-end energy / gradient-norm measurements."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset.common import read_metadata
from models import build_model
from utils.checkpoint import load_checkpoint, load_ema_shadow, load_model_state_dict, strip_orig_mod
from utils.config import config_from_dict

# Training-only speed flags; the evaluators turn them off.
EVAL_OFF_FLAGS = ("explore_reuse_prefix", "descent_wrapper_compile")
# Set by the trainer, not part of a run's recipe.
TRAIN_ONLY_DROP = ("load_checkpoint", "checkpoint_path")


def eval_config(ckpt: dict[str, Any], *, device: torch.device, halt_max_steps: int | None = None):
    """The checkpoint's training config, as the evaluator runs it: training-only speed flags off, and on CPU eager
    float32 (no compiled dE/dz)."""
    raw = {k: v for k, v in dict(ckpt["config"]).items() if k not in TRAIN_ONLY_DROP}
    raw["arch"] = dict(raw["arch"])
    for key in EVAL_OFF_FLAGS:
        if key in raw["arch"]:
            raw["arch"][key] = False
    if halt_max_steps is not None:
        raw["arch"]["halt_max_steps"] = int(halt_max_steps)
    config = config_from_dict(raw)
    if device.type == "cpu":
        config.arch.energy_compile = False
        config.arch.forward_dtype = "float32"
    return config


def check_eval_data(config, data_path: str):
    """The test split's metadata, after checking that it exists and has the checkpoint's board size."""
    if not os.path.isfile(os.path.join(data_path, "test", "dataset.json")):
        raise FileNotFoundError(f"no test split at {data_path} (expected {data_path}/test/dataset.json)")
    metadata = read_metadata(data_path, "test")
    h, w = config.arch.board_height, config.arch.board_width
    if metadata.seq_len != h * w:
        raise ValueError(
            f"{data_path} has {metadata.seq_len} cells per puzzle, but the checkpoint was trained on {h}x{w} boards "
            f"({config.dataset.name}): is --data_path the dataset the checkpoint was trained on?"
        )
    return metadata


def load_model(
    ckpt: dict[str, Any], config, metadata, *, device: torch.device, build_device: torch.device
) -> tuple[torch.nn.Module, int]:
    """Build on ``build_device`` (parameter init draws from that device's RNG), load the weights, then copy the EMA
    shadow into every tracked parameter. Returns (inference model in eval mode, EMA tensors)."""
    model = build_model(config, metadata, device=build_device, inference=True)
    load_model_state_dict(model, ckpt["model"], strict=True, assign=True)
    model.to(device)
    applied = 0
    shadow = load_ema_shadow(ckpt)
    if not shadow:
        raise RuntimeError("checkpoint has no EMA state")
    missing = []
    for name, p in model.named_parameters():
        canonical = strip_orig_mod(name)
        if canonical in shadow:
            p.data.copy_(shadow[canonical].to(p.device))
            applied += 1
        elif p.requires_grad:
            missing.append(canonical)
    if missing:
        raise RuntimeError(f"checkpoint has no EMA weights for {missing}")
    if applied != len(shadow):
        raise RuntimeError(f"EMA mismatch: applied {applied}/{len(shadow)} tensors")
    model.eval()
    if device.type == "cpu":
        model.model.config.energy_compile = False
    return model, applied


def write_result(out: str, arrays: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """``<out>.npz`` (the per-sample arrays) and ``<out>.json`` (``body``: meta and scored cells). Returns ``body``."""
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp.with_suffix(".npz"), **arrays)
    outp.with_suffix(".json").write_text(json.dumps(body, indent=1))
    return body


def apply_arch_overrides(model: torch.nn.Module, overrides: dict[str, Any]) -> None:
    """Eval-time settings (depth, noise, step size, init scale), written to the model's config."""
    config = model.model.config
    for key, value in overrides.items():
        if key not in type(config).model_fields:
            raise ValueError(f"unknown arch override {key!r}")
        setattr(config, key, value)


def tangential(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-token tangential part of g at v (radial component removed), float32."""
    g, v = g.float(), v.float()
    return g - (g * v).sum(-1, keepdim=True) / v.square().sum(-1, keepdim=True).clamp_min(1e-12) * v


def measure(level, z: torch.Tensor, x: torch.Tensor, cos_sin: Any, *, raw_energy: bool = False):
    """At v = normalize_descent_state(z): (E(v), |dE/dv|, |grad_M E| = tangential norm[, E(z) raw]) per row.

    Eager energy forward + backward, no noise, no randomness. Raises on a non-finite value, which would otherwise win
    the lowest-energy picks (argmin returns the first NaN)."""
    with torch.inference_mode(False), torch.enable_grad():
        v = level.normalize_descent_state(z.detach().clone()).detach().requires_grad_(True)
        e = level.energy(v, x, cos_sin)
        (g,) = torch.autograd.grad(e.sum(), v)
    flat = lambda t: t.detach().float().reshape(t.shape[0], -1).sum(1)
    v, g = v.detach(), g.detach()
    out = (flat(e), g.float().flatten(1).norm(dim=1), tangential(g, v).flatten(1).norm(dim=1))
    if raw_energy:
        with torch.no_grad():
            out = out + (flat(level.energy(z.detach(), x, cos_sin)),)
    if not all(bool(torch.isfinite(t).all()) for t in out):
        raise FloatingPointError("non-finite energy / gradient norm")
    return out


def measure_chunked(inner, carry, chunk: int):
    """``measure`` over row chunks of the carry (input embeddings recomputed per chunk), to bound memory. The chunk
    size sets the kernel shapes, so keep it fixed when comparing runs."""
    data = carry.current_data
    rows = int(carry.inner_carry.z.shape[0])
    step = int(chunk) or rows
    outs = [], [], []
    for s0 in range(0, rows, step):
        with torch.inference_mode(False), torch.enable_grad():
            inputs = data["inputs"][s0 : s0 + step].clone()
            x = inner.input_embeddings(inputs)
            res = measure(inner.level, carry.inner_carry.z[s0 : s0 + step], x, inner.cos_sin())
        for acc, r in zip(outs, res):
            acc.append(r)
    return tuple(torch.cat(o) for o in outs)


__all__ = [
    "apply_arch_overrides",
    "check_eval_data",
    "eval_config",
    "load_checkpoint",
    "load_model",
    "measure",
    "measure_chunked",
    "tangential",
    "write_result",
]
