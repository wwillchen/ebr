"""Maze-Hard protocol: every test maze x ``n`` samples in one batch, ``depth`` segments without halting.

The eval noise follows the training noise: a flat-noise checkpoint is evaluated with the same flat sigma, a cosine one
with the cosine schedule over ``depth`` segments. Scores (``protocol.maze_cells``) are read at segments 8 and ``depth``:
start 1 is sample 0, pass@n the lowest-energy of the n samples.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from dataset.puzzle_dataset import create_dataloader
from evaluators import protocol
from evaluators.common import (
    apply_arch_overrides,
    check_eval_data,
    eval_config,
    load_checkpoint,
    load_model,
    measure_chunked,
    write_result,
)
from evaluators.maze_path import score_maze
from models.losses import IGNORE_LABEL_ID
from utils.runtime import tree_to_device

EVAL_SETTINGS = {"noise_scale": 0.05, "init_std": 1.0, "step_size": 250.0}
RECORDED = ("gtan", "gnorm", "energy", "convergence_logit", "exact", "shortest", "valid", "ans")


def run(
    checkpoint: str,
    data_path: str,
    out: str,
    *,
    n: int = 5,
    depth: int = 16,
    batch: int = 1000,
    grad_chunk: int = 2500,
    draw_seed: int | str | None = None,
    flat_noise: float | str | None = "auto",
    device: str = "cuda",
    max_batches: int = 0,
) -> dict[str, Any]:
    """``flat_noise``: "auto" uses the checkpoint's training_noise_scale if it has one (flat-trained), else cosine.
    ``draw_seed``: seed of the initial states and noise of loader batch i is draw_seed + i; None = the training seed.
    ``grad_chunk``: rows per energy and gradient-norm measurement."""
    dev = torch.device(device)
    torch.set_float32_matmul_precision("high")
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    config = eval_config(ckpt, device=dev)
    train_seed = config.seed
    if flat_noise == "auto":
        flat_noise = config.arch.__pydantic_extra__.get("training_noise_scale")
    draw_seed = train_seed if draw_seed in (None, "auto") else int(draw_seed)
    config.global_batch_size = batch
    config.dataset.data_path = data_path
    check_eval_data(config, data_path)

    loader, metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=batch)
    model, ema_tensors = load_model(ckpt, config, metadata, device=dev, build_device=dev)
    depth = int(depth)
    settings = {**EVAL_SETTINGS, "halt_max_steps": depth}
    if flat_noise is not None:
        settings.update(noise_cosine=False, noise_scale=float(flat_noise))
    apply_arch_overrides(model, settings)
    model.model.inner.eval_full_batch_reset = True
    step = int(ckpt.get("step", 0))
    del ckpt

    hash_vector = None
    recorded = {k: [] for k in RECORDED}
    mazes = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, (_, batch_data, num_mazes) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch_data = tree_to_device(batch_data, dev)
            data = {k: v.repeat_interleave(n, dim=0) for k, v in batch_data.items()} if n > 1 else batch_data
            if hash_vector is None:
                hash_vector = _answer_hash_vector(data["labels"].shape[1], dev)
            torch.manual_seed(draw_seed + i)
            per_segment = _rollout_batch(
                model, data, rows=int(num_mazes) * n, depth=depth, grad_chunk=grad_chunk, hash_vector=hash_vector
            )
            for k in RECORDED:
                recorded[k].append(per_segment[k])
            mazes += int(num_mazes)

    arrays = {k: np.concatenate(v, axis=1) for k, v in recorded.items()}
    arrays = {k: v.astype(np.float32) if v.dtype.kind == "f" else v for k, v in arrays.items()}
    cells = {f"seg{s}": protocol.maze_cells(arrays, n=n, segment=s) for s in sorted({8, depth}) if s <= depth}
    meta = {
        "checkpoint": checkpoint,
        "step": step,
        "train_seed": train_seed,
        "draw_seed": draw_seed,
        "flat_noise": flat_noise,
        "depth": depth,
        "n": n,
        "batch": batch,
        "grad_chunk": grad_chunk,
        "ema_tensors": ema_tensors,
        "mazes": mazes,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return write_result(out, arrays, {"meta": meta, "cells": cells})


def _answer_hash_vector(seq_len: int, device: torch.device) -> torch.Tensor:
    """Fixed random weights; ``(preds * w).sum(1)`` is a hash of each decoded maze (``ans``)."""
    generator = torch.Generator().manual_seed(1234)
    return torch.randint(1, 2**31 - 1, (seq_len,), generator=generator, dtype=torch.int64).to(device)


def _rollout_batch(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    *,
    rows: int,
    depth: int,
    grad_chunk: int,
    hash_vector: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Run ``depth`` segments on one batch. Returns each RECORDED quantity as a (depth, rows) array.

    The first ``rows`` rows are real (the rest pad the last batch). After every segment: energy and gradient norms at
    the state, the convergence logit, the exact match, and the path scores of the decoded maze.
    """
    inner = model.model.inner
    labels = data["labels"].long()  # scoring only
    cells = labels.ne(IGNORE_LABEL_ID)
    inputs_np = data["inputs"][:rows].cpu().numpy()
    labels_np = labels[:rows].cpu().numpy()

    carry = model.initial_carry(data)
    per_segment = {k: [] for k in RECORDED}
    prev_hash, prev_scores = None, None
    for segment in range(depth):
        carry, _, _, outputs, finished = model(carry=carry, batch=data, return_keys={"preds", "convergence_logit"})
        if bool(finished) != (segment == depth - 1):
            raise RuntimeError(f"segment {segment + 1}: finished={bool(finished)}")
        energy, gnorm, gtan = measure_chunked(inner, carry, grad_chunk)
        preds = outputs["preds"].long()
        answer_hash = (preds * hash_vector).sum(1)[:rows].cpu().numpy()
        # Path scoring is slow; rescore only the rows whose decoded maze changed.
        scores = np.zeros((rows, 2), bool) if prev_scores is None else prev_scores.copy()
        changed = np.arange(rows) if prev_hash is None else np.nonzero(answer_hash != prev_hash)[0]
        preds_np = preds[:rows].cpu().numpy()
        for r in changed:
            s = score_maze(inputs_np[r], preds_np[r], labels_np[r])
            scores[r] = (s["shortest_path"] > 0, s["valid_path"] > 0)
        prev_hash, prev_scores = answer_hash, scores
        per_segment["gtan"].append(gtan[:rows].cpu().numpy())
        per_segment["gnorm"].append(gnorm[:rows].cpu().numpy())
        per_segment["energy"].append(energy[:rows].cpu().numpy())
        per_segment["convergence_logit"].append(outputs["convergence_logit"].float().reshape(-1)[:rows].cpu().numpy())
        per_segment["exact"].append((preds.eq(labels) | ~cells).all(-1)[:rows].cpu().numpy())
        per_segment["shortest"].append(scores[:, 0])
        per_segment["valid"].append(scores[:, 1])
        per_segment["ans"].append(answer_hash)
    return {k: np.stack(v) for k, v in per_segment.items()}
