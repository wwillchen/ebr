"""Sudoku-Extreme protocol: every test board, ``breadth`` samples per board, ``depth`` segments, in shards.

Each sample is one trajectory from a fresh random state, run for ``depth`` segments without halting. The scores
(``protocol.sudoku_cells``) read the segment ends: pass@1 is sample 0, pass@n the lowest-energy of the first n.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from dataset.puzzle_dataset import create_dataloader
from evaluators import protocol
from evaluators.common import check_eval_data, eval_config, load_checkpoint, load_model, measure, write_result
from models.losses import IGNORE_LABEL_ID
from utils.runtime import seed_everything, tree_to_device

RECORDED = ("exact", "energy", "gnorm", "gtan", "convergence_logit")  # per sample and segment


def run(
    checkpoint: str,
    data_path: str,
    out: str,
    *,
    breadth: int = 8,
    depth: int = 64,
    batch: int = 4096,
    shard: int = 0,
    num_shards: int = 1,
    max_boards: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    breadth = max(1, int(breadth))
    dev = torch.device(device)
    torch.set_float32_matmul_precision("high")
    ckpt = load_checkpoint(checkpoint, map_location=dev)
    config = eval_config(ckpt, device=dev)
    train_depth = config.arch.halt_max_steps
    depth = max(int(depth), train_depth)
    boards_per_batch = max(1, batch // breadth)
    config.global_batch_size = boards_per_batch
    config.dataset.data_path = data_path

    # This shard's range of loader batches.
    num_boards = int(check_eval_data(config, data_path).total_samples)
    if max_boards:
        num_boards = min(num_boards, int(max_boards))
    num_batches = math.ceil(num_boards / boards_per_batch)
    first, end = shard * num_batches // num_shards, (shard + 1) * num_batches // num_shards

    seed_everything(config.seed + shard)
    loader, metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=boards_per_batch
    )
    model, _ = load_model(ckpt, config, metadata, device=dev, build_device=dev)

    recorded = {k: [] for k in RECORDED}
    board_ids = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, (_, data, _) in enumerate(loader):
            if i < first:
                continue
            if i >= end:
                break
            data = tree_to_device(data, dev)
            if breadth > 1:
                data = {k: v.repeat_interleave(breadth, dim=0) for k, v in data.items()}
            board = torch.arange(data["inputs"].shape[0], device=dev) // breadth + i * boards_per_batch
            keep = data["labels"].ne(IGNORE_LABEL_ID).any(-1) & (board < num_boards)  # drop the last batch's padding
            per_segment = _rollout_batch(model, data, depth=depth, train_depth=train_depth)
            for k in RECORDED:
                recorded[k].append(per_segment[k][:, keep].cpu())
            board_ids.append(board[keep].cpu())
            print(f"batch {i - first + 1}/{end - first}", flush=True)

    arrays = {k: torch.cat(v, dim=1).numpy() for k, v in recorded.items()}
    arrays["board"] = torch.cat(board_ids).numpy().astype(np.int32)
    arrays["exact"] = arrays["exact"].astype(np.uint8)
    depths = sorted({d for d in (train_depth, 16, depth) if d <= depth})
    counts = protocol.sudoku_cells(arrays["exact"], arrays["energy"], arrays["board"], breadth, depths=depths)
    meta = {
        "checkpoint": checkpoint,
        "step": int(ckpt.get("step", -1)),
        "seed": config.seed,
        "shard": shard,
        "num_shards": num_shards,
        "batch_range": [first, end],
        "breadth": breadth,
        "depth": depth,
        "train_halt": train_depth,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return write_result(out, arrays, {"meta": meta, "counts": counts})


def _rollout_batch(
    model: torch.nn.Module, data: dict[str, torch.Tensor], *, depth: int, train_depth: int
) -> dict[str, torch.Tensor]:
    """Run ``depth`` segments on one batch, every row to the end. Returns each RECORDED quantity as (depth, rows)."""
    inner = model.model.inner
    labels = data["labels"].to(torch.long)  # scoring only
    cells = labels.ne(IGNORE_LABEL_ID)
    x = inner.input_embeddings(data["inputs"])
    cos_sin = inner.cos_sin()

    carry = model.initial_carry(data)
    per_segment = {k: [] for k in RECORDED}
    for segment in range(depth):
        if segment >= train_depth:  # past the trained depth: keep every row running
            carry.halted = torch.zeros_like(carry.halted)
            carry.steps = torch.full_like(carry.steps, train_depth - 1)
        if segment and bool(carry.halted.any()):
            raise RuntimeError("a row would be reset mid-evaluation")
        carry, _, _, outputs, _ = model(carry=carry, batch=data, return_keys=["preds", "convergence_logit"])
        energy, gnorm, gtan = measure(inner.level, carry.inner_carry.z, x, cos_sin)
        per_segment["exact"].append((outputs["preds"].to(torch.long).eq(labels) | ~cells).all(-1))
        per_segment["energy"].append(energy)
        per_segment["gnorm"].append(gnorm)
        per_segment["gtan"].append(gtan)
        per_segment["convergence_logit"].append(outputs["convergence_logit"].float().reshape(-1))
    return {k: torch.stack(v) for k, v in per_segment.items()}
