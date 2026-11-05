"""In-training evaluation: run every test batch to the last segment and average the loss head's metrics, plus the
dataset's extra metrics (``config.dataset.evaluators``, e.g. Maze-Hard path validity)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch
import tqdm

from evaluators.maze_path import EVALUATORS as MAZE_EVALUATORS
from models.losses import EXTREMUM_METRICS
from utils.runtime import tree_to_device

EVALUATORS = {**MAZE_EVALUATORS}


def create_evaluators(config: Any, metadata: Any) -> list[Any]:
    """The dataset's extra evaluators (``config.dataset.evaluators``) for the test split."""
    return [
        EVALUATORS[e.name](data_path=config.dataset.data_path, eval_metadata=metadata, **e.__pydantic_extra__)
        for e in config.dataset.evaluators
    ]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: Any, metadata: Any, evaluators: Sequence[Any] = (), *, progress: bool = True
) -> tuple[dict[str, Any], float]:
    """Run ``model`` (a loss head in eval mode) over ``loader``. Returns (metrics per set, seconds).

    Metrics are those of the last segment, averaged over the scored rows, plus ``residual_of_<k>_steps``: the summed
    distance the state moved in segment k.
    """
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    return_keys = set()
    for evaluator in evaluators:
        evaluator.begin_eval()
        return_keys.update(evaluator.required_outputs)

    sums: dict[str, dict[str, torch.Tensor]] = {name: {} for name in metadata.sets}
    extrema: dict[str, dict[str, float]] = {name: {} for name in metadata.sets}
    total = getattr(loader.dataset, "num_batches", lambda: None)()
    carry = None
    with torch.inference_mode():
        for set_name, batch, _ in tqdm.tqdm(loader, total=total, desc="Eval", disable=not progress, leave=False):
            batch = tree_to_device(batch, device)
            carry = model.initial_carry(batch) if carry is None else tree_to_device(carry, device)
            carry, metrics, outputs = _run_to_last_segment(model, carry, batch, return_keys)
            _accumulate(sums[set_name], extrema[set_name], metrics, device)
            for evaluator in evaluators:
                evaluator.update_batch(batch, outputs or {})

    results: dict[str, Any] = {}
    for name in metadata.sets:
        row = {k: float(v) for k, v in sums[name].items()}
        count = row.pop("count", 0.0)
        results[name] = {k: v / (count if count > 0 else 1.0) for k, v in row.items()}
        results[name].update(extrema[name])
    for evaluator in evaluators:
        extra = evaluator.result()
        if extra:
            results.update(extra)
    if device.type == "cuda":
        gib = float(1024**3)
        results["runtime/cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / gib
        results["runtime/cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / gib
        torch.cuda.synchronize(device)
    return results, time.perf_counter() - start


def _run_to_last_segment(model, carry, batch, return_keys):
    """Segments until every row has run ``halt_max_steps``. Returns the carry, the last segment's metrics (with the
    per-segment residuals added, cast to the dtype of ``count``) and its outputs. The carry goes on to the next batch,
    whose rows all start fresh because every row halted."""
    residuals = []
    previous = carry.inner_carry.z
    while True:
        carry, _loss, metrics, outputs, finished = model(carry=carry, batch=batch, return_keys=return_keys)
        current = carry.inner_carry.z
        residuals.append((current - previous).flatten(1).to(torch.float32).norm(dim=1).sum())
        previous = current
        if bool(finished):
            break
    metrics = dict(metrics)
    for k, residual in enumerate(residuals):
        metrics[f"residual_of_{k + 1}_steps"] = residual.to(metrics["count"])
    return carry, metrics, outputs


def _accumulate(sums: dict, extrema: dict, metrics: dict[str, torch.Tensor], device: torch.device) -> None:
    """Add a batch's row-summed metrics to ``sums``; extrema keep their running max / min."""
    for k, v in metrics.items():
        if k in EXTREMUM_METRICS:
            value, is_max = float(v), EXTREMUM_METRICS[k] == "max"
            current = extrema.get(k, float("-inf") if is_max else float("inf"))
            extrema[k] = max(current, value) if is_max else min(current, value)
        else:
            v32 = v.detach().to(device, torch.float32)
            sums[k] = sums[k] + v32 if k in sums else v32


__all__ = ["EVALUATORS", "create_evaluators", "evaluate"]
