"""Process-level helpers shared by the trainer and the evaluators: seeding, RNG isolation, moving nested state between
devices, and the torchrun (multi-GPU) collectives."""

from __future__ import annotations

import contextlib
import os
import random
from typing import Any

import numpy as np
import torch

# ----------------------------------------------------------------------------
# RNG
# ----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and every CUDA device), and PYTHONHASHSEED for child processes."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device


@contextlib.contextmanager
def isolated_rng(device: torch.device):
    """Save the torch CPU / all-CUDA-device / numpy / python RNG states (and PYTHONHASHSEED), restore them on exit."""
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    np_state, py_state = np.random.get_state(), random.getstate()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        np.random.set_state(np_state)
        random.setstate(py_state)
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed


# ----------------------------------------------------------------------------
# Nested state (dataclasses, dicts, lists of tensors)
# ----------------------------------------------------------------------------


def tree_map_tensors(obj: Any, fn) -> Any:
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return fn(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return type(obj)(**{k: tree_map_tensors(getattr(obj, k), fn) for k in obj.__dataclass_fields__})
    if isinstance(obj, dict):
        return {k: tree_map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(tree_map_tensors(v, fn) for v in obj)
    return obj


def tree_to_device(obj: Any, device) -> Any:
    return tree_map_tensors(obj, lambda t: t.to(device))


def tree_to_cpu_detached(obj: Any) -> Any:
    return tree_map_tensors(obj, lambda t: t.detach().cpu() if t.device.type != "cpu" else t.detach().clone())


# ----------------------------------------------------------------------------
# Distributed (torchrun). There is no DDP wrapper: gradients are summed by hand.
# ----------------------------------------------------------------------------


def dist_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def dist_world() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def init_distributed() -> tuple[int, int]:
    """Initialize NCCL from torchrun env vars. Returns (rank, world_size); (0, 1) when not distributed."""
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1))))
    return rank, world


def broadcast_model(model: torch.nn.Module) -> None:
    """Broadcast every parameter and buffer from rank 0 so all ranks start bit-identical."""
    import torch.distributed as dist

    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_floating_point() or t.dtype in (torch.int32, torch.int64, torch.uint8, torch.bool):
            dist.broadcast(t.data, src=0)


def allreduce_gradients(model: torch.nn.Module) -> None:
    """SUM param gradients across ranks (loss is already scaled by 1/global_batch_size)."""
    import torch.distributed as dist

    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def check_param_sync(model: torch.nn.Module) -> None:
    """Fail if the ranks' parameters differ (they stay in sync only through summed gradients)."""
    import torch.distributed as dist

    device = next(model.parameters()).device
    local = torch.zeros((), dtype=torch.float64, device=device)
    for p in model.parameters():
        local += p.detach().double().sum()
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    vals = [float(v) for v in gathered]
    if any(v != vals[0] for v in vals):
        raise RuntimeError(f"rank parameter checksums diverged: {vals}")


__all__ = [
    "allreduce_gradients",
    "broadcast_model",
    "check_param_sync",
    "dist_rank",
    "dist_world",
    "init_distributed",
    "isolated_rng",
    "seed_everything",
    "tree_map_tensors",
    "tree_to_cpu_detached",
    "tree_to_device",
]
