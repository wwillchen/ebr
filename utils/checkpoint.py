"""Training state, EMA, and checkpoint save / load / resume.

A checkpoint is one ``step_<N>_<run_id>.pth`` holding the model, the optimizers, the EMA weights, the data sampler
position, the carried batch state (``carry``, ``explore``) and the RNG states, so a resumed run picks up where it
stopped. With several ranks, each rank also writes ``step_<N>_<run_id>.rank<r>of<world>.pth`` with its own
carry, challengers and RNG.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
import shutil
import types
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from utils.runtime import dist_rank, dist_world, tree_to_cpu_detached, tree_to_device


class EMAHelper:
    """Exponential moving average of the trainable parameters, by name."""

    def __init__(self, mu: float = 0.999) -> None:
        self.mu = mu
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, module: nn.Module) -> None:
        self.shadow = {n: p.data.clone() for n, p in module.named_parameters() if p.requires_grad}

    def update(self, module: nn.Module) -> None:
        for n, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[n].data = (1.0 - self.mu) * p.data + self.mu * self.shadow[n].data

    def ema_copy(self, module: nn.Module) -> nn.Module:
        """A copy of ``module`` with the EMA weights."""
        module_copy = copy.deepcopy(module)
        for n, p in module_copy.named_parameters():
            if p.requires_grad and n in self.shadow:
                p.data.copy_(self.shadow[n].data)
        return module_copy

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = state


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int
    steps_per_epoch: int
    ema_helper: EMAHelper
    grad_clip: Any = None  # utils.optim.RelativeGradClip or AdaptiveGradClip when the config enables one


def _reasoner_inner(model: nn.Module) -> nn.Module:
    """The ``InnerNetwork`` of a loss head (``model.model.inner``), through a ``torch.compile`` wrapper if any."""
    return getattr(model, "_orig_mod", model).model.inner


def strip_orig_mod(name: str) -> str:
    """A parameter name without the ``_orig_mod`` components ``torch.compile`` adds."""
    return ".".join(p for p in name.split(".") if p != "_orig_mod")


def load_model_state_dict(
    model: nn.Module, state_dict: dict[str, torch.Tensor], *, strict: bool, assign: bool = False
) -> None:
    """Load weights, with or without ``torch.compile``'s ``_orig_mod`` prefixes on either side."""
    names = {strip_orig_mod(k): k for k in model.state_dict()}
    state_dict = {names.get(strip_orig_mod(k), strip_orig_mod(k)): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=strict, assign=assign)
    if not strict and (result.missing_keys or result.unexpected_keys):
        print(f"[checkpoint] missing keys: {result.missing_keys}; unexpected keys: {result.unexpected_keys}")


def load_ema_shadow(ckpt: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """The EMA weights of a checkpoint by parameter name, or None if it has none."""
    ema = ckpt.get("ema")
    if ema is None:
        return None
    shadow = ema.get("shadow") if isinstance(ema, dict) else None
    if shadow is None and isinstance(ema, dict):  # the shadow tensors at the top level
        shadow = {k: v for k, v in ema.items() if isinstance(v, torch.Tensor)}
    return {strip_orig_mod(k): v for k, v in (shadow or {}).items()}


def _rng_states(*, all_devices: bool) -> dict[str, Any]:
    """The torch, CUDA, numpy and python RNG states. ``all_devices``: every CUDA device, else only the current one."""
    cuda = None
    if torch.cuda.is_available():
        cuda = torch.cuda.get_rng_state_all() if all_devices else torch.cuda.get_rng_state()
    rng = {"torch": torch.random.get_rng_state(), ("cuda" if all_devices else "cuda_current"): cuda}
    return {"rng": rng, "numpy": np.random.get_state(), "python": random.getstate()}


def _as_byte_tensor(state: Any) -> torch.Tensor:
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state)
    elif not isinstance(state, torch.Tensor):
        state = torch.tensor(list(state), dtype=torch.uint8)
    return state.detach().to(device="cpu", dtype=torch.uint8, copy=True).contiguous()


def restore_rng(rng: dict[str, Any] | None) -> None:
    """Restore the torch and CUDA RNG states saved in a checkpoint's ``rng`` entry."""
    if not rng:
        return
    if rng.get("torch") is not None:
        torch.random.set_rng_state(_as_byte_tensor(rng["torch"]))
    if not torch.cuda.is_available():
        return
    if rng.get("cuda_current") is not None:  # a rank's own device
        torch.cuda.set_rng_state(_as_byte_tensor(rng["cuda_current"]))
    elif rng.get("cuda") is not None:
        states = rng["cuda"] if isinstance(rng["cuda"], (list, tuple)) else [rng["cuda"]]
        torch.cuda.set_rng_state_all([_as_byte_tensor(s) for s in states])


# ----------------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------------


def _atomic_save(obj: dict[str, Any], path: str) -> None:
    """Write to a temporary file next to ``path``, then rename it, so a preempted save never leaves a partial file."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(
    config: Any, state: TrainState, train_loader: Any, *, run_id: str | None, rank: int = 0, world: int = 1
) -> str:
    """Save the training state at ``state.step``. With several ranks every rank writes its own file first, and rank 0
    writes the main file only after all of them succeeded. Returns the main file's path."""
    name = f"step_{state.step}" + (f"_{run_id}" if run_id else "")
    path = os.path.join(config.checkpoint_path, f"{name}.pth")
    os.makedirs(config.checkpoint_path, exist_ok=True)
    explore = _reasoner_inner(state.model).explore_state_dict()

    if world > 1:
        rank_state = {
            "step": state.step,
            "world": world,
            "rank": rank,
            "carry": tree_to_cpu_detached(state.carry),
            **_rng_states(all_devices=False),
        }
        if explore is not None:
            rank_state["explore"] = tree_to_cpu_detached(explore)
        _on_every_rank(
            lambda: _atomic_save(rank_state, os.path.join(config.checkpoint_path, f"{name}.rank{rank}of{world}.pth")),
            what="rank checkpoint",
            world=world,
        )

    def save_main() -> None:
        if rank != 0:
            return
        ckpt: dict[str, Any] = {
            "step": state.step,
            "total_steps": state.total_steps,
            "model": state.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in state.optimizers],
            "optimizer_lrs": list(state.optimizer_lrs),
            "carry": tree_to_cpu_detached(state.carry),
            **_rng_states(all_devices=True),
            "train_dataset": train_loader.dataset.state_dict(),
            "config": config.model_dump(),
            "wandb_run_id": run_id,
        }
        if explore is not None:
            ckpt["explore"] = tree_to_cpu_detached(explore)
        if state.ema_helper is not None:
            ckpt["ema"] = {"mu": state.ema_helper.mu, "shadow": tree_to_cpu_detached(state.ema_helper.state_dict())}
        if state.grad_clip is not None:
            ckpt["grad_clip"] = tree_to_cpu_detached(state.grad_clip.state_dict())
        _atomic_save(ckpt, path)

    _on_every_rank(save_main, what="checkpoint", world=world)
    return path


def _on_every_rank(action, *, what: str, world: int) -> None:
    """Run ``action``; with several ranks, raise on every rank if it failed on any (so none waits at a barrier)."""
    if world == 1:
        action()
        return
    import torch.distributed as dist

    error = None
    try:
        action()
    except Exception as exc:  # reported to every rank below
        error = exc
    statuses = [None] * world
    dist.all_gather_object(
        statuses, None if error is None else f"rank {dist.get_rank()}: {type(error).__name__}: {error}"
    )
    failures = [s for s in statuses if s is not None]
    if failures:
        raise RuntimeError(f"saving the {what} failed: " + " | ".join(failures)) from error


# ----------------------------------------------------------------------------
# Load and resume
# ----------------------------------------------------------------------------


def load_checkpoint(path: str, map_location: Any) -> dict[str, Any]:
    """A checkpoint in the current layout (other layouts are converted, see ``_upgrade_checkpoint``)."""
    return _upgrade_checkpoint(torch_load(path, map_location), where=path)


def prime_attention_self_test(model: nn.Module) -> None:
    """Run the attention second-order self-test now, under a forked RNG.

    The self-test reseeds the global RNG. The trainer runs it before training (on resume, before the checkpointed RNG
    is restored), so it never touches the training draws.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    modules = [m for m in model.modules() if hasattr(m, "_ensure_verified")]
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        for m in modules:
            m._ensure_verified(device, torch.float32)


def seed_attention_self_test(model: nn.Module, seed: int) -> None:
    """Make the attention self-test, still to run at the first graded microstep, reseed the global RNG with ``seed``."""
    for m in model.modules():
        if hasattr(m, "self_test_seed"):
            m.self_test_seed = int(seed)


def load_training_state(config: Any, state: TrainState, train_loader: Any) -> dict[str, Any] | None:
    """Resume from ``config.load_checkpoint``: model, optimizers, EMA, carry, challengers, sampler and RNG.

    Returns ``wandb_run_id`` and ``rng_fix`` (see below), or None without a checkpoint.
    """
    if config.load_checkpoint is None:
        return None
    print(f"Loading checkpoint {config.load_checkpoint}")
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = load_checkpoint(config.load_checkpoint, map_location=map_location)
    device = next(state.model.parameters()).device

    load_model_state_dict(state.model, ckpt["model"], strict=True)
    for opt, opt_state in zip(state.optimizers, ckpt.get("optimizers", []), strict=False):
        opt.load_state_dict(opt_state)
    state.optimizer_lrs = tuple(ckpt.get("optimizer_lrs", state.optimizer_lrs))
    state.step = int(ckpt.get("step", state.step))
    if config.max_steps is not None:
        state.total_steps = min(state.total_steps, config.max_steps)
        state.step = min(state.step, state.total_steps)

    local = _rank_local_state(config.load_checkpoint, ckpt, state.step, map_location)
    if local.get("carry") is not None:
        state.carry = tree_to_device(local["carry"], device)
    _restore_all_rngs(local)

    sampler_state = ckpt.get("train_dataset")
    if sampler_state is not None:
        train_loader.dataset.load_state_dict(sampler_state)

    explore = (local.get("explore") if local is not ckpt else None) or ckpt.get("explore")
    if explore is not None:
        _reasoner_inner(state.model).load_explore_state_dict(tree_to_device(explore, device))
        print("Loaded explore-K challengers from checkpoint.")

    shadow = load_ema_shadow(ckpt)
    if shadow:
        state.ema_helper.load_state_dict(tree_to_device(shadow, device))
        print("Loaded EMA shadow from checkpoint.")
    elif shadow is not None:
        state.ema_helper.register(state.model)

    if state.grad_clip is not None:
        if ckpt.get("grad_clip"):
            state.grad_clip.load_state_dict(tree_to_device(ckpt["grad_clip"], device))
        else:
            print("Gradient clipping is new in this run: its running average starts from the first step's norm.")

    saved_config = ckpt.get("config") or {}
    config.run_name = saved_config.get("run_name", config.run_name)
    config.project_name = saved_config.get("project_name", config.project_name)
    prime_attention_self_test(state.model)
    mid_pass = isinstance(sampler_state, dict) and bool(sampler_state.get("train_state"))
    return {
        "wandb_run_id": ckpt.get("wandb_run_id"),
        # Mid-pass, recreating the loader iterator draws one seed from the global RNG, so the trainer restores this
        # RNG state again right after its first ``iter()``.
        "rng_fix": local.get("rng") if mid_pass else None,
    }


def _rank_local_state(path: str, ckpt: dict[str, Any], step: int, map_location: Any) -> dict[str, Any]:
    """The part of the checkpoint that differs per rank: carry, challengers and RNG.

    With one rank it is in the checkpoint itself; with several, each rank has its own file and the run must resume with
    the same number of ranks.
    """
    world = dist_world()
    if world == 1:
        return ckpt
    rank_path = path.replace(".pth", f".rank{dist_rank()}of{world}.pth")
    if not os.path.exists(rank_path):
        raise FileNotFoundError(f"resuming on {world} ranks needs {rank_path} (saved with another number of ranks?)")
    local = torch_load(rank_path, map_location=map_location)
    if int(local.get("step", -1)) != step:
        raise ValueError(f"{rank_path} is at step {local.get('step')}, the checkpoint at {step}")
    return local


def _restore_all_rngs(local: dict[str, Any]) -> None:
    """Torch (CPU and CUDA), numpy and python RNG states, each if saved."""
    restore_rng(local.get("rng"))
    if local.get("numpy") is not None:
        np.random.set_state(local["numpy"])
    if local.get("python") is not None:
        random.setstate(local["python"])


def snapshot_code(run_dir: str, repo_root: str) -> None:
    """Copy the code and configs next to the checkpoints."""
    code_dir = os.path.join(run_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for item in ("models", "utils", "dataset", "evaluators", "config", "pretrain.py", "evaluate.py"):
        src, dst = os.path.join(repo_root, item), os.path.join(code_dir, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=ignore)
        elif os.path.isfile(src):
            shutil.copy(src, dst)


# ----------------------------------------------------------------------------
# Other checkpoint layouts
#
# Some checkpoints pickle the carry classes under other module names, carry an unused halting head (``inner.q_head``,
# never trained: it holds its init values and has no optimizer state), or store the challengers as
# ``explore_challengers``.
# ----------------------------------------------------------------------------

LEGACY_HALT_HEAD = ("inner.q_head.weight", "inner.q_head.bias")
_LEGACY_HALT_HEAD_INIT = {"inner.q_head.weight": 0.0, "inner.q_head.bias": -5.0}


def torch_load(path: str, map_location: Any) -> Any:
    """``torch.load`` that also finds the carry classes under other module names (``models.single_latent``, ...)."""
    from models.energy import LatentCarry, ModelCarry

    carries = {"LatentCarry": LatentCarry, "ModelCarry": ModelCarry}

    class Unpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if (
                module in ("models.single_latent", "models.single_latent_energy", "ebr.models.energy")
                and name in carries
            ):
                return carries[name]
            if module.startswith("ebr.models."):
                module = module[len("ebr.") :]
            return super().find_class(module, name)

    pickle_module = types.SimpleNamespace(Unpickler=Unpickler, load=pickle.load, __name__="pickle")
    return torch.load(path, map_location=map_location, weights_only=False, pickle_module=pickle_module)


def _upgrade_checkpoint(ckpt: dict[str, Any], *, where: str) -> dict[str, Any]:
    if ckpt.get("explore") is None and ckpt.get("explore_challengers") is not None:
        ckpt["explore"] = {"challenger_z": ckpt["explore_challengers"]}
    model = ckpt.get("model")
    if model is None or not any(_halt_head_key(strip_orig_mod(k)) for k in model):
        return ckpt
    if ckpt.get("optimizers"):
        ckpt["optimizers"] = drop_legacy_halt_head_slots(ckpt["optimizers"], model)
    ckpt["model"] = drop_legacy_halt_head(model, where=where)
    ema = ckpt.get("ema")
    if isinstance(ema, dict):
        if isinstance(ema.get("shadow"), dict):
            ema["shadow"] = drop_legacy_halt_head(ema["shadow"], where=f"{where} (EMA)")
        else:
            ckpt["ema"] = drop_legacy_halt_head(ema, where=f"{where} (EMA)")
    return ckpt


def _halt_head_key(name: str) -> str | None:
    return next((k for k in LEGACY_HALT_HEAD if name == k or name.endswith("." + k)), None)


def drop_legacy_halt_head(state_dict: dict[str, torch.Tensor], *, where: str) -> dict[str, torch.Tensor]:
    """``state_dict`` without the halting head, after checking that it still holds its init values."""
    out = {}
    for k, v in state_dict.items():
        head_key = _halt_head_key(strip_orig_mod(k))
        if head_key is None:
            out[k] = v
        elif not bool((v == _LEGACY_HALT_HEAD_INIT[head_key]).all()):
            raise ValueError(f"{where}: {k} is not at its init value; this checkpoint used a learned halt head")
    return out


def drop_legacy_halt_head_slots(
    optimizers: list[dict[str, Any]], model_state: dict[str, torch.Tensor]
) -> list[dict[str, Any]]:
    """The optimizer states re-indexed as if the halting head never existed.

    The head sat in the AdamW group (the last optimizer, see ``utils.optim.build_optimizers``) and must have no state
    there. Every other parameter keeps its state and order. ``model_state`` lists the parameters in model order.
    """
    from utils.optim import ADAMW_PATTERNS

    names = [strip_orig_mod(k) for k in model_state]
    if not any(_halt_head_key(n) for n in names):
        return optimizers
    adamw = [
        n
        for n, t in zip(names, model_state.values(), strict=True)
        if _halt_head_key(n) or t.ndim != 2 or any(p in n.lower() for p in ADAMW_PATTERNS)
    ]
    adamw_state = dict(optimizers[-1])
    (group,) = adamw_state["param_groups"]
    if len(group["params"]) != len(adamw):
        raise ValueError(f"AdamW group has {len(group['params'])} params, expected {len(adamw)} from the model")
    drop = {group["params"][i] for i, n in enumerate(adamw) if _halt_head_key(n)}
    if any(k in adamw_state["state"] for k in drop):
        raise ValueError("the legacy halt head has optimizer state; this checkpoint used a learned halt head")
    keep = [p for p in group["params"] if p not in drop]
    new_index = {old: new for new, old in enumerate(keep)}
    adamw_state["state"] = {new_index[k]: v for k, v in adamw_state["state"].items()}
    adamw_state["param_groups"] = [{**group, "params": [new_index[p] for p in keep]}]
    return list(optimizers[:-1]) + [adamw_state]


__all__ = [
    "EMAHelper",
    "LEGACY_HALT_HEAD",
    "TrainState",
    "load_checkpoint",
    "load_ema_shadow",
    "load_model_state_dict",
    "load_training_state",
    "prime_attention_self_test",
    "restore_rng",
    "save_checkpoint",
    "seed_attention_self_test",
    "snapshot_code",
    "strip_orig_mod",
    "torch_load",
]
