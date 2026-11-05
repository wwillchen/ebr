"""Training entry point: python pretrain.py train/<task> [key=value ...].

Each optimizer step runs one segment on the batch (``EnergyReasoner``): rows that halted get a new puzzle, the others
continue from their carried state. The EMA weights are evaluated on the test set every ``eval_interval_steps`` and the
full training state is checkpointed every ``checkpoint_interval_steps``.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from dataset.puzzle_dataset import PuzzleDatasetMetadata, create_dataloader
from evaluators.in_loop import create_evaluators, evaluate
from models import build_model
from models.losses import EXTREMUM_METRICS
from utils.checkpoint import (
    EMAHelper,
    TrainState,
    load_training_state,
    prime_attention_self_test,
    restore_rng,
    save_checkpoint,
    seed_attention_self_test,
    snapshot_code,
)
from utils.config import PretrainConfig, compose_config
from utils.optim import AdaptiveGradClip, RelativeGradClip, build_optimizers, lr_with_warmup
from utils.runtime import (
    allreduce_gradients,
    broadcast_model,
    check_param_sync,
    dist_rank,
    dist_world,
    init_distributed,
    isolated_rng,
    seed_everything,
    tree_to_device,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

torch.set_float32_matmul_precision("high")  # TF32 matmuls

GRAD_NORM_LOG_INTERVAL = 25
# The model reports metrics as sums over rows. Most are averaged over the rows that halted this step (``count``);
# these are averaged over the whole batch instead.
_PER_BATCH_PREFIXES = (
    "microstep_ce",
    "reset_cycles",
    "explore_",
    "noisy_init",
    "convergence_residual",
    "converged_frac",
)
_PER_BATCH_KEYS = {"z_mean", "z_std"}
# Averages over the halted rows, dropped on steps where no row halted.
_HALTED_ROW_METRICS = {
    "accuracy",
    "exact_accuracy",
    "convergence_accuracy",
    "convergence_precision",
    "convergence_recall",
    "convergence_logit",
    "max_convergence_logit",
    "min_convergence_logit",
    "steps",
    "max_steps",
    "min_steps",
}


def log(msg: str) -> None:
    prefix = f"[ebr r{dist_rank()}]" if dist_world() > 1 else "[ebr]"
    print(f"{prefix} {msg}", flush=True)


def init_train_state(config: PretrainConfig, metadata: PuzzleDatasetMetadata, device: torch.device) -> TrainState:
    samples = int(metadata.total_groups * metadata.mean_puzzle_examples)
    steps_per_epoch = max(samples // max(config.global_batch_size, 1), 1)
    total_steps = config.epochs * steps_per_epoch
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    log(
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} vocab={metadata.vocab_size} seq_len={metadata.seq_len}"
    )
    model = build_model(config, metadata, device=device)
    optimizers, lrs, summary = build_optimizers(model, config)
    log(summary)
    ema = EMAHelper(mu=config.ema_rate)
    ema.register(model)
    clip = None
    if config.grad_clip_relative is not None:
        clip = RelativeGradClip(config.grad_clip_relative, start_step=config.lr_warmup_steps)
        log(
            f"gradient clipping at {config.grad_clip_relative:g}x the running average norm after step {clip.start_step}"
        )
    elif config.grad_clip_adaptive is not None:
        clip = AdaptiveGradClip(config.grad_clip_adaptive)
        log(
            f"adaptive gradient clipping at median + {config.grad_clip_adaptive:g} spreads of the last {clip.window} "
            f"log norms, from step {clip.min_history + 1}"
        )
    return TrainState(
        model=model,
        optimizers=optimizers,
        optimizer_lrs=lrs,
        carry=None,
        step=0,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        ema_helper=ema,
        grad_clip=clip,
    )


def train_batch(config: PretrainConfig, state: TrainState, batch: Any, global_batch_size: int) -> dict[str, Any] | None:
    """One optimizer step on ``batch``. Returns the logged metrics, or None once the step budget is used up."""
    state.step += 1
    if state.step > state.total_steps:
        state.step = state.total_steps
        return None
    device = next(state.model.parameters()).device
    batch = tree_to_device(batch, device)

    if state.carry is None:
        state.carry = state.model.initial_carry(batch)
    state.carry, loss, metrics, _, _ = state.model(carry=state.carry, batch=batch, return_keys=[])
    ((1.0 / global_batch_size) * loss).backward()
    metrics = dict(metrics)
    if dist_world() > 1:
        allreduce_gradients(state.model)
        metrics = _sum_over_ranks(metrics, device)

    log_norms = state.step == 1 or state.step % GRAD_NORM_LOG_INTERVAL == 0
    grad_norms = _grad_norms(state.model) if log_norms else {}  # before clipping
    if state.grad_clip is not None:
        state.grad_clip([p.grad for p in state.model.parameters() if p.grad is not None], state.step)
        if log_norms:
            grad_norms.update(state.grad_clip.pop_metrics())
    lr = None
    for opt, base_lr in zip(state.optimizers, state.optimizer_lrs, strict=True):
        lr = lr_with_warmup(
            state.step,
            base_lr=base_lr,
            warmup_steps=config.lr_warmup_steps,
            total_steps=state.total_steps,
            cooldown_steps=config.lr_cooldown_steps,
        )
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        opt.zero_grad()

    out = _average_metrics(metrics, global_batch_size)
    out["train/lr"] = lr
    out["train/epoch"] = (state.step - 1) // max(state.steps_per_epoch, 1)
    out.update(grad_norms)
    return out


def _sum_over_ranks(metrics: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Sum the row-summed metrics over ranks; the extrema take the max / min."""
    keys = sorted(metrics)
    if not keys:
        return metrics
    values = torch.stack([metrics[k].detach().to(torch.float32).to(device) for k in keys])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    summed = {k: values[i] for i, k in enumerate(keys)}
    for k, kind in EXTREMUM_METRICS.items():
        if k in metrics:
            t = metrics[k].detach().to(torch.float32).to(device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX if kind == "max" else dist.ReduceOp.MIN)
            summed[k] = t
    return summed


def _grad_norms(model: torch.nn.Module) -> dict[str, float]:
    total_sq, largest, tensors = 0.0, 0.0, 0
    for p in model.parameters():
        if p.grad is not None:
            norm = float(torch.norm(p.grad.detach().to(torch.float32)))
            total_sq, largest, tensors = total_sq + norm * norm, max(largest, norm), tensors + 1
    return {
        "train/grad_norm/total": math.sqrt(total_sq),
        "train/grad_norm/max": largest,
        "train/grad_norm/num_tensors": float(tensors),
    }


def _average_metrics(metrics: dict[str, torch.Tensor], global_batch_size: int) -> dict[str, float]:
    """``train/<key>``: row sums divided by the halted rows, or by the batch for losses and per-batch metrics.

    On a step where no row halted, the halted-row metrics are left out.
    """
    values = {k: float(v) for k, v in metrics.items()}
    count = values.pop("count", 0.0)
    out = {}
    for k, v in values.items():
        if not count and k in _HALTED_ROW_METRICS:
            continue
        if k in _PER_BATCH_KEYS or k.startswith(_PER_BATCH_PREFIXES) or k.endswith("loss"):
            out[f"train/{k}"] = v / global_batch_size
        elif k in EXTREMUM_METRICS:
            out[f"train/{k}"] = v
        else:
            out[f"train/{k}"] = v / max(count, 1.0)
    return out


class MetricLogger:
    """Appends metrics to ``<run_dir>/metrics.jsonl`` and, if enabled, to Weights & Biases."""

    def __init__(self, run_dir: str, config: PretrainConfig, run_id: str, resume: bool) -> None:
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self.wandb = None
        mode = (config.wandb_mode or "disabled").lower()
        if mode not in ("disabled", "disable", "off", "none"):
            try:
                import wandb

                kwargs: dict[str, Any] = dict(
                    project=config.project_name, mode=mode, dir=run_dir, id=run_id, config=config.model_dump()
                )
                if resume:
                    kwargs["resume"] = "allow"
                else:
                    kwargs.update(name=config.run_name, tags=[config.dataset.name])
                self.wandb = wandb.init(**kwargs)
            except Exception as exc:  # logging must not stop training
                log(f"wandb disabled ({exc})")

    def log(self, metrics: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics)

    def finish(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


def _flatten_eval(metrics: dict[str, Any], step: int) -> dict[str, Any]:
    flat: dict[str, Any] = {"step": step}
    for k, v in metrics.items():
        if isinstance(v, dict):
            flat.update({f"eval/{k}/{kk}": vv for kk, vv in v.items()})
        else:
            flat[f"eval/{k}"] = v
    return flat


class StopSignal:
    """SIGTERM / SIGUSR1 (e.g. Slurm preemption): every rank checkpoints and stops after the current step."""

    def __init__(self) -> None:
        self.signal_number = 0

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.signal_number = signum

    def stop_reason(self, device: torch.device, world: int) -> str | None:
        code = 0
        if self.signal_number:
            code = 1 if self.signal_number == signal.SIGTERM else 2
        if world > 1:
            requested = torch.tensor(code, dtype=torch.int32, device=device)
            dist.all_reduce(requested, op=dist.ReduceOp.MAX)
            code = int(requested.item())
        return {0: None, 1: "SIGTERM", 2: "SIGUSR1"}[code]


def train(config: PretrainConfig, raw_config: dict[str, Any]) -> None:
    stop = StopSignal()
    previous = {}
    if threading.current_thread() is threading.main_thread():  # signal handlers can only be set there
        for signum in (signal.SIGTERM, signal.SIGUSR1):
            previous[signum] = signal.signal(signum, stop.request_stop)
    try:
        _train(config, raw_config, stop)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _make_run_dir(resume_info: dict[str, Any] | None, rank: int, world: int) -> tuple[str, str]:
    """``<OUTPUT_ROOT>/outputs/<run_id>/<Y-m-d>/<H-M-S>/``, chosen by rank 0. A resumed run keeps its run id."""
    run_id = (resume_info or {}).get("wandb_run_id") or secrets.token_hex(4)
    now = datetime.now()
    root = os.environ.get("OUTPUT_ROOT") or REPO_ROOT
    stamp = (f"{now.year}-{now.month}-{now.day}", f"{now.hour}-{now.minute}-{now.second}")
    run_dir = os.path.abspath(os.path.join(root, "outputs", run_id, *stamp))
    if world > 1:
        shared = [run_id, run_dir]
        dist.broadcast_object_list(shared, src=0)
        run_id, run_dir = shared
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir


def _barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


@dataclass
class InLoopEval:
    """The test split and the dataset's extra evaluators, for the in-loop eval of the EMA weights."""

    loader: Any
    metadata: PuzzleDatasetMetadata
    evaluators: list[Any]


def _open_eval_set(config: PretrainConfig) -> InLoopEval | None:
    """The in-loop eval data, or None when evals are off or the dataset has no test split."""
    if config.eval_interval_steps is None:
        return None
    try:
        loader, metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.eval_batch_size or config.global_batch_size,
        )
        evaluators = create_evaluators(config, metadata)
    except FileNotFoundError:  # no test split: train without in-loop evals
        return None
    return InLoopEval(loader, metadata, evaluators)


def _build_train_state(
    config: PretrainConfig, metadata: PuzzleDatasetMetadata, device: torch.device, rank: int, world: int
) -> TrainState:
    # Reseeded so the init does not depend on the loaders (seeds the CPU and every CUDA device).
    torch.manual_seed(config.seed)
    state = init_train_state(config, metadata, device)
    if world > 1 or config.attention_self_test == "before_training":
        prime_attention_self_test(state.model)  # it reseeds the global RNG; run it now, outside the training stream
    else:
        seed_attention_self_test(state.model, config.seed)  # it runs in step 1 and reseeds the RNG with this seed
    if world > 1:
        broadcast_model(state.model)  # no DDP wrapper: every rank starts from rank 0's weights
        state.ema_helper.register(state.model)
        seed_everything(config.seed + rank)  # each rank's own training stream; init and batches are unaffected
    return state


def _start_run_dir(
    config: PretrainConfig,
    raw_config: dict[str, Any],
    state: TrainState,
    run_id: str,
    run_dir: str,
    *,
    resumed: bool,
    rank: int,
    world: int,
) -> MetricLogger | None:
    """Create the checkpoint directory, save the config and code (rank 0). Returns the metric log (rank 0 only)."""
    config.checkpoint_path = os.path.join(run_dir, "checkpoints")
    _barrier(world)
    os.makedirs(config.checkpoint_path, exist_ok=True)
    if rank == 0:
        saved = OmegaConf.create({**raw_config, "run_id": run_id, "run_dir": run_dir})
        OmegaConf.save(config=saved, f=os.path.join(run_dir, "all_config.yaml"))
        snapshot_code(run_dir, REPO_ROOT)
    num_params = sum(p.numel() for p in state.model.parameters())
    log(f"run {config.run_name} id={run_id} dir={run_dir}")
    if rank != 0:
        return None
    log(f"model parameters: {num_params:,}")
    logger = MetricLogger(run_dir, config, run_id, resume=resumed)
    logger.log({"num_params": num_params, "step": state.step})
    return logger


@dataclass
class Run:
    """One training run on this rank: its state, data, and where it writes (run dir, checkpoints, metrics)."""

    config: PretrainConfig
    state: TrainState
    train_loader: Any
    eval_set: InLoopEval | None
    run_id: str
    run_dir: str
    logger: MetricLogger | None  # rank 0 only
    device: torch.device
    rank: int
    world: int

    def log_step(self, metrics: dict[str, Any], start: tuple[int, float]) -> None:
        """Log the step's metrics; every 50 steps also print them, with the speed since ``start`` = (step, time)."""
        step = self.state.step
        metrics["step"] = step
        if self.logger:
            self.logger.log(metrics)
        if self.rank == 0 and (step % 50 == 0 or step == 1):
            start_step, start_time = start
            speed = (step - start_step) / max(time.perf_counter() - start_time, 1e-9)
            log(
                f"step {step}/{self.state.total_steps} loss={metrics.get('train/total_loss', float('nan')):.4f} "
                f"exact={metrics.get('train/exact_accuracy', float('nan')):.3f} lr={metrics['train/lr']:.2e} "
                f"({speed:.2f} it/s)"
            )

    def checkpoint(self) -> None:
        path = save_checkpoint(
            self.config, self.state, self.train_loader, run_id=self.run_id, rank=self.rank, world=self.world
        )
        if self.world > 1:
            dist.barrier()
            check_param_sync(self.state.model)
        if self.rank == 0:
            log(f"saved checkpoint {path}")

    def run_eval(self) -> None:
        """Evaluate the EMA weights on rank 0 (the other ranks wait).

        ``eval_rng: reseed`` seeds the eval with the run's seed and restores the training RNG afterwards;
        ``coupled`` draws from the training RNG.
        """
        if self.rank == 0:
            config, state, data = self.config, self.state, self.eval_set
            model = state.ema_helper.ema_copy(state.model)
            model.eval()
            if config.eval_rng == "coupled":
                metrics, elapsed = evaluate(model, data.loader, data.metadata, data.evaluators, progress=True)
            else:
                with isolated_rng(self.device):
                    seed_everything(config.seed)
                    metrics, elapsed = evaluate(model, data.loader, data.metadata, data.evaluators, progress=True)
            flat = _flatten_eval(metrics, state.step)
            flat["eval/time_sec"] = elapsed
            self.logger.log(flat)
            headline = {
                k: round(v, 4) for k, v in flat.items() if k.endswith(("exact_accuracy", "accuracy", "shortest_path"))
            }
            log(f"eval @ {state.step}: {headline} ({elapsed:.0f}s)")
            del model
            state.model.train()
        _barrier(self.world)

    def write_status(self, lifecycle: str, reason: str | None) -> None:
        """``<run_dir>/training_status.json`` (rank 0), replaced atomically."""
        if self.rank != 0:
            return
        status = {
            "state": lifecycle,
            "step": self.state.step,
            "total_steps": self.state.total_steps,
            "stop_reason": reason,
            "completed": lifecycle == "completed",
        }
        path = os.path.join(self.run_dir, "training_status.json")
        with open(path + ".tmp", "w") as f:
            json.dump(status, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path + ".tmp", path)


def _train_loop(run: Run, stop: StopSignal, rng_fix: Any) -> str | None:
    """Train to ``total_steps`` or until a stop signal, then checkpoint the last step. Returns the stop reason.

    Each pass over the train loader creates a new iterator, which draws from the CPU RNG. A resumed run first finishes
    the checkpointed pass, restoring the RNG after that draw (``rng_fix``), then starts fresh passes.
    """
    config, state = run.config, run.state
    last_checkpoint_step = None
    start = (state.step, time.perf_counter())  # a resumed run's speed counts only the steps it ran
    state.model.train()
    stop_reason = stop.stop_reason(run.device, run.world)
    while state.step < state.total_steps and stop_reason is None:
        batches = iter(run.train_loader)
        if rng_fix is not None:
            restore_rng(rng_fix)
        for _set_name, batch, global_batch_size in batches:
            metrics = train_batch(config, state, batch, global_batch_size)
            if metrics is None:
                break
            run.log_step(metrics, start)
            state.ema_helper.update(state.model)
            stop_reason = stop.stop_reason(run.device, run.world)
            if config.checkpoint_interval_steps and state.step % config.checkpoint_interval_steps == 0:
                run.checkpoint()
                last_checkpoint_step = state.step
            eval_due = config.eval_interval_steps and state.step % config.eval_interval_steps == 0
            if eval_due and stop_reason is None and run.eval_set is not None:
                run.run_eval()
                if last_checkpoint_step == state.step and config.eval_rng == "coupled":
                    run.checkpoint()  # the eval drew from the training RNG after the save
            if state.step >= state.total_steps or stop_reason is not None:
                break
        rng_fix = None
    if last_checkpoint_step != state.step:
        run.checkpoint()
    return stop_reason


def _train(config: PretrainConfig, raw_config: dict[str, Any], stop: StopSignal) -> None:
    rank, world = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    if config.global_batch_size % world:
        raise ValueError(f"global_batch_size {config.global_batch_size} must be divisible by world_size {world}")

    log(f"dataset {config.dataset.name} from {config.dataset.data_path}" + (f" (world {world})" if world > 1 else ""))
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=config.train_epochs_per_iter or config.epochs,
        global_batch_size=config.global_batch_size,
        rank=rank,
        num_replicas=world,
    )
    eval_set = _open_eval_set(config)
    state = _build_train_state(config, train_metadata, device, rank, world)
    resume_info = load_training_state(config, state, train_loader)

    run_id, run_dir = _make_run_dir(resume_info, rank, world)
    resumed = resume_info is not None
    logger = _start_run_dir(config, raw_config, state, run_id, run_dir, resumed=resumed, rank=rank, world=world)
    run = Run(config, state, train_loader, eval_set, run_id, run_dir, logger, device, rank, world)

    run.write_status("running", None)
    try:
        stop_reason = _train_loop(run, stop, rng_fix=(resume_info or {}).get("rng_fix"))
        completed = state.step >= state.total_steps
        run.write_status("completed" if completed else "stopped", "target_reached" if completed else stop_reason)
        log("training completed" if completed else f"training stopped at {state.step}: {stop_reason}")
    except BaseException as exc:
        run.write_status("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if logger:
            logger.finish()


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python pretrain.py <config name under config/, e.g. train/sudoku> [key=value ...]")
        return
    config, raw = compose_config(argv[0], argv[1:])
    if config.run_name is None:
        config.run_name = raw["run_name"] = f"ebr_{os.path.basename(argv[0])}_s{config.seed}"
    train(config, raw)


if __name__ == "__main__":
    main()
