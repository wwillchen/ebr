"""Training configuration: the schema (validated from the Hydra-composed YAML), composing it, and reading configs in
other schemas."""

from __future__ import annotations

import copy
import os
from typing import Any

import pydantic

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    # Weight of the convergence loss (lambda); the endpoint CE has weight 1.
    convergence_loss_weight: float


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")  # the other keys are the EnergyConfig fields, checked below
    loss: LossConfig

    @pydantic.model_validator(mode="after")
    def _known_keys(self) -> ArchConfig:
        from models.energy import EnergyConfig

        unknown = sorted(set(self.__pydantic_extra__) - set(EnergyConfig.model_fields))
        if unknown:
            raise ValueError(f"unknown arch options: {unknown}")
        return self


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")  # the other keys are the evaluator's arguments
    name: str


class DatasetConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: str
    data_path: str
    evaluators: list[EvaluatorConfig] = []  # extra in-loop metrics, e.g. Maze-Hard path validity
    vectorized_sampler: pydantic.StrictBool = False


class PretrainConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    arch: ArchConfig
    dataset: DatasetConfig
    seed: int = 0

    global_batch_size: int
    eval_batch_size: int | None = None
    epochs: int
    max_steps: int | None = None
    train_epochs_per_iter: int | None = None  # epochs per pass of the train loader (sets the shuffle schedule)
    # In-loop eval of the EMA weights every eval_interval_steps. reseed: seeded with the run's seed and isolated from
    # the training RNG. coupled: draws from and advances the training RNG.
    eval_interval_steps: int | None = None
    eval_rng: str = "reseed"
    # When the attention second-order self-test runs. It reseeds the global RNG (models.attention.verify_second_order).
    # before_training: once before step 1 under a forked RNG, so it does not touch the training draws. first_step: at
    # the first graded microstep on one GPU, reseeding the RNG with the run's seed, and training continues from there.
    # Several ranks always run it before training.
    attention_self_test: str = "before_training"
    # Full-state checkpoint interval (plus the final step); null saves only at the end.
    checkpoint_interval_steps: int | None = 5000

    # Muon for hidden matrices, AdamW for the rest (utils/optim.py); linear warmup, then a constant learning rate.
    lr: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    ema_rate: float = 0.999
    # Off (null) by default. A factor k > 1 clips the global gradient norm at k times its running average
    # (half-life 200 steps), starting after the LR warmup (utils.optim.RelativeGradClip).
    grad_clip_relative: float | None = None
    # Off (null) by default; 4 in train/maze_hard_fast. z > 0 clips the global gradient norm at
    # exp(median + z * lower-half spread) of the log norms of the last 200 steps (utils.optim.AdaptiveGradClip).
    grad_clip_adaptive: float | None = None
    # Off by default. lr_cooldown_steps: decay the LR linearly to 0 over the last N steps of the run.
    # muon_split_fused: Muon orthogonalizes the q, k, v blocks of the fused attention projection (and the gate, up
    # blocks of the SwiGLU one) separately, as if each were stored as its own matrix.
    lr_cooldown_steps: int | None = None
    muon_split_fused: pydantic.StrictBool = False

    project_name: str | None = None
    run_name: str | None = None
    wandb_mode: str | None = None
    load_checkpoint: str | None = None
    checkpoint_path: str | None = None  # set by the trainer

    @pydantic.model_validator(mode="after")
    def _validate(self) -> PretrainConfig:
        for key in (
            "train_epochs_per_iter",
            "eval_interval_steps",
            "checkpoint_interval_steps",
            "max_steps",
            "lr_cooldown_steps",
        ):
            value = getattr(self, key)
            if value is not None and value <= 0:
                raise ValueError(f"{key} must be a positive integer when provided")
        if self.eval_rng not in ("reseed", "coupled"):
            raise ValueError("eval_rng must be reseed or coupled")
        if self.attention_self_test not in ("before_training", "first_step"):
            raise ValueError("attention_self_test must be before_training or first_step")
        if not 0.0 < self.ema_rate <= 1.0:
            raise ValueError("ema_rate must be in (0, 1]")
        if self.grad_clip_relative is not None and not self.grad_clip_relative > 1.0:
            raise ValueError("grad_clip_relative must be > 1 (a multiple of the running average gradient norm)")
        if self.grad_clip_adaptive is not None and not self.grad_clip_adaptive > 0.0:
            raise ValueError("grad_clip_adaptive must be > 0 (spreads above the median log gradient norm)")
        if self.grad_clip_relative is not None and self.grad_clip_adaptive is not None:
            raise ValueError("grad_clip_relative and grad_clip_adaptive are exclusive")
        return self


# ----------------------------------------------------------------------------
# Configs in other schemas (e.g. saved in checkpoints)
#
# They carry options this code does not expose. Each such option maps to the values that mean what the code does:
# the key is dropped when its value is one of them and rejected otherwise (ANY: it never affected the model or the
# optimizer step). Renamed options are moved to their current names. Other unknown top-level keys (logging intervals,
# run bookkeeping) are dropped.
# ----------------------------------------------------------------------------
ANY = object()
MUON_KWARGS = {
    "momentum": 0.95,
    "nesterov": True,
    "ns_steps": 5,
    "ns_coefficients": [3.4445, -4.775, 2.0315],
    "eps": 1.0e-7,
    "adjust_lr_fn": "match_rms_adamw",
    "adamw_eps": 1.0e-8,
    "aux_adamw_patterns": ["embed", "lm_head", "q_head", "energy_head.output"],
}
REMOVED_TOP = {
    "optimizer": ("muon",),
    "optimizer_kwargs": (MUON_KWARGS,),
    "lr_min_ratio": (1.0,),
    "grad_clip_norm": (None,),
    "grad_accum_shards": (1,),
    "gradient_accumulation_steps": (1,),
    "ema": (True,),
    "gradient_checkpoint": (False,),
    "retain_graph": (False,),
}
REMOVED_ARCH = {
    "name": ("energy@EnergyReasoner", "single_latent_energy@SingleLatentEnergyModel"),
    "short_name": ANY,
    "mlp_t": (False,),
    "pos_encodings": ("rope2d",),
    "causal": (False,),
    "rms_norm_eps": (1e-5,),
    "rope_theta": (10000.0,),
    "attn_learn_logit_scale": (True,),
    "attn_verify_second_order": (True,),
    "attn_chunk_q": (0, None),
    "attn_symmetric": (False,),
    "attn_mlp_parallel": (False,),
    "attn_structural_mask": (None, "", "none"),
    "puzzle_emb_len": (0,),
    "puzzle_emb_ndim": (0,),
    "puzzle_emb_mode": ("prefix",),
    "energy_reduction": ("sum",),
    "descent_field": ("energy",),
    "sphere_z": (True,),
    "randomize_cycles": (True,),
    "replay_buffer": (False,),
    "replay_paired": (False,),
    "rest_halt": (True,),
    "rest_halt_logit": ("linear",),
    "rest_halt_eps": ANY,
    "label_seed_alpha_max": (1.0, None),
    "late_nearest_metric": ("ce",),
    "late_nearest_set_size": (0,),
    "late_nearest_probe": ANY,
    "energy_compile_backend": ("inductor",),
    "energy_compile_mode": (None,),
    "energy_compile_verify": (True,),
    "descent_wrapper_exact_casts": (False,),
    "eval_gnorm_stop_tol": (0, 0.0),
    "eval_gnorm_stop_min_steps": ANY,
    "eval_grad_norm": ANY,
}
REMOVED_LOSS = {
    "name": ("losses@ACTLossHead", "losses@InferenceLossHead"),
    "loss_type": ("stablemax_cross_entropy",),
    "verifier": (None,),
    "verifier_halt_target": (False,),
}
# Renamed options: old key -> current key.
RENAMED_ARCH = {
    "label_seed": "noisy_solution_init",
    "rest_halt_tau": "convergence_threshold",
    "rest_halt_temp": "convergence_temperature",
    "rest_halt_moves": "convergence_window",
}
RENAMED_LOSS = {"q_halt_loss_weight": "convergence_loss_weight"}


def _drop(d: dict[str, Any], table: dict[str, Any], where: str) -> None:
    for key, allowed in table.items():
        if key in d:
            value = d.pop(key)
            if allowed is not ANY and value not in allowed:
                raise ValueError(f"{where}{key}={value!r} is not supported by this code")


def _rename(d: dict[str, Any], table: dict[str, str], where: str) -> None:
    for old, new in table.items():
        if old in d:
            value = d.pop(old)
            if new in d and d[new] != value:
                raise ValueError(f"{where}{old}={value!r} conflicts with {where}{new}={d[new]!r}")
            d.setdefault(new, value)


def upgrade_arch(arch: dict[str, Any]) -> dict[str, Any]:
    """An ``arch`` dict (without ``loss``) with the removed options dropped (after checking their values) and the
    renamed ones under their current names."""
    arch = dict(arch)
    _drop(arch, REMOVED_ARCH, "arch.")
    for key in ("batch_size", "num_puzzle_identifiers", "pad_id", "input_seq_len"):  # were filled in from the data
        arch.pop(key, None)
    _rename(arch, RENAMED_ARCH, "arch.")
    k = int(arch.get("explore_k", 1))
    if arch.pop("explore_carry", k > 1) != (k > 1):
        raise ValueError("arch.explore_carry must equal explore_k > 1")
    if int(arch.pop("explore_chunk_mult", k)) < k:
        raise ValueError("arch.explore_chunk_mult must be >= explore_k (all candidates in one chunk)")
    return arch


def upgrade_config(raw: dict[str, Any]) -> dict[str, Any]:
    """A resolved config dict (a YAML's or a checkpoint's) in the current schema."""
    raw = copy.deepcopy(dict(raw))
    reseed = raw.pop("eval_reseed", None)  # boolean spelling of eval_rng
    if reseed is not None:
        raw.setdefault("eval_rng", "reseed" if reseed else "coupled")
    if raw.get("eval_rng") == "isolated":  # isolated without the reseed: read as reseed
        raw["eval_rng"] = "reseed"
    for key in ("exact_resume", "legacy_resume"):  # resume modes this code does not have
        raw.pop(key, None)
    _drop(raw, REMOVED_TOP, "")
    raw = {k: v for k, v in raw.items() if k in PretrainConfig.model_fields}
    loss = dict(raw["arch"]["loss"])
    raw["arch"] = upgrade_arch({k: v for k, v in raw["arch"].items() if k != "loss"})
    _drop(loss, REMOVED_LOSS, "arch.loss.")
    _rename(loss, RENAMED_LOSS, "arch.loss.")
    raw["arch"]["loss"] = loss
    return raw


def compose_config(config_name: str, overrides: list[str]) -> tuple[PretrainConfig, dict[str, Any]]:
    """Compose ``config/<config_name>.yaml`` (Hydra defaults lists) with CLI overrides: (validated config, raw dict)."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    path = os.path.join(CONFIG_DIR, f"{config_name}.yaml")
    if not os.path.isfile(path):
        group = os.path.dirname(path)
        known = sorted(f[:-5] for f in os.listdir(group) if f.endswith(".yaml")) if os.path.isdir(group) else []
        raise FileNotFoundError(f"no config {path}; {group} has {known}")
    with initialize_config_dir(config_dir=os.path.abspath(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    container = OmegaConf.to_container(cfg, resolve=True)
    container.pop("hydra", None)
    return PretrainConfig(**container), container


def config_from_dict(raw: dict[str, Any]) -> PretrainConfig:
    """A config from a resolved dict, e.g. a checkpoint's, in any schema ``upgrade_config`` reads."""
    return PretrainConfig(**upgrade_config(raw))


__all__ = [
    "ArchConfig",
    "DatasetConfig",
    "EvaluatorConfig",
    "LossConfig",
    "PretrainConfig",
    "compose_config",
    "config_from_dict",
    "upgrade_arch",
    "upgrade_config",
]
