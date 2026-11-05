"""The training loss and the metrics logged with it.

Each segment's loss is the stablemax cross entropy of its endpoint decode plus ``convergence_loss_weight`` times the
convergence loss: the binary cross entropy of sigmoid(convergence logit) against whether the endpoint decodes exactly
to the label. Training rows add the interior CE of the graded microsteps (``aux_loss`` from the model).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100  # label of the cells that are not graded (padding)


# Metrics reduced with max / min instead of a sum (across ranks, eval batches and the logged mean).
EXTREMUM_METRICS = {
    "max_steps": "max",
    "max_convergence_logit": "max",
    "min_steps": "min",
    "min_convergence_logit": "min",
}


def _stablemax_transform(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Log of stablemax, a softmax with exp(x) replaced by x + 1 (x >= 0) and 1 / (1 - x) (x < 0)."""
    transformed = _stablemax_transform(x)
    return torch.log(transformed / torch.sum(transformed, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Stablemax cross entropy per cell, computed in float64, 0 outside ``valid_mask``."""
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    labels = torch.where(valid_mask, labels, 0)
    label_logprobs = torch.gather(logprobs, index=labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid_mask, label_logprobs, 0)


class LossHead(nn.Module):
    """Wraps the reasoner: runs one segment and returns its loss and metrics.

    Metrics are row sums over the rows that halted this segment (``count`` of them), so the trainer can average over
    any number of segments and ranks.
    """

    def __init__(self, model: nn.Module, convergence_loss_weight: float) -> None:
        super().__init__()
        self.model = model
        self.convergence_loss_weight = float(convergence_loss_weight)

    def initial_carry(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.initial_carry(*args, **kwargs)

    def _scored_rows(self, carry: Any, graded_cells: torch.Tensor) -> torch.Tensor:
        """Rows whose metrics count this segment: training counts the rows that halted."""
        return carry.halted & (graded_cells > 0)

    def forward(self, return_keys: Sequence[str], **model_kwargs: Any):
        """Returns (carry, loss, metrics, the requested outputs detached, whether every row halted)."""
        carry, outputs = self.model(**model_kwargs)
        logits = outputs["logits"] = outputs["logits"].contiguous()
        labels = carry.current_data["labels"].to(logits.device)
        graded = outputs.pop("graded_labels", None)
        if graded is not None:
            labels = graded  # reverse SXM: the endpoint CE and the convergence target use the chosen solution
        preds = outputs["preds"] = torch.argmax(logits, dim=-1)
        convergence = outputs["convergence_logit"]

        with torch.no_grad():
            cells = labels != IGNORE_LABEL_ID
            graded_cells = cells.sum(-1)
            solved = (cells & (preds == labels)).sum(-1) == graded_cells
            metrics = self._metrics(carry, preds, labels, cells, solved, convergence)

        divisor = graded_cells.clamp_min(1).unsqueeze(-1)
        lm_loss = (
            stablemax_cross_entropy(logits, labels, valid_mask=cells) / divisor
        ).sum()
        convergence_loss = F.binary_cross_entropy_with_logits(
            convergence, solved.to(convergence.dtype), reduction="sum"
        )
        metrics["lm_loss"] = lm_loss.detach()
        metrics["convergence_loss"] = convergence_loss.detach()
        total_loss = lm_loss + self.convergence_loss_weight * convergence_loss
        aux_loss = outputs.pop("aux_loss", None)
        if aux_loss is not None:
            metrics["aux_loss"] = aux_loss.detach()
            total_loss = total_loss + aux_loss
        for key, value in (outputs.pop("aux_metrics", None) or {}).items():
            metrics[key] = value.detach()
        metrics["total_loss"] = total_loss.detach()
        detached = {key: outputs[key].detach() for key in return_keys if key in outputs}
        return carry, total_loss, metrics, detached, carry.halted.all()

    def _metrics(
        self,
        carry: Any,
        preds: torch.Tensor,
        labels: torch.Tensor,
        cells: torch.Tensor,
        solved: torch.Tensor,
        convergence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graded_cells = cells.sum(-1)
        scored = self._scored_rows(carry, graded_cells)
        zero = torch.zeros_like(graded_cells, dtype=torch.float32)
        cell_accuracy = ((cells & (preds == labels)).to(torch.float32) / graded_cells.clamp_min(1).unsqueeze(-1)).sum(
            -1
        )
        steps = carry.steps.to(torch.float32)
        converged = convergence > 0
        z = carry.inner_carry.z.to(torch.float32)
        return {
            "count": scored.sum(),
            "accuracy": torch.where(scored, cell_accuracy, 0).sum(),
            "exact_accuracy": (scored & solved).sum(),
            "z_mean": z.mean(dim=-1).mean(dim=-1).sum(),
            "z_std": z.std(dim=-1).mean(dim=-1).sum(),
            # How well convergence (logit > 0) predicts a solved endpoint.
            "convergence_logit": torch.where(scored, convergence, zero).sum(),
            "max_convergence_logit": torch.where(scored, convergence.to(torch.float32), zero - torch.inf).max(),
            "min_convergence_logit": torch.where(scored, convergence.to(torch.float32), zero + torch.inf).min(),
            "convergence_accuracy": (scored & (converged == solved)).sum(),
            "convergence_precision": torch.where(
                scored,
                (converged & solved).to(torch.float32).sum() / (convergence >= 0).to(torch.float32).sum().clamp_min(1),
                zero,
            ).sum(),
            "convergence_recall": torch.where(
                scored, (converged & solved).to(torch.float32).sum() / solved.to(torch.float32).sum().clamp_min(1), zero
            ).sum(),
            "steps": torch.where(scored, steps, 0).sum(),
            "max_steps": torch.where(scored, steps, torch.zeros_like(steps)).max(),
            "min_steps": torch.where(
                scored, steps, torch.full_like(steps, float(self.model.config.halt_max_steps))
            ).min(),
        }


class InferenceLossHead(LossHead):
    """Evaluation: metrics on the rows at the last segment; done when every row has run ``halt_max_steps``."""

    def _scored_rows(self, carry: Any, graded_cells: torch.Tensor) -> torch.Tensor:
        return (carry.steps >= self.model.config.halt_max_steps) & (graded_cells > 0)

    def forward(self, return_keys: Sequence[str], **model_kwargs: Any):
        carry, total_loss, metrics, detached, _ = super().forward(return_keys, **model_kwargs)
        for key in (
            "max_steps",
            "min_steps",
            "aux_loss",
            "convergence_logit",
            "max_convergence_logit",
            "min_convergence_logit",
        ):
            metrics.pop(key, None)
        finished = (carry.halted | (carry.steps >= self.model.config.halt_max_steps)).all()
        return carry, total_loss, metrics, detached, finished


__all__ = [
    "EXTREMUM_METRICS",
    "IGNORE_LABEL_ID",
    "InferenceLossHead",
    "LossHead",
    "log_stablemax",
    "stablemax_cross_entropy",
]
