"""Model factory."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.energy import EnergyReasoner
from models.losses import InferenceLossHead, LossHead


def build_model(config: Any, metadata: Any, *, device: torch.device, inference: bool = False) -> nn.Module:
    """``LossHead(EnergyReasoner)`` (``InferenceLossHead`` for evaluation) on ``device``.

    Built under ``torch.device(device)``, so parameter init draws from that device's RNG.
    """
    arch = dict(config.arch.__pydantic_extra__)
    arch.update(seq_len=metadata.seq_len, vocab_size=metadata.vocab_size)
    head_cls = InferenceLossHead if inference else LossHead
    with torch.device(device):
        return head_cls(EnergyReasoner(arch), convergence_loss_weight=config.arch.loss.convergence_loss_weight)


__all__ = ["EnergyReasoner", "InferenceLossHead", "LossHead", "build_model"]
