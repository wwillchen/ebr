"""Reverse SXM for N-Queens and Graph Coloring: grade a trajectory against the valid solution closest to it.

Each input of these tasks has several valid solutions, and the dataset stores each one as its own row. Grading a state
against its row's stored solution pulls a trajectory that found a different valid solution back toward the stored one.
Reverse SXM grades the endpoint and the last few graded states of a segment against the closest solution (lowest cross
entropy) in the row's solution set: the solutions of the same input that are in the training batch.

If every trajectory of an input took its closest solution, they could all collapse onto one. So solutions are assigned
without reuse: once a row of an input takes a solution, the next rows of that input cannot, until ``refill_fraction``
of the input's solutions are taken and all of them become available again.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.losses import IGNORE_LABEL_ID, log_stablemax


def solution_sets(inputs: torch.Tensor, labels: torch.Tensor, max_solutions: int) -> torch.Tensor:
    """Each row's solution set ``[B, max_solutions, L]``: its own label, then the labels of the other rows with the
    same input (in row order), padded with empty (all ignored) solutions.

    The loader may cut the batch's last input short, so the rows of that input get an empty set (they keep their own
    label).
    """
    n = inputs.shape[0]
    _, input_id = torch.unique(inputs, dim=0, return_inverse=True)
    largest = int(torch.bincount(input_id).max())
    if largest > max_solutions:
        raise ValueError(f"an input has {largest} solutions in the batch, explore_rank_max_completions={max_solutions}")

    rows = torch.arange(n, device=inputs.device)
    same_input = (input_id[:, None] == input_id[None, :]) & (rows[:, None] != rows[None, :])
    others = torch.where(same_input, rows, n).sort(dim=1).values  # other rows of the same input first, n = none
    others = F.pad(others, (0, max_solutions), value=n)[:, : max_solutions - 1]
    labels_or_empty = torch.cat([labels, torch.full_like(labels[:1], IGNORE_LABEL_ID)])  # row n: the empty solution
    sets = torch.cat([labels.unsqueeze(1), labels_or_empty[others]], dim=1)
    sets[input_id == input_id[-1]] = IGNORE_LABEL_ID
    return sets


@torch.no_grad()
def reverse_sxm_targets(
    logits: torch.Tensor, sets: torch.Tensor, own_labels: torch.Tensor, inputs: torch.Tensor, refill_fraction: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grading targets ``[n, L]`` of one state with logits ``[n, L, V]``.

    Rows with an empty set keep ``own_labels``, rows with one solution are graded against it, and rows with several go
    through ``_assign_without_reuse``. Also returns, for logging, which rows have a set and which ones the no-reuse
    rule moved off their nearest solution.
    """
    sets = sets.to(logits.device)
    ce = _solution_ce(logits, sets)
    filled = (sets != IGNORE_LABEL_ID).any(-1)  # [n, M] non-empty slots
    nearest = ce.argmin(-1)
    slot = nearest.clone()
    several = filled.sum(-1) >= 2
    if several.any():
        slot[several] = _assign_without_reuse(
            ce[several], sets[several], inputs.to(logits.device)[several], refill_fraction
        )

    has_set = filled.any(-1)
    chosen = sets[torch.arange(len(slot), device=slot.device), slot].to(own_labels.dtype)
    target = torch.where(has_set[:, None], chosen, own_labels)
    return target, has_set, slot != nearest


def _solution_ce(logits: torch.Tensor, sets: torch.Tensor) -> torch.Tensor:
    """Mean per-cell stablemax cross entropy ``[n, M]`` (float64) of every solution in ``sets`` ``[n, M, L]`` under
    ``logits`` ``[n, L, V]``. Empty slots get inf."""
    cells = sets != IGNORE_LABEL_ID
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    tokens = torch.where(cells, sets, 0).long()
    nll = -logprobs.gather(2, tokens.transpose(1, 2)).transpose(1, 2)  # [n, M, L]
    ce = torch.where(cells, nll, 0.0).sum(-1) / cells.sum(-1).clamp_min(1)
    return torch.where(cells.any(-1), ce, torch.inf)


def _assign_without_reuse(
    ce: torch.Tensor, sets: torch.Tensor, inputs: torch.Tensor, refill_fraction: float
) -> torch.Tensor:
    """Slot ``[n]`` each row is graded against, for rows that each have at least two solutions.

    The rows of one input take turns in order of confidence (lowest best CE first, ties in row order). Each takes the
    lowest-CE solution of its set that is still free, or its nearest one if none is free. Once
    ``ceil(refill_fraction * distinct solutions)`` rows have taken one, all of the input's solutions are free again.
    Inputs do not interact, so turn ``t`` is taken by the ``t``-th row of every input at once.
    """
    n, m = ce.shape
    device = ce.device
    _, input_id = torch.unique(inputs, dim=0, return_inverse=True)
    num_inputs = int(input_id.max()) + 1

    # One id per distinct (input, solution) pair, and the number of turns after which each input's solutions refill.
    filled = (sets != IGNORE_LABEL_ID).any(-1)
    keys = torch.cat([input_id.view(n, 1, 1).expand(n, m, 1), sets.long()], dim=-1)
    unique_keys, ids = torch.unique(keys[filled], dim=0, return_inverse=True)
    solution_id = torch.zeros((n, m), dtype=torch.long, device=device)
    solution_id[filled] = ids
    solution_input = unique_keys[:, 0]
    distinct = torch.bincount(solution_input, minlength=num_inputs)
    refill_after = torch.ceil(distinct.double() * refill_fraction).long().clamp_min(1)

    # Turn order: sort rows by best CE, then stably by input, and number each input's rows from 0.
    order = torch.argsort(ce.min(-1).values, stable=True)
    order = order[torch.argsort(input_id[order], stable=True)]
    rows_per_input = torch.bincount(input_id, minlength=num_inputs)
    first = torch.cumsum(rows_per_input, 0) - rows_per_input  # where each input starts in ``order``
    turn = torch.empty_like(input_id)
    turn[order] = torch.arange(n, device=device) - first[input_id[order]]

    nearest = ce.argmin(-1)
    slot = nearest.clone()
    rows = torch.arange(n, device=device)
    taken = torch.zeros(len(unique_keys), dtype=torch.bool, device=device)
    turns_since_refill = torch.zeros(num_inputs, dtype=torch.long, device=device)
    for t in range(int(rows_per_input.max())):
        playing = turn == t
        free = filled & ~taken[solution_id]
        pick = torch.where(free, ce, torch.inf).argmin(-1)
        pick = torch.where(free.any(-1), pick, nearest)
        slot = torch.where(playing, pick, slot)
        taken[solution_id[rows[playing], pick[playing]]] = True
        turns_since_refill += torch.bincount(input_id[playing], minlength=num_inputs)
        refill = turns_since_refill >= refill_after
        taken &= ~refill[solution_input]
        turns_since_refill[refill] = 0
    return slot


__all__ = ["reverse_sxm_targets", "solution_sets"]
