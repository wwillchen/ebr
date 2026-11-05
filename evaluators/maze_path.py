"""Maze-Hard path metrics: valid, shortest (any optimal path) and canonical (the label)."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from dataset.common import MAZE_CHARSET as CHARSET

WALL, EMPTY, START, GOAL, PATH = (CHARSET.index(c) + 1 for c in "# SGo")
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_single(grid: np.ndarray, value: int) -> tuple[int, int] | None:
    ys, xs = np.nonzero(grid == value)
    return (int(ys[0]), int(xs[0])) if len(ys) == 1 else None


def bfs_distance(open_mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    n, m = open_mask.shape
    dist = np.full((n, m), -1, dtype=np.int32)
    dist[start] = 0
    q: deque = deque([start])
    while q:
        y, x = q.popleft()
        if (y, x) == goal:
            return int(dist[y, x])
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < m and open_mask[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return -1


def path_is_simple_chain(path_mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """True when the marked cells form one path from ``start`` to ``goal``: both ends have one marked neighbor,
    every other cell two, and walking from ``start`` reaches ``goal`` through every marked cell."""
    n, m = path_mask.shape
    cells = int(path_mask.sum())
    if cells < 2 or not path_mask[start] or not path_mask[goal] or start == goal:
        return False
    padded = np.pad(path_mask, 1)
    neighbors = padded[:-2, 1:-1].astype(np.int32) + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    neighbors = np.where(path_mask, neighbors, -1)
    if neighbors[start] != 1 or neighbors[goal] != 1:
        return False
    interior = path_mask.copy()
    interior[start] = interior[goal] = False
    if np.any(neighbors[interior] != 2):
        return False
    visited, previous, current = 1, None, start
    while current != goal:
        step = None
        for dy, dx in DIRS:
            ny, nx = current[0] + dy, current[1] + dx
            if 0 <= ny < n and 0 <= nx < m and path_mask[ny, nx] and (ny, nx) != previous:
                step = (ny, nx)
                break
        if step is None:
            return False
        previous, current = current, step
        visited += 1
        if visited > cells:
            return False
    return visited == cells


def score_maze(inp: np.ndarray, pred: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """One maze: is the predicted path valid (a single path from start to goal through open cells), shortest, and
    equal to the label; plus whether it crosses a wall and how many path cells it is off by."""
    n = int(round(inp.size**0.5))
    inp, pred, label = inp.reshape(n, n), pred.reshape(n, n), label.reshape(n, n)
    out = dict(valid_path=0.0, shortest_path=0.0, canonical_path=0.0, wall_violation=0.0, path_cells_err=0.0)
    start, goal = _find_single(inp, START), _find_single(inp, GOAL)
    open_mask = inp != WALL
    pred_path = pred == PATH
    out["path_cells_err"] = float(abs(int(pred_path.sum()) - int((label == PATH).sum())))
    out["wall_violation"] = float(bool(np.any(pred_path & ~open_mask)))
    out["canonical_path"] = float(np.array_equal(pred, label))
    if start is None or goal is None or out["wall_violation"]:
        return out
    path_mask = pred_path.copy()
    path_mask[start] = path_mask[goal] = True
    if not path_is_simple_chain(path_mask, start, goal):
        return out
    out["valid_path"] = 1.0
    shortest = bfs_distance(open_mask, start, goal)
    out["shortest_path"] = float(shortest >= 0 and int(path_mask.sum()) - 1 == shortest)
    return out


class MazePath:
    """In-loop evaluator: averages ``score_maze`` over the test mazes (rows that are all padding are skipped)."""

    required_outputs = {"inputs", "labels", "preds"}

    def __init__(self, data_path: str, eval_metadata: Any, **_: Any) -> None:
        self.pad_id = int(getattr(eval_metadata, "pad_id", 0))
        self._sums: dict[str, float] = {}
        self._count = 0

    def begin_eval(self) -> None:
        self._sums, self._count = {}, 0

    def update_batch(self, batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        inputs = batch["inputs"].detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        preds = outputs["preds"].detach().cpu().numpy()
        for i in np.nonzero(np.any(inputs != self.pad_id, axis=-1))[0]:
            for k, v in score_maze(inputs[i], preds[i], labels[i]).items():
                self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._count += 1

    def result(self) -> dict[str, float] | None:
        if self._count == 0:
            return None
        metrics = {f"maze/{k}": v / self._count for k, v in self._sums.items()}
        metrics["maze/tiebreak_gap"] = metrics["maze/shortest_path"] - metrics["maze/canonical_path"]
        metrics["maze/count"] = float(self._count)
        return metrics


EVALUATORS = {"maze_path@MazePath": MazePath}

__all__ = ["EVALUATORS", "MazePath", "bfs_distance", "path_is_simple_chain", "score_maze"]
