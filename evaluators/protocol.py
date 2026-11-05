"""Sudoku / Maze scoring rules on the per-sample arrays the runners record (N-Queens / Graph Coloring: gram.py)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def lowest_energy_pick(energy: np.ndarray, n: int) -> np.ndarray:
    """energy [M, B] -> index of the lowest-energy sample among the first n per row."""
    return np.asarray(energy)[:, :n].argmin(1)


# ------------------------------------------------------------------------------------------------- Sudoku
SUDOKU_NS = (1, 2, 5, 8)


def sudoku_cells(
    exact: np.ndarray,
    energy: np.ndarray,
    board: np.ndarray,
    breadth: int,
    depths: Sequence[int] = (16, 64),
) -> dict[str, dict[str, float]]:
    """Per depth: how many boards are solved by sample 0 (pass@1), by the lowest-energy of the first n samples
    (pass@n) and by any sample (oracle).

    ``exact`` / ``energy`` are [depth, rows] with rows = boards x breadth (a board's samples contiguous) and ``board``
    the board of each row. Counts, not percentages, so the shards can be merged exactly (``merge_counts``).
    """
    exact = np.asarray(exact).astype(bool)
    _, rows = exact.shape
    if rows % breadth:
        raise ValueError("rows are not a multiple of the breadth")
    board_of_sample = np.asarray(board).reshape(-1, breadth)
    if not (board_of_sample == board_of_sample[:, :1]).all():
        raise ValueError("samples of one board are not contiguous")
    boards = rows // breadth
    board_ids = np.arange(boards)
    out = {}
    for depth in depths:
        solved = exact[depth - 1].reshape(boards, breadth)
        energy_at_depth = np.asarray(energy[depth - 1]).reshape(boards, breadth)
        counts = {"boards": boards, "pass@1": int(solved[:, 0].sum()), "oracle": int(solved.any(1).sum())}
        for n in SUDOKU_NS:
            counts[f"pass@{n}"] = int(solved[board_ids, lowest_energy_pick(energy_at_depth, n)].sum())
        out[f"D{depth}"] = counts
    return out


def merge_counts(parts: Sequence[dict[str, dict[str, int]]]) -> dict[str, dict[str, float]]:
    """Sum the shards' counts and return percentages per depth."""
    out = {}
    for key in parts[0]:
        total = {k: sum(p[key][k] for p in parts) for k in parts[0][key]}
        out[key] = {
            "boards": total["boards"],
            **{k: 100.0 * v / total["boards"] for k, v in total.items() if k != "boards"},
        }
    return out


# --------------------------------------------------------------------------------------------------- Maze
def maze_cells(arr: dict[str, np.ndarray], n: int = 5, segment: int = 16) -> dict[str, float]:
    """Percent of mazes solved at ``segment`` by sample 0 (start 1) and by the lowest-energy of the n samples (pass@n),
    for the canonical, shortest and valid path; oracle over the n samples for the canonical one.

    ``arr``: [segments, rows] arrays (exact, shortest, valid, energy), rows = mazes x n.
    """
    mazes = arr["exact"].shape[1] // n

    def at_segment(key: str) -> np.ndarray:
        return np.asarray(arr[key][segment - 1]).reshape(mazes, n)

    pick = at_segment("energy").astype(np.float64).argmin(1)
    maze_ids = np.arange(mazes)
    out = {"mazes": mazes}
    for key, name in (("exact", "canonical"), ("shortest", "shortest"), ("valid", "valid")):
        solved = at_segment(key).astype(bool)
        out[f"start1_{name}"] = 100.0 * solved[:, 0].mean()
        out[f"pass@{n}_{name}"] = 100.0 * solved[maze_ids, pick].mean()
    out[f"oracle_{n}_canonical"] = 100.0 * at_segment("exact").astype(bool).any(1).mean()
    return out
