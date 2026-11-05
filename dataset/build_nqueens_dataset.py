"""Build N-Queens 8 / 10 on GRAM's split (no download)."""

from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import write_hrm_split

PAD, EMPTY, QUEEN = 0, 1, 2
REMOVE_COUNTS = {8: [5, 6, 7], 10: [7, 8, 9]}
SOURCE_NOTE = "GRAM's split (ahn-ml/gram-release f2e32b9 dataset/build_nqueens_dataset.py, seed 42, test_ratio 0.15)"
# GRAM's builder writes each split's puzzles in Python set order, which changes from build to build; this file pins
# one order.
GRAM_ORDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nqueens_order.npz")

cli = ArgParser()


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str | None = None  # default data/nqueens-<n>
    remove_counts: list[int] | None = None  # default GRAM's: 5 6 7 (N=8), 7 8 9 (N=10)
    seed: int = 42
    test_ratio: float = 0.15


def nqueens_solutions(n: int) -> list[np.ndarray]:
    """All solutions as (n, n) uint8 boards, GRAM's backtracking order (row by row, columns ascending)."""
    out: list[np.ndarray] = []

    def solve(cols: list[int], row: int) -> None:
        if row == n:
            board = np.zeros((n, n), dtype=np.uint8)
            board[np.arange(n), cols] = 1
            out.append(board)
            return
        for c in range(n):
            if all(c != pc and abs(pc - c) != row - r for r, pc in enumerate(cols)):
                solve(cols + [c], row + 1)

    solve([], 0)
    return out


def gram_split(n: int, remove_counts: list[int], seed: int, test_ratio: float):
    """GRAM's split: ``(keys, test_size, rows_by_key, solutions)``.

    ``keys`` are the inputs in shuffled order; the first ``test_size`` are the test inputs. ``rows_by_key`` maps an
    input (uint8 board bytes) to its list of (input, label) uint8 boards.
    """
    solutions = nqueens_solutions(n)
    rows_by_key: dict[bytes, list] = {}
    for sol in solutions:
        queens = list(zip(*np.where(sol == 1)))
        for k in remove_counts:
            if k > len(queens):
                continue
            for removed in combinations(queens, k):
                inp = sol.copy()
                for r, c in removed:
                    inp[r, c] = 0
                rows_by_key.setdefault(inp.tobytes(), []).append((inp, sol.copy()))
    keys = list(rows_by_key.keys())
    np.random.seed(seed)
    np.random.shuffle(keys)
    return keys, int(len(keys) * test_ratio), rows_by_key, solutions


def build(config: DataProcessConfig) -> str:
    """Write the train / test splits. Each input is one puzzle whose rows are all its completions (sorted), so a
    batch holds every solution of an input together."""
    n = config.n
    remove_counts = config.remove_counts or REMOVE_COUNTS[n]
    output_dir = config.output_dir or f"data/nqueens-{n}"
    keys, test_size, rows_by_key, solutions = gram_split(n, remove_counts, config.seed, config.test_ratio)
    split_keys = {"test": keys[:test_size], "train": keys[test_size:]}
    # GRAM's default split gets the stored puzzle order; any other split keeps the order of the shuffled input list.
    if n in REMOVE_COUNTS and remove_counts == REMOVE_COUNTS[n] and (config.seed, config.test_ratio) == (42, 0.15):
        with np.load(GRAM_ORDER) as z:
            for split in ("train", "test"):
                order = z[f"n{n}_{split}"]
                if sorted(order.tolist()) != list(range(len(split_keys[split]))):
                    raise ValueError(f"{GRAM_ORDER} does not match the {split} split")
                split_keys[split] = [split_keys[split][i] for i in order]

    solution_tokens = np.stack([s.reshape(-1) for s in solutions]).astype(np.int32) + 1
    stats = {
        "N": n,
        "k_removed": list(remove_counts),
        "num_complete_solutions": len(solutions),
        "seq_len": n * n,
        "vocab": {"pad": PAD, "empty": EMPTY, "queen": QUEEN},
        "source": SOURCE_NOTE,
        "splits": {},
    }
    all_counts: dict[int, int] = {}
    for split in ("train", "test"):
        split_inputs = split_keys[split]
        inputs, labels, puzzle_indices, solution_counts = [], [], [0], []
        for key in split_inputs:
            rows = rows_by_key[key]
            x = rows[0][0].reshape(-1).astype(np.int32) + 1
            completions = sorted((r[1].reshape(-1).astype(np.int32) + 1 for r in rows), key=lambda y: y.tobytes())
            given = x == QUEEN
            expected = {s.tobytes() for s in solution_tokens if (s[given] == QUEEN).all()}
            assert {y.tobytes() for y in completions} == expected and len(completions) == len(expected), (
                "rows are not exactly the completions"
            )
            for y in completions:
                inputs.append(x)
                labels.append(y)
            puzzle_indices.append(len(inputs))
            solution_counts.append(len(completions))
        num = len(split_inputs)
        arrays = {
            "inputs": np.stack(inputs).astype(np.int32),
            "labels": np.stack(labels).astype(np.int32),
            "puzzle_identifiers": np.zeros(num, dtype=np.int32),
            "puzzle_indices": np.array(puzzle_indices, dtype=np.int64),
            "group_indices": np.arange(num + 1, dtype=np.int64),
            "solution_counts": np.array(solution_counts, dtype=np.int32),
        }
        metadata = {
            "seq_len": n * n,
            "vocab_size": 3,
            "pad_id": PAD,
            "ignore_label_id": PAD,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": 1,
            "total_groups": num,
            "mean_puzzle_examples": len(inputs) / num,
            "sets": ["all"],
        }
        write_hrm_split(output_dir, split, arrays, metadata, indent=2)
        counts: dict[int, int] = {}
        for c in solution_counts:
            counts[c] = counts.get(c, 0) + 1
            all_counts[c] = all_counts.get(c, 0) + 1
        stats["splits"][split] = {
            "unique_inputs": num,
            "pairs": len(inputs),
            "solution_count_distribution": {str(c): counts[c] for c in sorted(counts)},
        }
        print(f"[nqueens-{n}/{split}] {num} unique inputs, {len(inputs)} rows")
    stats["num_unique_inputs"] = len(keys)
    stats["solution_count_distribution"] = {str(c): all_counts[c] for c in sorted(all_counts)}
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        f.write(json.dumps(stats, indent=2))
    with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
        f.write(json.dumps(["<blank>"]))
    return output_dir


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    build(config)


if __name__ == "__main__":
    cli()
