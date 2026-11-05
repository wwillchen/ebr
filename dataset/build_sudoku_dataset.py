"""Build Sudoku-Extreme from sapientinc/sudoku-extreme (1k puzzles x 1000 augmentations)."""

from __future__ import annotations

import csv
import json
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import source_file, write_hrm_split

cli = ArgParser()

SOURCE_REPO = "sapientinc/sudoku-extreme"
SOURCE_REVISION = "58942f96baeb572ca3127e2a9e9c70f330783d6b"


class DataProcessConfig(BaseModel):
    source_repo: str = SOURCE_REPO
    source_revision: str = SOURCE_REVISION
    source_dir: str | None = None  # directory holding train.csv / test.csv (skips the download)
    output_dir: str = "data/sudoku-extreme-1k-aug-1000"
    subsample_size: int | None = 1000
    min_difficulty: int | None = None
    num_aug: int = 1000
    seed: int = 42


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """Validity-preserving random relabel / band / stack / transpose (global numpy RNG)."""
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    transpose_flag = np.random.rand() < 0.5
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        return digit_map[x.flatten()[mapping].reshape(9, 9).copy()]

    return apply(board), apply(solution)


def source_csv(config: DataProcessConfig, set_name: str) -> str:
    return source_file(config.source_repo, config.source_revision, f"{set_name}.csv", config.source_dir)


def convert_subset(set_name: str, config: DataProcessConfig) -> None:
    inputs, labels = [], []
    with open(source_csv(config, set_name), newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for _source, q, a, rating in reader:
            if config.min_difficulty is None or int(rating) >= config.min_difficulty:
                assert len(q) == 81 and len(a) == 81
                inputs.append(np.frombuffer(q.replace(".", "0").encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))

    if set_name == "train" and config.subsample_size is not None and config.subsample_size < len(inputs):
        idx = np.random.choice(len(inputs), size=config.subsample_size, replace=False)
        inputs = [inputs[i] for i in idx]
        labels = [labels[i] for i in idx]

    num_aug = config.num_aug if set_name == "train" else 0
    rows_in, rows_out, puzzle_indices, group_indices = [], [], [0], [0]
    for orig_inp, orig_out in zip(inputs, labels):
        for aug_idx in range(1 + num_aug):
            inp, out = (orig_inp, orig_out) if aug_idx == 0 else shuffle_sudoku(orig_inp, orig_out)
            rows_in.append(inp)
            rows_out.append(out)
            puzzle_indices.append(len(rows_in))
        group_indices.append(len(puzzle_indices) - 1)

    def to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1

    arrays = {
        "inputs": to_numpy(rows_in),
        "labels": to_numpy(rows_out),
        "group_indices": np.array(group_indices, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "puzzle_identifiers": np.zeros(len(rows_in), dtype=np.int32),
    }
    metadata = dict(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=11,
        seq_len=81,
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1.0,
        sets=["all"],
        total_samples=len(rows_in),
    )
    write_hrm_split(config.output_dir, set_name, arrays, metadata)
    print(f"[sudoku/{set_name}] {len(rows_in)} examples, {len(group_indices) - 1} groups")


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    np.random.seed(config.seed)
    convert_subset("train", config)
    convert_subset("test", config)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
