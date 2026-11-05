"""Build Maze-Hard 30x30 from sapientinc/maze-30x30-hard-1k."""

from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import MAZE_CHARSET as CHARSET
from dataset.common import dihedral_transform, source_file, write_hrm_split

SOURCE_REPO = "sapientinc/maze-30x30-hard-1k"
SOURCE_REVISION = "549754de3d67eebe57b6b22c7d226b9412d55b07"

cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = SOURCE_REPO
    source_revision: str = SOURCE_REVISION
    source_dir: str | None = None  # directory holding train.csv / test.csv (skips the download)
    output_dir: str = "data/maze-30x30-hard-1k"
    subsample_size: int | None = None
    aug: bool = False
    seed: int = 42


def source_csv(config: DataProcessConfig, set_name: str) -> str:
    return source_file(config.source_repo, config.source_revision, f"{set_name}.csv", config.source_dir)


def convert_subset(set_name: str, config: DataProcessConfig) -> None:
    all_chars, grid, inputs, labels = set(), None, [], []
    with open(source_csv(config, set_name), newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for _source, q, a, _rating in reader:
            all_chars.update(q)
            all_chars.update(a)
            if grid is None:
                n = int(len(q) ** 0.5)
                assert n * n == len(q), f"non-square maze of length {len(q)}"
                grid = (n, n)
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid))
    assert not (all_chars - set(CHARSET)), f"unexpected chars: {all_chars - set(CHARSET)}"

    if set_name == "train" and config.subsample_size is not None and config.subsample_size < len(inputs):
        idx = np.random.default_rng(config.seed).choice(len(inputs), size=config.subsample_size, replace=False)
        inputs = [inputs[i] for i in idx]
        labels = [labels[i] for i in idx]

    rows_in, rows_out, puzzle_indices, group_indices = [], [], [0], [0]
    for inp, out in zip(inputs, labels):
        for aug_idx in range(8 if (set_name == "train" and config.aug) else 1):
            rows_in.append(dihedral_transform(inp, aug_idx))
            rows_out.append(dihedral_transform(out, aug_idx))
            puzzle_indices.append(len(rows_in))
        group_indices.append(len(puzzle_indices) - 1)

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1
    arrays = {
        "inputs": np.vstack([char2id[s.reshape(-1)] for s in rows_in]),
        "labels": np.vstack([char2id[s.reshape(-1)] for s in rows_out]),
        "group_indices": np.array(group_indices, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "puzzle_identifiers": np.zeros(len(rows_in), dtype=np.int32),
    }
    metadata = dict(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=len(CHARSET) + 1,
        seq_len=int(math.prod(grid)),
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1.0,
        sets=["all"],
        total_samples=None,
    )
    write_hrm_split(config.output_dir, set_name, arrays, metadata)
    print(f"[maze/{set_name}] {len(rows_in)} examples ({grid[0]}x{grid[1]})")


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    convert_subset("train", config)
    convert_subset("test", config)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
