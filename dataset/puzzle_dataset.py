"""The data loader: whole batches from a split on disk (layout in ``dataset/common.py``), resumable mid-epoch.

Training batches: each pass over the data shuffles the groups, and every group adds one of its puzzles, drawn at
random, with all of that puzzle's rows (in random order). A multi-solution input therefore brings its solutions into
the batch together, which reverse SXM relies on. A pass runs ``epochs_per_iter`` epochs, and the sampler position and
RNG are saved in checkpoints (``state_dict``), so a resumed run sees the same batches.

Test batches: the rows in order, in batches of ``global_batch_size``, the last one padded.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any

import numpy as np
import pydantic
import torch
from torch.utils.data import DataLoader, IterableDataset

from dataset.common import PuzzleDatasetMetadata, read_metadata
from models.losses import IGNORE_LABEL_ID

FIELDS = ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices")
MEMORY_MAPPED = ("inputs", "labels")


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int = 0
    num_replicas: int = 1
    # Speed only: when every puzzle has exactly one example (Sudoku, Maze), sample a training batch with one vectorized
    # RNG call instead of a Python loop over its groups. Same batches and the same sampler / RNG state.
    vectorized_sampler: bool = False


def _sample_batch(rng, group_order, puzzle_indices, group_indices, start_index, global_batch_size):
    """The next training batch: from group ``group_order[start_index]`` on, one random puzzle per group with all its
    rows (in random order), until the batch is full. The last puzzle is cut to fit.

    Returns (next start_index, row ids, puzzle id of each row).
    """
    rows, row_puzzles, size = [], [], 0
    while start_index < group_order.size and size < global_batch_size:
        group = group_order[start_index]
        puzzle = rng.integers(group_indices[group], group_indices[group + 1])
        start_index += 1
        first_row = puzzle_indices[puzzle]
        puzzle_rows = int(puzzle_indices[puzzle + 1] - first_row)
        take = min(puzzle_rows, global_batch_size - size)
        row_puzzles.append(np.full(take, puzzle, dtype=np.int32))
        rows.append(first_row + rng.choice(puzzle_rows, take, replace=False))
        size += take
    return start_index, np.concatenate(rows), np.concatenate(row_puzzles)


def _sample_batch_single(rng, group_order, puzzle_indices, group_indices, start_index, global_batch_size):
    """``_sample_batch`` when every puzzle has exactly one example: the same batch and the same RNG stream.

    There each group adds one row, so the loop takes the next ``min(B, remaining)`` groups and draws one
    ``integers(lo, hi)`` per group, while ``choice(1, 1, replace=False)`` returns ``[0]`` without drawing. numpy's array
    ``integers(lo_array, hi_array)`` draws the same values in the same order as those scalar calls (stream state
    included), so one call replaces the loop.
    """
    n = min(int(global_batch_size), int(group_order.size) - int(start_index))
    groups = np.asarray(group_order[start_index : start_index + n], dtype=np.int64)
    puzzle_ids = rng.integers(group_indices[groups], group_indices[groups + 1])
    return start_index + n, puzzle_indices[puzzle_ids].astype(np.int64), puzzle_ids.astype(np.int32)


class PuzzleDataset(IterableDataset):
    """Yields ``(set_name, batch, rows)``: a batch of this rank's rows (padded to the local batch size) and the
    number of real rows in the global batch."""

    def __init__(self, config: PuzzleDatasetConfig, split: str = "train") -> None:
        super().__init__()
        if not os.path.isdir(os.path.join(config.dataset_path, split)):
            raise FileNotFoundError(f"Dataset split {split} in {config.dataset_path} does not exist.")
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        assert config.global_batch_size % config.num_replicas == 0
        self.local_batch_size = config.global_batch_size // config.num_replicas
        self._data: dict[str, dict[str, np.ndarray]] | None = None
        self._iters = 0  # training passes started; pass i shuffles with seed + i
        # Sampler position, saved in checkpoints: set_names, set_idx, group_order, start_index, rng_state, epoch_idx.
        self._train_state: dict[str, Any] | None = None

    def _path(self, set_name: str, field: str) -> str:
        return os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field}.npy")

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        metadata = read_metadata(self.config.dataset_path, self.split)
        if metadata.total_samples is None:
            total = 0
            for set_name in metadata.sets:
                puzzle_indices = np.load(self._path(set_name, "puzzle_indices"), mmap_mode="r")
                if puzzle_indices.size:
                    total += int(puzzle_indices[-1])
            metadata.total_samples = total
        return metadata

    def _lazy_load(self) -> None:
        if self._data is None:
            self._data = {
                set_name: {
                    field: np.load(self._path(set_name, field), mmap_mode="r" if field in MEMORY_MAPPED else None)
                    for field in FIELDS
                }
                for set_name in self.metadata.sets
            }

    def num_batches(self) -> int | None:
        if not self.config.test_set_mode:
            return None
        self._lazy_load()
        return sum(math.ceil(len(d["inputs"]) / self.config.global_batch_size) for d in self._data.values())

    def _collate(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """int32 tensors, ignored labels as IGNORE_LABEL_ID, padded to the local batch size."""
        batch = {k: v.astype(np.int32) for k, v in batch.items()}
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
        missing = self.local_batch_size - batch["puzzle_identifiers"].size
        if missing > 0:
            fill = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }
            batch = {
                k: np.pad(v, ((0, missing),) + ((0, 0),) * (v.ndim - 1), constant_values=fill.get(k, 0))
                for k, v in batch.items()
            }
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        for set_name, data in self._data.items():
            total = len(data["inputs"])
            for start in range(0, total, self.config.global_batch_size):
                end = min(total, start + self.config.global_batch_size)
                local_start = start + self.config.rank * self.local_batch_size
                local_end = min(start + (self.config.rank + 1) * self.local_batch_size, end)
                # The puzzle of each row: the last puzzle that starts at or before it.
                puzzles = []
                p = np.searchsorted(data["puzzle_indices"], local_start, side="right") - 1
                for row in range(local_start, local_end):
                    while p + 1 < len(data["puzzle_indices"]) and row >= data["puzzle_indices"][p + 1]:
                        p += 1
                    puzzles.append(p)
                batch = {
                    "inputs": data["inputs"][local_start:local_end],
                    "labels": data["labels"][local_start:local_end],
                    "puzzle_identifiers": data["puzzle_identifiers"][puzzles],
                }
                yield set_name, self._collate(batch), end - start

    def _single_example(self, set_name: str) -> bool:
        """True when every puzzle of ``set_name`` has exactly one example (the vectorized sampler's precondition)."""
        cache = self.__dict__.setdefault("_single_cache", {})
        if set_name not in cache:
            cache[set_name] = bool(np.all(np.diff(np.asarray(self._data[set_name]["puzzle_indices"])) == 1))
        return cache[set_name]

    def _iter_train(self):
        """One training pass: ``epochs_per_iter`` epochs over every set, resuming from ``_train_state`` if set."""
        set_names = list(self._data.keys())
        if self._train_state is None:
            self._train_state = {
                "set_names": set_names,
                "set_idx": 0,
                "group_order": None,
                "start_index": 0,
                "rng_state": None,
                "epoch_idx": 0,
            }
        elif self._train_state.get("set_names") != set_names:
            raise RuntimeError("Dataset structure changed between checkpoints; cannot resume.")
        state = self._train_state
        while state["set_idx"] < len(set_names):
            set_name = set_names[state["set_idx"]]
            data = self._data[set_name]
            sample = _sample_batch
            if self.config.vectorized_sampler and self._single_example(set_name):
                sample = _sample_batch_single

            # A new pass seeds a fresh shuffle RNG; a resumed one continues the saved RNG, epoch and group order.
            if state.get("rng_state") is None:
                self._iters += 1
                rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            else:
                rng = np.random.Generator(np.random.Philox())
                rng.bit_generator.state = state["rng_state"]
            epoch = int(state.get("epoch_idx", 0))
            saved_order = state.get("group_order")
            group_order = np.asarray(saved_order if saved_order is not None else [], dtype=np.int64)
            start_index = int(state.get("start_index", 0))

            # Run epochs until the pass has run ``epochs_per_iter`` of them; a resumed pass first finishes its epoch.
            while epoch < self.config.epochs_per_iter or start_index < group_order.size:
                if start_index >= group_order.size:  # a new epoch: reshuffle the groups
                    group_order = rng.permutation(data["group_indices"].size - 1)
                    start_index = 0
                    epoch += 1
                    if group_order.size == 0:
                        break
                while start_index < group_order.size:
                    start_index, rows, row_puzzles = sample(
                        rng,
                        group_order,
                        data["puzzle_indices"],
                        data["group_indices"],
                        start_index,
                        self.config.global_batch_size,
                    )
                    state.update(
                        start_index=start_index,
                        rng_state=rng.bit_generator.state,
                        group_order=group_order.tolist(),
                        epoch_idx=epoch,
                    )
                    if row_puzzles.size < self.config.global_batch_size:  # drop the epoch's last, partial batch
                        start_index = group_order.size
                        break
                    local = slice(
                        self.config.rank * self.local_batch_size, (self.config.rank + 1) * self.local_batch_size
                    )
                    batch = {
                        "inputs": data["inputs"][rows[local]],
                        "labels": data["labels"][rows[local]],
                        "puzzle_identifiers": data["puzzle_identifiers"][row_puzzles[local]],
                    }
                    yield set_name, self._collate(batch), row_puzzles.size
                group_order = np.empty((0,), dtype=np.int64)
            state.update(group_order=None, start_index=0, rng_state=None, epoch_idx=0)
            state["set_idx"] += 1
        self._train_state = None

    def __iter__(self):
        self._lazy_load()
        yield from (self._iter_test() if self.config.test_set_mode else self._iter_train())

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"split": self.split, "_iters": self._iters, "train_state": self._train_state})

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            self._train_state, self._iters = None, 0
            return
        if state.get("split", self.split) != self.split:
            raise ValueError(f"state for split {state.get('split')} loaded into split {self.split}")
        self._iters = int(state.get("_iters", 0))
        saved = state.get("train_state")
        if not saved:
            self._train_state = None
            return
        self._train_state = {
            "set_names": list(saved.get("set_names", [])),
            "set_idx": int(saved.get("set_idx", 0)),
            "start_index": int(saved.get("start_index", 0)),
            "epoch_idx": int(saved.get("epoch_idx", 0)),
            "group_order": list(saved["group_order"]) if saved.get("group_order") is not None else None,
            "rng_state": copy.deepcopy(saved.get("rng_state")),
        }


def create_dataloader(
    config: Any,
    split: str,
    *,
    test_set_mode: bool,
    epochs_per_iter: int,
    global_batch_size: int,
    rank: int = 0,
    num_replicas: int = 1,
) -> tuple[DataLoader, PuzzleDatasetMetadata]:
    """A loader over ``config.dataset.data_path``/``split`` (a ``PretrainConfig``) and the split's metadata."""
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=config.seed,
            dataset_path=config.dataset.data_path,
            global_batch_size=global_batch_size,
            test_set_mode=test_set_mode,
            epochs_per_iter=epochs_per_iter,
            rank=rank,
            num_replicas=num_replicas,
            vectorized_sampler=config.dataset.vectorized_sampler,
        ),
        split=split,
    )
    # batch_size=None: the dataset yields whole batches. num_workers=0 keeps the sampler state in sync with the steps.
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=torch.cuda.is_available()), dataset.metadata


__all__ = ["PuzzleDataset", "PuzzleDatasetConfig", "PuzzleDatasetMetadata", "create_dataloader", "read_metadata"]
