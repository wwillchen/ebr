"""What the dataset builders and the data loader share: the on-disk layout and its metadata, and the dihedral
transforms.

A split is a directory ``<data>/<split>/`` of ``all__<field>.npy`` arrays and a ``dataset.json``:

* ``inputs`` / ``labels`` ``[rows, seq_len]``: one example per row (token ids, 0 = padding);
* ``puzzle_indices``: the rows of puzzle p are ``puzzle_indices[p]:puzzle_indices[p + 1]``. A multi-solution input
  (N-Queens, Graph Coloring) is one puzzle with one row per solution;
* ``group_indices``: the puzzles of group g are ``group_indices[g]:group_indices[g + 1]``. A group is a puzzle with
  its augmentations, and training draws one puzzle per group;
* ``puzzle_identifiers``: a per-puzzle id (0 here).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pydantic

# Maze-Hard cell characters; token id = index + 1 (0 is padding): wall, empty, start, goal, path.
MAZE_CHARSET = "# SGo"


class PuzzleDatasetMetadata(pydantic.BaseModel):
    """``<split>/dataset.json``. Fields a builder leaves out are None; builder-specific fields (Graph Coloring's
    ``nodes``) are kept as extras."""

    model_config = pydantic.ConfigDict(extra="allow")

    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int | None = None
    sets: list[str]
    total_samples: int | None = None


def read_metadata(dataset_path: str, split: str) -> PuzzleDatasetMetadata:
    """``<dataset_path>/<split>/dataset.json``, validated."""
    with open(os.path.join(dataset_path, split, "dataset.json"), encoding="utf-8") as f:
        return PuzzleDatasetMetadata(**json.load(f))


def source_file(repo: str, revision: str, filename: str, source_dir: str | None = None) -> str:
    """Local path of ``filename`` from the Hugging Face dataset ``repo`` at ``revision`` (downloaded and cached), or
    ``<source_dir>/<filename>`` when ``source_dir`` is given."""
    if source_dir is not None:
        return os.path.join(source_dir, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


# The 8 symmetries of a square grid, by id, and the id of each one's inverse.
_DIHEDRAL = (
    lambda a: a,  # identity
    lambda a: np.rot90(a, k=1),
    lambda a: np.rot90(a, k=2),
    lambda a: np.rot90(a, k=3),
    np.fliplr,  # horizontal flip
    np.flipud,  # vertical flip
    lambda a: a.T,  # reflection in the main diagonal
    lambda a: np.fliplr(np.rot90(a, k=1)),  # reflection in the anti-diagonal
)
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Symmetry ``tid`` (0..7) of a 2D grid; any other id leaves it unchanged."""
    return _DIHEDRAL[tid](arr) if 0 <= tid < len(_DIHEDRAL) else arr


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def write_hrm_split(output_dir: str, split: str, arrays: dict, metadata: dict, indent=None) -> None:
    """Write ``<output_dir>/<split>/dataset.json`` and ``all__<name>.npy``."""
    save_dir = os.path.join(output_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=indent)
    for name, value in arrays.items():
        np.save(os.path.join(save_dir, f"all__{name}.npy"), value)
