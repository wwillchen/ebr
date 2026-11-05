"""Download and convert GRAM's released Graph Coloring data (cun-bjy/graph-coloring, pinned revision)."""

from __future__ import annotations

import json
import os

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.common import source_file

SOURCE_REPO = "cun-bjy/graph-coloring"
SOURCE_REVISION = "08d5f1f8791a532cfb48fac88660e285fc72d857"
RELEASE_SIZES = {
    8: dict(train_rows=7002, train_graphs=2059, test_graphs=255, test_solutions=912),
    10: dict(train_rows=13465, train_graphs=1542, test_graphs=192, test_solutions=1783),
}

cli = ArgParser()


class DataProcessConfig(BaseModel):
    nodes: int = 8
    output_dir: str | None = None  # default data/graph-coloring-<nodes>
    source_dir: str | None = None  # directory holding graph-coloring-<nodes>/*.npy (skips the download)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_source(nodes: int, source_dir: str | None = None) -> dict[str, np.ndarray]:
    out = {}
    for name in ("train_adjacencies", "train_colorings", "test_adjacencies", "test_colorings"):
        path = source_file(SOURCE_REPO, SOURCE_REVISION, f"graph-coloring-{nodes}/{name}.npy", source_dir)
        out[name] = np.load(path, allow_pickle=False)
    return out


def canonical_colorings(adj: np.ndarray) -> np.ndarray:
    """All proper colorings with at most 3 colors, as restricted-growth strings (tokens 3..5), lexicographic."""
    n = len(adj)
    colors, sols = [0] * n, []

    def visit(v: int, used: int) -> None:
        if v == n:
            sols.append(tuple(c + 3 for c in colors))
            return
        for c in range(min(used + 1, 3)):
            if any(adj[v, u] and colors[u] == c for u in range(v)):
                continue
            colors[v] = c
            visit(v + 1, max(used, c + 1))

    visit(0, 0)
    return np.asarray(sols, dtype=np.int32).reshape(-1, n)


def _is_canonical(row) -> bool:
    """True when the colors (tokens 3..5) first appear in the order 3, 4, 5: one coloring per color relabeling."""
    next_new = 3
    for c in row:
        if c > next_new:
            return False
        next_new = max(next_new, c + 1)
    return True


def convert(src: dict[str, np.ndarray], nodes: int, output_dir: str) -> None:
    """Check the released arrays and write the train / test splits.

    The release stores one row per (graph, coloring), the rows of a graph contiguous. Every graph appears once (no
    train / test overlap). Train lists every canonical coloring of each graph once; test lists one row per graph, and
    is written with all its canonical colorings so that coverage can be scored. A row's input is the upper triangle of
    the adjacency matrix (tokens 1 / 2), its label the coloring (tokens 3..5) padded to the input length.
    """
    sizes = RELEASE_SIZES[nodes]
    upper_i, upper_j = np.triu_indices(nodes, 1)
    seen: set[bytes] = set()
    for split in ("train", "test"):
        adj, col = src[f"{split}_adjacencies"], src[f"{split}_colorings"]
        _require(np.all((adj == 0) | (adj == 1)) and np.all((col >= 0) & (col <= 2)), f"{split}: bad values")
        graphs, all_colorings = _checked_graphs(split, adj, col, seen)
        counts = np.asarray([len(c) for c in all_colorings], dtype=np.int64)
        _require(len(graphs) == sizes[f"{split}_graphs"], f"{split}: graph count mismatch")
        if split == "train":
            _require(len(adj) == sizes["train_rows"], "train: row count mismatch")
            inputs = adj[:, upper_i, upper_j].astype(np.int32) + 1
            colorings = col.astype(np.int32) + 3
        else:
            colorings = np.concatenate(all_colorings)
            _require(len(colorings) == sizes["test_solutions"], "test: solution count mismatch")
            upper = [(g[upper_i, upper_j].astype(np.int32) + 1)[None] for g in graphs]
            inputs = np.concatenate([np.repeat(u, c, axis=0) for u, c in zip(upper, counts)])
        _write_split(os.path.join(output_dir, split), nodes, inputs, colorings, counts)
        print(f"[graph-coloring-{nodes}/{split}] {len(counts)} graphs, {len(colorings)} rows")


def _checked_graphs(
    split: str, adj: np.ndarray, col: np.ndarray, seen: set[bytes]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Walk the release's graphs (each one's contiguous rows), check them, and return each graph and all of its
    canonical colorings. ``seen`` collects the graphs across splits, so a train / test overlap is caught."""
    graphs, all_colorings, begin = [], [], 0
    while begin < len(adj):
        a = adj[begin].astype(bool)
        _require(not a.diagonal().any() and np.array_equal(a, a.T), f"{split}: not a simple graph")
        key = a.tobytes()
        _require(key not in seen, f"{split}: duplicate graph or train/test overlap")
        seen.add(key)
        end = begin + 1
        if split == "train":
            while end < len(adj) and np.array_equal(adj[end], adj[begin]):
                end += 1
        colorings = canonical_colorings(a)
        allowed = {tuple(r) for r in colorings}
        given = col[begin:end] + 3
        _require(all(_is_canonical(r) and tuple(r) in allowed for r in given), f"{split}: improper target")
        if split == "train":
            _require(
                {tuple(r) for r in given} == allowed and len(given) == len(allowed),
                "train: each graph must list every canonical coloring once",
            )
        graphs.append(a)
        all_colorings.append(colorings)
        begin = end
    return graphs, all_colorings


def _write_split(out_dir: str, nodes: int, inputs: np.ndarray, colorings: np.ndarray, counts: np.ndarray) -> None:
    """One split in the loader's layout: rows of (input, coloring padded to the input width), grouped by graph."""
    width = nodes * (nodes - 1) // 2
    arrays = dict(
        inputs=inputs,
        labels=np.pad(colorings, ((0, 0), (0, width - nodes))),
        solution_counts=counts,
        puzzle_identifiers=np.zeros(len(counts), dtype=np.int32),
        puzzle_indices=np.concatenate(([0], np.cumsum(counts))),
        group_indices=np.arange(len(counts) + 1, dtype=np.int64),
    )
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"all__{name}.npy"), arr, allow_pickle=False)
    metadata = dict(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=6,
        seq_len=width,
        num_puzzle_identifiers=1,
        total_groups=len(counts),
        total_puzzles=len(counts),
        total_samples=len(colorings),
        mean_puzzle_examples=float(len(colorings) / len(counts)),
        sets=["all"],
        nodes=nodes,
        native_label_width=nodes,
        label_width=width,
        labels_padded_for_ebr=True,
    )
    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        f.write(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")


@cli.command(singleton=True)
def main(config: DataProcessConfig) -> None:
    _require(config.nodes in RELEASE_SIZES, "nodes must be 8 or 10")
    output_dir = config.output_dir or f"data/graph-coloring-{config.nodes}"
    convert(load_source(config.nodes, config.source_dir), config.nodes, output_dir)


if __name__ == "__main__":
    cli()
