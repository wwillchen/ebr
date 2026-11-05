"""N-Queens / Graph Coloring: sample the model (``run``), pick each sample's stopping segment, and score.

Every unique test input gets the same ``samples`` draws. Answers are checked against the puzzle rules. N-Queens: valid
= n non-attacking queens that keep the given queens. Graph Coloring: valid = no conflict and canonical color order.

Cov@20 is one rule on both tasks: the distinct valid solutions an input's samples find, at most 20 per input, summed
over the inputs and divided by the sum of min(|S(x)|, 20), where S(x) is the input's full solution set.
"""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluators.common import check_eval_data, eval_config, load_checkpoint, load_model, measure, write_result

COV_CAP = 20  # solutions per input that 20 samples can cover


# ---------------------------------------------------------------------------------------------------- N-Queens
def nqueens_solutions(board: np.ndarray, n: int):
    """The first ``COV_CAP`` completions of a partial board (depth-first over the empty rows, columns ascending)."""
    queens = [(r, c) for r, c in np.argwhere(board == 2)]
    cols, d1, d2 = {c for _, c in queens}, {r - c for r, c in queens}, {r + c for r, c in queens}
    rows = [r for r in range(n) if r not in {r for r, _ in queens}]
    out = []

    def dfs(i, placed):
        if len(out) >= COV_CAP:
            return
        if i == len(rows):
            b = np.ones((n, n), np.int64)
            for r, c in queens + placed:
                b[r, c] = 2
            out.append(b.reshape(-1).tobytes())
            return
        r = rows[i]
        for c in range(n):
            if c in cols or r - c in d1 or r + c in d2:
                continue
            cols.add(c), d1.add(r - c), d2.add(r + c)
            dfs(i + 1, placed + [(r, c)])
            cols.discard(c), d1.discard(r - c), d2.discard(r + c)

    dfs(0, [])
    return out


def nqueens_valid(board: np.ndarray, n: int) -> bool:
    """n queens, no two sharing a row, column or diagonal."""
    pos = np.argwhere(board.reshape(n, n) == 2)
    if len(pos) != n:
        return False
    r, c = pos[:, 0], pos[:, 1]
    return len(set(r)) == len(set(c)) == len(set(r - c)) == len(set(r + c)) == n


def score_nqueens(inputs: np.ndarray, predictions: np.ndarray, rows_per_input: np.ndarray) -> dict[str, float]:
    """inputs [U, n*n] (1 empty, 2 queen), predictions [S, U, n*n], rows_per_input [U] (test rows = solutions).

    ``acc_n1``: sample 0 is a completion (valid and keeps the given queens), averaged over the test rows.
    ``cov20``: distinct completions found, at most 20 per input, over the sum of min(|S(x)|, 20) (as Graph Coloring).
    """
    S, U, L = predictions.shape
    n = int(round(L**0.5))
    if n * n != L or inputs.shape != (U, L) or len(rows_per_input) != U:
        raise ValueError("expected inputs [U, n*n], predictions [S, U, n*n], rows_per_input [U]")
    w = np.asarray(rows_per_input, np.float64) / np.sum(rows_per_input)
    acc = np.zeros(U)
    found_capped = total_capped = 0
    for u in range(U):
        board = inputs[u].reshape(n, n)
        given = board == 2
        completes = [nqueens_valid(p, n) and not (given & (p.reshape(n, n) != 2)).any() for p in predictions[:, u]]
        acc[u] = completes[0]
        found = len({p.astype(np.int64).tobytes() for p, ok in zip(predictions[:, u], completes) if ok})
        found_capped += min(found, COV_CAP)
        total_capped += len(nqueens_solutions(board, n))  # min(|S(x)|, 20)
    return {"acc_n1": float(acc @ w), "cov20": found_capped / total_capped}


# ----------------------------------------------------------------------------------------------- Graph Coloring
def coloring_adjacency(x: np.ndarray, n: int) -> np.ndarray:
    """Input encoding: strict upper triangle, row-major, 1 = no edge, 2 = edge."""
    adj = np.zeros((n, n), bool)
    adj[np.triu_indices(n, 1)] = np.asarray(x) == 2
    return adj | adj.T


def canonical(colors) -> tuple:
    """Relabel colors in order of first appearance."""
    m: dict[int, int] = {}
    return tuple(m.setdefault(c, len(m)) for c in colors)


def all_colorings(adj: np.ndarray) -> set:
    """Every valid 3-coloring up to color permutation."""
    n = len(adj)
    edges = list(zip(*np.nonzero(np.triu(adj, 1))))
    return {canonical(c) for c in product(range(3), repeat=n) if all(c[i] != c[j] for i, j in edges)}


def score_graph_coloring(inputs: np.ndarray, predictions: np.ndarray, nodes: int) -> dict[str, float]:
    """inputs [U, n(n-1)/2], predictions [S, U, >= n] with colors 3.. in the first n positions."""
    S, U = predictions.shape[:2]
    if inputs.shape != (U, nodes * (nodes - 1) // 2):
        raise ValueError("expected inputs [U, n(n-1)/2] matching predictions [S, U, n]")
    found_capped = total_capped = conflicts_n1 = 0
    for u in range(U):
        adj = coloring_adjacency(inputs[u], nodes)
        edges = list(zip(*np.nonzero(np.triu(adj, 1))))
        sols = all_colorings(adj)
        got = set()
        for s in range(S):
            c = tuple(int(v) - 3 for v in predictions[s, u, :nodes])
            if all(c[i] != c[j] for i, j in edges) and canonical(c) == c:
                got.add(c)
            if s == 0:
                conflicts_n1 += sum(c[i] == c[j] for i, j in edges)
        f = len(got & sols)
        found_capped, total_capped = found_capped + min(f, COV_CAP), total_capped + min(len(sols), COV_CAP)
    return {"conflicts_n1": int(conflicts_n1), "cov20": found_capped / total_capped}


# ------------------------------------------------------------------------------------------- stopping and selection
GTAN_TAU = 0.2


def gtan_stop_index(gtan: np.ndarray, tau: float, cap: int) -> np.ndarray:
    """gtan [D, ...] -> 0-based segment index at which each sample stops: the first segment end with
    tangential |grad_M E| < tau, else segment ``cap`` (the trained depth)."""
    g = np.asarray(gtan)[:cap]
    hit = g < tau
    return np.where(hit.any(0), hit.argmax(0), cap - 1)


def gram_cells(
    task: str,
    decode: np.ndarray,
    energy_raw: np.ndarray,
    stop: np.ndarray,
    inputs: np.ndarray,
    nodes: int,
    rows_per_input=None,
) -> dict[str, float]:
    """decode [D, S, U, L], energy_raw [D, S, U], stop [S, U] (0-based segment) -> N = 1 / N = 20 / Cov@20.

    Every sample is read out at its own stop. N = 1 = sample 0; N = 20 = the lowest raw energy E(z) of the S samples.
    """
    S, U = stop.shape
    s_idx, u_idx = np.meshgrid(np.arange(S), np.arange(U), indexing="ij")
    P = np.asarray(decode)[stop, s_idx, u_idx].astype(np.int64)  # [S, U, L]
    pick = np.asarray(energy_raw)[stop, s_idx, u_idx].argmin(0)
    sel = P[pick, np.arange(U)][None]
    seg = float((stop + 1).mean())
    if task == "gc":
        g = score_graph_coloring(inputs, P, nodes)
        return {
            "n1": g["conflicts_n1"],
            "n20": score_graph_coloring(inputs, sel, nodes)["conflicts_n1"],
            "cov20": 100.0 * g["cov20"],
            "segments": seg,
        }
    g = score_nqueens(inputs, P, rows_per_input)
    return {
        "n1": 100.0 * g["acc_n1"],
        "n20": 100.0 * score_nqueens(inputs, sel, rows_per_input)["acc_n1"],
        "cov20": 100.0 * g["cov20"],
        "segments": seg,
    }


# --------------------------------------------------------------------------------------------- sampling runner
def load_inputs(task: str, data: Path):
    """Unique test inputs (first row of each puzzle range) plus what scoring needs (no labels: answers are checked
    against the puzzle rules)."""
    test = data / "test"
    inputs_all = np.load(test / "all__inputs.npy")
    pi = np.load(test / "all__puzzle_indices.npy")
    inputs = np.ascontiguousarray(inputs_all[pi[:-1].astype(int)]).astype(np.int32)
    counts = np.load(test / "all__solution_counts.npy") if (test / "all__solution_counts.npy").exists() else None
    if task == "gc":
        for i, (a, b) in enumerate(zip(pi[:-1], pi[1:])):
            if not (inputs_all[a:b] == inputs[i]).all():
                raise ValueError("a graph row range contains different inputs")
        return inputs, {"solution_counts": counts}
    return inputs, {"puzzle_indices": pi, "solution_counts": counts}


def run(
    checkpoint: str,
    data_path: str,
    out: str,
    *,
    task: str,
    samples: int = 20,
    batch: int | None = None,
    depth: int | None = None,
    seed: int | None = None,
    tau: float = GTAN_TAU,
    device: str = "cuda",
) -> dict[str, Any]:
    """Draw ``samples`` trajectories per test input (``task`` gc or nq), record every segment end, and score them.

    Inputs go through in batches of ``batch``, sample by sample. Every batch after the first starts from the RNG state
    the previous one had at the trained depth, so the draws do not depend on ``depth``. ``seed``: the training seed if
    None.
    """
    if task not in ("gc", "nq"):
        raise ValueError(f"task must be gc (Graph Coloring) or nq (N-Queens), got {task!r}")
    dev = torch.device(device)
    data = Path(data_path)
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    train_seed = int(ckpt["config"]["seed"])
    seed = train_seed if seed is None else int(seed)
    train_depth = int(ckpt["config"]["arch"]["halt_max_steps"])
    depth = max(int(depth or train_depth), train_depth)
    if depth > train_depth and ckpt["config"]["arch"].get("noise_cosine"):
        raise ValueError("a deeper eval would change the noise schedule of this checkpoint")
    batch = int(batch or (1280 if task == "gc" else 1024))

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = eval_config(ckpt, device=dev, halt_max_steps=depth)
    metadata = check_eval_data(config, str(data))
    data_task = "gc" if getattr(metadata, "nodes", None) is not None else "nq"
    if task != data_task:
        raise ValueError(f"task {task} does not match {data_path}, which holds {data_task} data")
    inputs, extra = load_inputs(task, data)
    model, ema_tensors = load_model(ckpt, config, metadata, device=dev, build_device=torch.device("cpu"))
    step = int(ckpt.get("step", -1))
    del ckpt

    num_inputs, samples = len(inputs), int(samples)
    seq_len = inputs.shape[1]
    nodes = int(getattr(metadata, "nodes", None) or round(seq_len**0.5))  # graph nodes, or the N of N-Queens
    answer_len = nodes if task == "gc" else seq_len  # a coloring is the first ``nodes`` cells
    all_inputs = torch.as_tensor(inputs, dtype=torch.int32)
    recorded = {
        k: np.zeros((depth, samples, num_inputs), np.float32) for k in ("energy_raw", "energy", "gtan", "gnorm")
    }
    decode = np.zeros((depth, samples, num_inputs, answer_len), np.int8)

    t0 = time.time()
    for s in range(samples):
        for start in range(0, num_inputs, batch):
            rows = _batch_rows(task, start, batch, num_inputs)
            keep = min(batch, num_inputs - start)
            trajectory, rng_at_train_depth = _rollout_batch(
                model, all_inputs[rows].to(dev), depth=depth, train_depth=train_depth, answer_len=answer_len
            )
            out_rows = slice(start, start + keep)
            decode[:, s, out_rows] = trajectory.pop("decode")[:, :keep]
            for k, v in trajectory.items():
                recorded[k][:, s, out_rows] = v[:, :keep]
            _set_rng_state(rng_at_train_depth)
        print(f"[{time.time() - t0:.0f}s] sample {s + 1}/{samples}", flush=True)

    cells = _score(task, decode, recorded, inputs, extra, nodes, tau, train_depth)
    meta = {
        "checkpoint": checkpoint,
        "step": step,
        "task": task,
        "seed": seed,
        "train_seed": train_seed,
        "train_halt": train_depth,
        "depth": depth,
        "samples": samples,
        "batch": batch,
        "ema_tensors": ema_tensors,
        "tau": tau,
        "seconds": round(time.time() - t0, 1),
        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
    }
    arrays = dict(
        decode=decode,
        inputs=inputs,
        nodes=np.int32(nodes),
        train_halt=np.int32(train_depth),
        task=np.array(task),
        **recorded,
        **{k: v for k, v in extra.items() if v is not None},
    )
    return write_result(out, arrays, {"meta": meta, "cells": cells})


def _score(
    task: str,
    decode: np.ndarray,
    recorded: dict[str, np.ndarray],
    inputs: np.ndarray,
    extra: dict[str, Any],
    nodes: int,
    tau: float,
    train_depth: int,
) -> dict[str, dict[str, Any]]:
    """The scores for two stopping rules: stop each sample at |grad_M E| < tau, or at the trained depth."""
    _, samples, num_inputs = recorded["gtan"].shape
    score_kwargs = {} if task == "gc" else {"rows_per_input": np.diff(extra["puzzle_indices"])}
    stops = {
        f"gtan<{tau}": gtan_stop_index(recorded["gtan"], tau, train_depth),
        f"fixed D{train_depth}": np.full((samples, num_inputs), train_depth - 1),
    }
    return {
        name: gram_cells(task, decode, recorded["energy_raw"], stop, inputs, nodes, **score_kwargs)
        for name, stop in stops.items()
    }


def _rollout_batch(
    model: torch.nn.Module, x: torch.Tensor, *, depth: int, train_depth: int, answer_len: int
) -> tuple[dict[str, np.ndarray], Any]:
    """One sample of ``depth`` segments on the inputs ``x``.

    Returns the decoded answer and the energy measurements after every segment, each (depth, rows, ...), and the RNG
    state at the trained depth (where the next batch starts).
    """
    inner = model.model.inner
    data = {
        "inputs": x,
        "labels": torch.full_like(x, -100),  # no labels: answers are checked by rules
        "puzzle_identifiers": torch.zeros(len(x), dtype=torch.int32, device=x.device),
    }
    per_segment = {k: [] for k in ("decode", "energy", "energy_raw", "gtan", "gnorm")}
    rng_at_train_depth = None
    with torch.no_grad():
        carry = model.initial_carry(data)
        x_embedded = inner.input_embeddings(data["inputs"])
        cos_sin = inner.cos_sin()
        for segment in range(depth):
            carry, _, _, outputs, finished = model(carry=carry, batch=data, return_keys=["preds", "logits"])
            if segment + 1 == train_depth:
                rng_at_train_depth = _rng_state(x.device)
            energy, gnorm, gtan, energy_raw = measure(
                inner.level, carry.inner_carry.z, x_embedded, cos_sin, raw_energy=True
            )
            per_segment["decode"].append(outputs["logits"].argmax(-1)[:, :answer_len].to(torch.int8).cpu().numpy())
            per_segment["energy"].append(energy.cpu().numpy())
            per_segment["energy_raw"].append(energy_raw.cpu().numpy())
            per_segment["gtan"].append(gtan.cpu().numpy())
            per_segment["gnorm"].append(gnorm.cpu().numpy())
        if not bool(torch.as_tensor(finished).all().item()):
            raise RuntimeError("eval did not finish at the requested depth")
    return {k: np.stack(v) for k, v in per_segment.items()}, rng_at_train_depth


def _batch_rows(task: str, start: int, batch: int, num_inputs: int) -> torch.Tensor:
    """The input rows of the batch starting at ``start``, padded to ``batch`` rows.

    Graph Coloring wraps around to the first inputs. N-Queens pads with the next inputs, stopping at the end of the
    data, so its last batch can be shorter. The batch shape sets the noise draws, so this padding is part of the
    protocol.
    """
    if task == "gc":
        return (torch.arange(batch) + start) % num_inputs
    keep = min(batch, num_inputs - start)
    rows = torch.arange(start, start + keep)
    if batch > keep:
        rows = torch.cat([rows, torch.arange(start, min(start + (batch - keep), num_inputs))])
    return rows


def _rng_state(dev: torch.device):
    return torch.get_rng_state(), (torch.cuda.get_rng_state_all() if dev.type == "cuda" else None)


def _set_rng_state(state) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])
