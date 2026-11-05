"""Evaluate a checkpoint with a benchmark protocol (one config per benchmark in config/eval/).

Usage:
    python evaluate.py --config config/eval/<task>.yaml --checkpoint <step_N.pth> --out <dir> [--key value ...]
    python evaluate.py --merge <dir>      # Sudoku: merge the shard results

Any key of the protocol (see PROTOCOL_KEYS) can be overridden as ``--key value``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.set_float32_matmul_precision("high")

PROTOCOL_KEYS = {
    "sudoku": ("breadth", "depth", "batch", "shard", "num_shards", "max_boards", "device"),
    "maze": ("n", "depth", "batch", "grad_chunk", "draw_seed", "flat_noise", "device", "max_batches"),
    "gram": ("task", "samples", "batch", "depth", "seed", "tau", "device"),
}


def _parse_flags(argv: list[str]) -> dict[str, Any]:
    """``--key value`` or ``--key=value`` pairs; values are read as YAML (numbers, booleans, null)."""
    flags: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        if "=" in argv[i]:
            key, value = argv[i].lstrip("-").split("=", 1)
            i += 1
        else:
            key, value = argv[i].lstrip("-"), argv[i + 1]
            i += 2
        flags[key] = yaml.safe_load(value)
    return flags


def run_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one protocol from a merged config dict (config file + flags)."""
    protocol = cfg.pop("protocol")
    checkpoint, data_path = cfg.pop("checkpoint"), cfg.pop("data_path")
    out_dir = Path(cfg.pop("out"))
    if protocol == "sudoku":
        out = str(out_dir / f"shard_{int(cfg.get('shard', 0))}_of_{int(cfg.get('num_shards', 1))}")
    else:
        out = str(out_dir / (cfg.get("task") or protocol))
    if not torch.cuda.is_available():
        cfg.setdefault("device", "cpu")
    unknown = set(cfg) - set(PROTOCOL_KEYS[protocol])
    if unknown:
        raise ValueError(f"unknown keys for protocol {protocol}: {sorted(unknown)}")
    if protocol == "sudoku":
        from evaluators import sudoku as runner
    elif protocol == "maze":
        from evaluators import maze as runner
    else:
        from evaluators import gram as runner
    return runner.run(checkpoint, data_path, out, **cfg)


def merge_sudoku(directory: str) -> dict[str, Any]:
    """Merge the shard results in ``directory`` into ``merged.json`` (percentages per depth)."""
    from evaluators import protocol

    parts = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(directory).glob("*.json"))
        if p.name != "merged.json"
    ]
    shards = sorted(p["meta"]["shard"] for p in parts)
    if shards != list(range(parts[0]["meta"]["num_shards"])):
        raise ValueError(f"missing shards: have {shards}")
    cells = protocol.merge_counts([p["counts"] for p in parts])
    (Path(directory) / "merged.json").write_text(json.dumps(cells, indent=1))
    return cells


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    flags = _parse_flags(argv)
    if "merge" in flags:
        print(json.dumps(merge_sudoku(flags["merge"]), indent=1))
        return
    cfg = dict(OmegaConf.to_container(OmegaConf.load(flags.pop("config")), resolve=True))
    cfg.update(flags)
    result = run_protocol(cfg)
    print(json.dumps(result.get("cells", result.get("counts")), indent=1))


if __name__ == "__main__":
    main()
