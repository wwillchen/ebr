#!/usr/bin/env bash
# Evaluate a checkpoint with a benchmark's protocol: bash scripts/eval.sh <task> <checkpoint> <out_dir> [--key value ...]
#   task: sudoku | maze_hard | nqueens_8 | nqueens_10 | graph_coloring_8 | graph_coloring_10 (config/eval/<task>.yaml).
#   Sudoku runs its shards one after another and merges them (NUM_SHARDS, default 16).
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:?usage: bash scripts/eval.sh <task> <checkpoint> <out_dir> [--key value ...]}
CKPT=${2:?checkpoint}
OUT=${3:?out dir}
shift 3
CFG="config/eval/${TASK}.yaml"
if [[ ! -f "${CFG}" ]]; then
  echo "unknown task '${TASK}': expected one of $(ls config/eval | sed 's/\.yaml$//' | tr '\n' ' ')" >&2
  exit 2
fi
if [[ "${TASK}" == sudoku ]]; then
  S=${NUM_SHARDS:-16}
  for ((i = 0; i < S; i++)); do
    python evaluate.py --config "${CFG}" --checkpoint "${CKPT}" --shard "${i}" --num_shards "${S}" --out "${OUT}" "$@"
  done
  python evaluate.py --merge "${OUT}"
else
  python evaluate.py --config "${CFG}" --checkpoint "${CKPT}" --out "${OUT}" "$@"
fi
