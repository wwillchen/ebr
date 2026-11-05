#!/usr/bin/env bash
# Train one benchmark: bash scripts/train.sh <task> [key=value ...]
#   task: sudoku | sudoku_paper | maze_hard | maze_hard_fast | nqueens_8 | nqueens_10 | graph_coloring_8 | graph_coloring_10
#   (config/train/<task>.yaml). Any config key can be overridden, e.g. seed=1 dataset.data_path=/data/sudoku.
#   NPROC=<n> runs torchrun on n GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:?usage: bash scripts/train.sh <task> [key=value ...]}
shift
if [[ ! -f "config/train/${TASK}.yaml" ]]; then
  echo "unknown task '${TASK}': expected one of $(ls config/train | sed 's/\.yaml$//' | tr '\n' ' ')" >&2
  exit 2
fi
if [[ "${TASK}" == sudoku || "${TASK}" == *_fast ]]; then
  export EBR_FUSED_ATTN_IMPL=${EBR_FUSED_ATTN_IMPL:-triton}  # fail rather than fall back to the slow reference
fi
NPROC=${NPROC:-1}
if [[ "${NPROC}" -gt 1 ]]; then
  launch=(torchrun --nproc-per-node "${NPROC}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1)
else
  launch=(python)
fi
exec "${launch[@]}" pretrain.py "train/${TASK}" "$@"
