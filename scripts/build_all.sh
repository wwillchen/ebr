#!/usr/bin/env bash
# Build every dataset into data/ (WITH_ARC=1 also builds ARC).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m dataset.build_sudoku_dataset --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
python -m dataset.build_maze_hard_dataset --output-dir data/maze-30x30-hard-1k
python -m dataset.build_nqueens_dataset --n 8 --output-dir data/nqueens-8
python -m dataset.build_nqueens_dataset --n 10 --output-dir data/nqueens-10
python -m dataset.build_graph_coloring_dataset --nodes 8 --output-dir data/graph-coloring-8
python -m dataset.build_graph_coloring_dataset --nodes 10 --output-dir data/graph-coloring-10

if [[ "${WITH_ARC:-0}" == 1 ]]; then bash scripts/build_arc.sh; fi
