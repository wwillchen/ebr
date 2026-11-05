#!/usr/bin/env bash
# Build the ARC datasets (after scripts/download_arc.sh). ARC training is not released yet.
set -euo pipefail
cd "$(dirname "$0")/.."
PREFIX="${ARC_PREFIX:-kaggle/combined/arc-agi}"

python -m dataset.build_arc_dataset \
  --input-file-prefix "$PREFIX" \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

python -m dataset.build_arc_dataset \
  --input-file-prefix "$PREFIX" \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
