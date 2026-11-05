#!/usr/bin/env bash
# Download datasets and pretrained checkpoints: bash scripts/download.sh [all|datasets|checkpoints]
set -euo pipefail
cd "$(dirname "$0")/.."

WHAT=${1:-all}
REPO=wxingxu/ebr
REVISION=da0172a51409281079094bb5f6d19b5bdc1c44bb
CKPT_DIR=${CKPT_DIR:-checkpoints}
PYTHON=${PYTHON:-python}

case "$WHAT" in
  all | datasets | checkpoints) ;;
  *) echo "usage: bash scripts/download.sh [all|datasets|checkpoints]" >&2; exit 2 ;;
esac

if [[ "$WHAT" != checkpoints ]]; then
  bash scripts/build_all.sh
fi

if [[ "$WHAT" != datasets ]]; then
  $PYTHON - "$REPO" "$REVISION" "$CKPT_DIR" <<'EOF'
import hashlib
import sys

from huggingface_hub import hf_hub_download

repo, revision, out = sys.argv[1:4]
FILES = {
    "sudoku-extreme/sudoku.pth": "541e57489d8239e36f4124ab63e6ce1ca73322d352e64e8b507f35a83016bd7a",
    "maze-hard/maze.pth": "812d1092870f30a7becc169a3526ec43b0c48a532107f69cd3cca2c8feb87cde",
    "n-queens/nq8.pth": "0b37f11d54a6a9ec9a59a28bd0fcdfb25202df8a9887be5a71fb1e4317494319",
    "n-queens/nq10.pth": "94c307a2ae9feac59d57c113b36df591fa27efb86b757f9fe04c079be5b62497",
    "graph-coloring/gc8.pth": "0014171682b385586aa8c47e9ccf16d3eabba1c12d76b54b4875fadc0bfa0ece",
    "graph-coloring/gc10.pth": "700ecf9f8ba974050e01716642c746a21b3c62f5c1790f9520d99f7a87de7d6f",
}
for name, digest in FILES.items():
    path = hf_hub_download(repo, name, revision=revision, local_dir=out)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != digest:
        sys.exit(f"sha256 mismatch for {name}: {h.hexdigest()}")
    print(f"ok  {path}")
EOF
fi
