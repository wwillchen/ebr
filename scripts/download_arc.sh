#!/usr/bin/env bash
# Fetch the raw ARC JSON (pinned, sha256-checked): bash scripts/download_arc.sh [--upstream] [DEST=kaggle/combined]
set -euo pipefail

UPSTREAM=0
if [[ "${1:-}" == "--upstream" ]]; then UPSTREAM=1; shift; fi
DEST="${1:-kaggle/combined}"
GRAM_COMMIT=f2e32b9b11737890bfd4f39ead4cbea90835f91b
BASE="https://raw.githubusercontent.com/ahn-ml/gram-release/${GRAM_COMMIT}/kaggle/combined"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }

mkdir -p "$DEST"
while read -r digest name; do
  [[ -z "$name" ]] && continue
  out="$DEST/$name"
  if [[ -f "$out" && "$(sha256 "$out")" == "$digest" ]]; then echo "ok        $name"; continue; fi
  curl -fsSL "$BASE/$name" -o "$out.part"
  got="$(sha256 "$out.part")"
  if [[ "$got" != "$digest" ]]; then echo "sha256 mismatch for $name: $got" >&2; rm -f "$out.part"; exit 1; fi
  mv "$out.part" "$out"; echo "fetched   $name"
done <<'EOF'
4e60bd1de956c2519c3236ba9f2c352c971d5971bfc568ceafa716dd9b1a91fd arc-agi_concept_challenges.json
3ec5e95e416e49c85dac28475fc5b73f6998d13b7e91ff0219eee48881c327ee arc-agi_concept_solutions.json
2b05f8f7d33b73fdbefd69914c399e9409d1949c7f16aabc78e677443a64b208 arc-agi_evaluation_challenges.json
629622b0eecf42c6cb4da628955ef76ec43214f0d00081f8cf63082ade496c52 arc-agi_evaluation_solutions.json
e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8 arc-agi_evaluation2_challenges.json
84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1 arc-agi_evaluation2_solutions.json
5b02a4753e9088134744d952b048a03f6f23abf4eb6c04e3f27822cf3d439634 arc-agi_training_challenges.json
47212ad04ef4211221ac6141cba8c8d425e20e90da90ace575ef286c5f992534 arc-agi_training_solutions.json
779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5 arc-agi_training2_challenges.json
9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9 arc-agi_training2_solutions.json
EOF

if [[ "$UPSTREAM" == 1 ]]; then
  RAW="$(dirname "$DEST")/upstream"
  mkdir -p "$RAW"
  while read -r repo commit; do
    dir="$RAW/$(basename "$repo")"
    [[ -d "$dir/.git" ]] || git clone -q "https://github.com/$repo" "$dir"
    git -C "$dir" fetch -q origin "$commit" 2>/dev/null || true
    git -C "$dir" checkout -q "$commit"
    [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || { echo "could not pin $repo at $commit" >&2; exit 1; }
    echo "upstream  $repo @ $commit -> $dir"
  done <<'EOF'
arcprize/ARC-AGI f12822c4d550121c35a275008d964afbbed47d2f
arcprize/ARC-AGI-2 f3283f727488ad98fe575ea6a5ac981e4a188e49
victorvikram/ConceptARC 0e67da6af879e4bad3d7cd3c196e8d551b445725
EOF
fi
echo "ARC raw JSON ready in $DEST"
