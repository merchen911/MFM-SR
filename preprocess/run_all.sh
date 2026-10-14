#!/usr/bin/env bash
# Run preprocessing stages 1-5 in order. Each stage skips outputs that already exist.
#   bash preprocess/run_all.sh            # build missing outputs
#   bash preprocess/run_all.sh --force    # rebuild everything
# Environment variables:
#   PYTHON    python executable (default: python)
#   DEVICE    device for feature extraction, e.g. cpu or cuda:0 (default: cpu)
#   DATASETS  space-separated dataset names (default: all four)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cpu}"
read -r -a DS <<< "${DATASETS:-Video_Games Boys_Girls Women Men}"
FORCE=()
if [[ "${1:-}" == "--force" ]]; then FORCE=(--force); fi

"$PYTHON" preprocess/build_interactions.py --datasets "${DS[@]}" ${FORCE[@]+"${FORCE[@]}"}
"$PYTHON" preprocess/clean_categories.py --datasets "${DS[@]}" ${FORCE[@]+"${FORCE[@]}"}
"$PYTHON" preprocess/extract_text.py --datasets "${DS[@]}" --device "$DEVICE" ${FORCE[@]+"${FORCE[@]}"}
"$PYTHON" preprocess/extract_image.py --datasets "${DS[@]}" --device "$DEVICE" ${FORCE[@]+"${FORCE[@]}"}
"$PYTHON" preprocess/write_atomic.py --datasets "${DS[@]}" ${FORCE[@]+"${FORCE[@]}"}
