#!/usr/bin/env bash
# Run experiments and build their reports. Output goes to $RESULTS (default
# runs/), so the reference data in results/ is left untouched. Finished cases
# are skipped, so re-running the same command resumes after an interruption.
#   scripts/reproduce.sh /path/to/Qwen3-8B [experiments/eN_*.toml ...]
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:?usage: $0 MODEL_DIR [EXPERIMENT.toml ...]}
shift
out=${RESULTS:-runs}
(( $# )) || set -- experiments/e[0-9]*.toml
for exp in "$@"; do
    echo "=== $exp ($(date '+%F %T'))"
    python -m bench.run "$exp" --model "$model" --results "$out"
    python -m bench.report "$out/$(basename "$exp" .toml)"
done
