#!/usr/bin/env bash
# Baseline (no FPL) unicycle training at bound_level=40: Stage II only.
# Reproduces the "Without FPL" unicycle row of Table I in the paper.
#
# Prerequisites (see README.md):
#   - Python env with requirements.txt installed and gurobipy available
#   - Activate the env yourself before running this script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SECONDS=0

python neural_network_lyapunov/examples/path_following_unicycle/monotonic_train_path_following_demo.py \
    --bound_level=40 \
    --search_R

duration=${SECONDS}
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"
