#!/usr/bin/env bash
# Baseline (no FPL) pendulum training: Stage II only (MILP certify-repair).
# Reproduces the "Without FPL" pendulum row of Table I in the paper.
#
# Prerequisites (see README.md):
#   - Python env with requirements.txt installed and gurobipy available
#   - Activate the env yourself before running this script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SECONDS=0

for i in {1..10}; do
    python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py \
        --bound_level=${i} \
        --bound_level_last=$((i-1))
done

duration=${SECONDS}
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"
