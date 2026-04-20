#!/usr/bin/env bash
# FPL-accelerated pendulum training: Stage I (FPL pre-training) + Stage II (MILP).
# Reproduces the "With FPL" pendulum row of Table I in the paper.
#
# Prerequisites (see README.md):
#   - Python env with requirements.txt installed and gurobipy available
#   - Activate the env yourself before running this script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

SECONDS=0

python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py \
    --use_fpl \
    --bound_level=10 \
    --search_R \
    --pretrain_num_epochs=200 \
    --batch_size=128 \
    --learning_rate=1e-2

duration=${SECONDS}
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"
