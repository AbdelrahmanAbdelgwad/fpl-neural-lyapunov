#!/usr/bin/env bash
# FPL-accelerated unicycle training: Stage I (FPL pre-training) + Stage II (MILP).
# Reproduces the "With FPL" unicycle row of Table I in the paper.
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
    --use_fpl \
    --pretrain_num_epochs=80 \
    --batch_size=512 \
    --learning_rate=1e-2 \
    --search_R

duration=${SECONDS}
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"
