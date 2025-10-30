#!/bin/bash -l
# Update conda path to yours
source /home/abdelrahman/anaconda3/etc/profile.d/conda.sh

# Create and activate environment if it doesn't exist
conda create -n py3lyap python=3.8 -y
conda activate py3lyap

# Install requirements
pip install -r requirements.txt
pip install gurobipy  # Missing dependency

python setup.py
source ./config/setup_environments.sh

# Start timer
SECONDS=0

python neural_network_lyapunov/examples/path_following_unicycle/monotonic_train_path_following_demo.py \
    --bound_level=40 \
    --use_fpl \
    --pretrain_num_epochs=80 \
    --batch_size=512 \
    --learning_rate=1e-2 \
    --search_R \
    # --max_iterations=500 \

# Stop timer and report
duration=$SECONDS
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"


conda deactivate