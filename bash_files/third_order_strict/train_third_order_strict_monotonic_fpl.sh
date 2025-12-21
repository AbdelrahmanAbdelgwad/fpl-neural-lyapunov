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

# Stay in current directory instead of wrong path
# cd /home/zw2445/Documents/neural-network-lyapunov  # Remove this line

# Start timer
SECONDS=0

for i in {1..50}; do
    python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py \
        --search_R \
        --use_fpl \
        --pretrain_num_epochs=80 \
        --bound_level=$i \
        --bound_level_last=$(($i-1)) \
        --max_iterations=500 \
        
done

# Stop timer and report
duration=$SECONDS
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"


conda deactivate