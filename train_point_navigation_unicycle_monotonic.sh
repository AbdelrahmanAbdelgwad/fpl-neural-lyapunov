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

for i in {1..40}; do
    if [ "$i" -eq 1 ]; then
        python neural_network_lyapunov/examples/point_navigation/monotonic_train_point_navigation_demo.py \
            --bound_level=$i \
            --bound_level_last=$(($i-1)) \
            # --train_forward_model \
            # --generate_dynamics_data \

    else
        python neural_network_lyapunov/examples/point_navigation/monotonic_train_point_navigation_demo.py \
            --bound_level=$i \
            --bound_level_last=$(($i-1))
    fi
done

# Stop timer and report
duration=$SECONDS
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"


conda deactivate