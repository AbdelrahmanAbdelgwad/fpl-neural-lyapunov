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

# CKPT_DIR="neural_network_lyapunov/examples/cart_pole/data/monotonic_bound40_fpl"

for i in {1..100}; do
    if [ "$i" -eq 1 ]; then
        python neural_network_lyapunov/examples/cart_pole/monotonic_train_cart_pole_demo.py \
            --use_fpl \
            --bound_level=$i \
            --pretrain_num_epochs=5000 \
            --max_iterations=1 \
            --load_lyapunov_relu neural_network_lyapunov/examples/cart_pole/data/preprocess/lqr_lyapunov_monotonic.pt \
            --load_controller_relu neural_network_lyapunov/examples/cart_pole/data/preprocess/lqr_controller.pt \
            --load_lyapunov_R neural_network_lyapunov/examples/cart_pole/data/preprocess/lqr_R_monotonic.pt \


    else
    
        python neural_network_lyapunov/examples/cart_pole/monotonic_train_cart_pole_demo.py \
            --use_fpl \
            --bound_level=$i \
            --bound_level_last=$(($i-1)) \
            --max_iterations=1 \
  
    fi
done


# Stop timer and report
duration=$SECONDS
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"


conda deactivate