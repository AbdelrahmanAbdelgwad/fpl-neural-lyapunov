#!/bin/bash -l
source /home/abdelrahman/anaconda3/etc/profile.d/conda.sh


# Navigate to project root
cd /home/abdelrahman/projects/Neural_Lyapunov_Control/neural-network-lyap-control-roa
source config/setup_environments.sh

# Activate conda environment (create if needed)
conda activate py3lyap 2>/dev/null || {
    conda create -n py3lyap python=3.8 -y
    conda activate py3lyap
}

# Install requirements
pip install -r requirements.txt
pip install gurobipy

# Add project to PYTHONPATH instead of installing
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run training
SECONDS=0

for i in {1..40}; do
    if [ "$i" -eq 1 ]; then
        python neural_network_lyapunov/examples/cart_pole/monotonic_train_cart_pole_demo.py \
            --bound_level=$i \
            --bound_level_last=$(($i-1)) \
            --load_lyapunov_relu neural_network_lyapunov/examples/cart_pole/data/preprocess/lqr_lyapunov_monotonic.pt \
            --load_controller_relu neural_network_lyapunov/examples/cart_pole/data/preprocess/lqr_controller.pt \
            --load_lyapunov_R neural_network_lyapunov/examples/cart_pole/data/preprocess/lqr_R_monotonic.pt \


    else
    
        python neural_network_lyapunov/examples/cart_pole/monotonic_train_cart_pole_demo.py \
            --bound_level=$i \
            --bound_level_last=$(($i-1)) \

    fi
done

duration=$SECONDS
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"

conda deactivate