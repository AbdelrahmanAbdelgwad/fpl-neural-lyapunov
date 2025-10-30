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

for i in {1..10}; do
    if [ "$i" -eq 1 ]; then
        python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py \
            --bound_level=$i \
            --bound_level_last=$(($i-1)) \
            # --search_R \

            # --max_iterations=1000
    else
        python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py \
            --bound_level=$i \
            --bound_level_last=$(($i-1)) \
            # --search_R \

            # --max_iterations=1000
    fi
done

duration=$SECONDS
echo "Total execution time: $((duration / 3600))h $(((duration / 60) % 60))m $((duration % 60))s"

conda deactivate