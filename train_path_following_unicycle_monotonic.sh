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

for i in {1..40}; 
do 
    python neural_network_lyapunov/examples/path_following_unicycle/monotonic_train_path_following_demo.py --bound_level=$i --bound_level_last=$(($i-1)); 
done

conda deactivate