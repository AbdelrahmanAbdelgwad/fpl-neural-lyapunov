#!/bin/bash -l
source /home/zw2445/anaconda3/etc/profile.d/conda.sh
conda activate py3lyap
python setup.py
source ./config/setup_environments.sh
cd /home/zw2445/Documents/neural-network-lyapunov

for i in {1..40}; 
do 
    python neural_network_lyapunov/examples/path_following_unicycle/monotonic_train_path_following_demo.py --bound_level=$i --bound_level_last=$(($i-1)); 
done
# python neural_network_lyapunov/examples/path_following_unicycle/monotonic_linf_train_path_following_demo.py --bound_level=12 --bound_level_last=11
conda deactivate