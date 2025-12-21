#!/bin/bash -l
source /home/zw2445/anaconda3/etc/profile.d/conda.sh
conda activate py3lyap
python setup.py
source ./config/setup_environments.sh
cd /home/zw2445//Documents/neural-network-lyapunov
# for i in {1..50}; 
# do 
#     python neural_network_lyapunov/examples/third_order_strict/monotonic_train_third_order_demo.py --bound_level=$i --bound_level_last=$(($i-1)); 
# done
python neural_network_lyapunov/examples/third_order_strict/monotonic_linf_train_third_order_demo.py --bound_level=1 --bound_level_last=0; 
conda deactivate