#!/bin/bash -l
source /home/zw2445/anaconda3/etc/profile.d/conda.sh
conda activate py3lyap
python setup.py
source ./config/setup_environments.sh
cd /home/zw2445//Documents/neural-network-lyapunov
python neural_network_lyapunov/examples/cart_pole/monotonic_train_cart_pole_demo.py --bound_level=1 --bound_level_last=0
# do 
#     python neural_network_lyapunov/examples/cart_pole/monotonic_train_cart_pole_demo.py --bound_level=$i --bound_level_last=$(($i-1)); 
# done
conda deactivate