#!/bin/bash -l
source /home/zw2445/anaconda3/etc/profile.d/conda.sh
conda activate py3lyap
python setup.py
source ./config/setup_environments.sh
cd /home/zw2445/Documents/neural-network-lyapunov
# python neural_network_lyapunov/examples/quadrotor2d/monotonic_train_quadrotor_2d_demo.py --bound_level=1
# for i in {2,3,4,5,6,7,8}; 
# do 
#     python neural_network_lyapunov/examples/quadrotor2d/monotonic_train_quadrotor_2d_demo.py --bound_level=$i --bound_level_last=$(($i-1)); 
# done
# python neural_network_lyapunov/examples/quadrotor2d/train_quadrotor_2d_demo.py --bound_level=1
for i in {1..40}; 
do 
    python neural_network_lyapunov/examples/path_following_unicycle/train_path_following_demo.py --bound_level=$i --bound_level_last=$(($i-1)); 
done
# python neural_network_lyapunov/examples/quadrotor2d/train_quadrotor_2d_demo.py --bound_level=10 --bound_level_last=8
conda deactivate