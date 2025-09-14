#!/bin/sh
#!/bin/bash -l

#$ -P mapermon         # Specify the SCC project name you want to use
#$ -j y               # Merge the error and output streams into a single file
#$ -l gpus=1
#$ -l gpu_c=5.0       # Specify minimum GPU capability. Current choices for CAPABILITY are 3.5, 5.0, 6.0, 7.0, and 8.6
#$ -pe omp 28         # Request multiple slots for Shared Memory applications (OpenMP, pthread).

source /share/pkg.7/miniconda/4.9.2/install/etc/profile.d/conda.sh
# source ~/opt/anaconda3/etc/profile.d/conda.sh
conda activate py3lyap
python setup.py
source ./config/setup_environments.sh
# cd ~/Desktop/Research/Lyapunov/neural-network-lyapunov
cd /home/abdelrahman/projects/Neural_Lyapunov_Control/neural-network-lyap-control-roa

python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py --bound_level=2
for i in {4,6,8,10}; 
do 
    python neural_network_lyapunov/examples/pendulum/monotonic_train_pendulum_demo.py --bound_level=$i --bound_level_last=$(($i-2)); 
done
# python neural_network_lyapunov/examples/pendulum/train_pendulum_demo.py --bound_level=10 --bound_level_last=8
conda deactivate

python neural_network_lyapunov/examples/pendulum/monotonic_linf_train_pendulum_demo.py --bound_level=1 --bound_level_last=0