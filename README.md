# Introduction
This repo contains the code for the project as introduced in
   - [Lyapunov Neural Network with Region of Attraction Search](https://arxiv.org/pdf/2403.10621) <br>
       Zili Wang, Sean B. Andersson, and Roberto Tron <br>
       IEEE American Control Conference 2024 

# Setup

Please follow `REF_README.md` to set up the environment.

## Run experiments
TODO: add instructions to run (1) system model training (`preprocess` folder). (2) controller initialization (`preprocess` folder) (3) multi-polygon training (4) result visualization (`visualization` folder) 
You could run
```
$ bash train_path_following_unicycle_monotonic.sh
```
This will synthesize a stabilizing controller together with a Lyapunov certificate for the unit-circle path-following unicycle model.

You could also run
```
$ bash train_path_following_unicycle.sh
```
to generate the baseline results.


# Setup

## Python requirements
We use python 3 in this project. You could first install the packages in requirements.txt.

## Install gurobi
Please download gurobi from https://www.gurobi.com/products/gurobi-optimizer/. We require at least gurobi 9.5. After downloading the software, please install its Python API by following https://www.gurobi.com/documentation/9.0/quickstart_mac/the_grb_python_interface_f.html

To check your gurobi installation, type the following command in your terminal:
```
$ python3 -c "import gurobipy"
```
There should be no error thrown when executing the command.

## Setup environment variable
In the terminal, please run
```
$ python3 setup.py
```
It will print out the command to setup the environment variables. Execute that command in your terminal.

## Run a toy example
You could run
```
$ python3 neural_network_lyapunov/test/train_toy_system_controller_demo.py --dimension=1
```
This will synthesize a stabilizing controller with a Lyapunov function for a toy 1D system (TODO: add some visualization at the end of the demo). You should see that the error printed on the screen decreases to almost 0. (The code is non-deterministic, so if it doesn't converge to 0 in the first trial, you can re-run the demo and hopefully it converges in the second trial).

# Contributing to repo
## Linting
We use `flake8` to check if the python code follows PEP standard. Before submitting the PR, you could run
```
$ cd neural_network_lyapunov
$ flake8 ./
```
to check if there are any violations.

## Unit test
I am a strong believer of unit test. We strongly encourage to add tests to the functions in the PR.

