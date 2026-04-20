# Accelerating Lyapunov-Stable Neural Control using Fulfillment Priority Logic

This repository contains the code accompanying the ACC 2026 paper:

> **Accelerating Lyapunov-Stable Neural Control using Fulfillment Priority Logic**
> Abdelrahman Abdelgawad, Bassel El Mabsout, Zili Wang, Renato Mancuso, Sean B. Andersson, Roberto Tron
> *American Control Conference (ACC), 2026*

The method combines monotonic neural Lyapunov functions
([Wang, Andersson, Tron, ACC 2024](https://doi.org/10.23919/ACC60939.2024.10644877))
with Fulfillment Priority Logic
([Mabsout, Abdelgawad, Mancuso, IROS 2025](https://doi.org/10.1109/IROS60139.2025.11247623))
to dramatically reduce training time while preserving formal Lyapunov stability
guarantees obtained via MILP verification.

The pipeline has two stages:

1. **Stage I — FPL pre-training.** A sampling-based FPL objective jointly
   initializes the controller and Lyapunov networks using gradient descent on
   short rollouts. No MILP calls in this stage.
2. **Stage II — MILP certify-repair.** Exactly the procedure of the monotonic
   Lyapunov baseline: iteratively solve a MILP for the worst-case
   Lyapunov-decrease violation and take a gradient step on the active-set
   surrogate until the system is certified.

Prepending Stage I reduces wall-clock training time by ~95% on the pendulum
and ~94% on the path-following unicycle benchmarks (see paper, Table I).

> **Note.** This repository was forked from an earlier codebase. Irrelevant
> examples have been removed; what remains is what's needed to reproduce the
> ACC 2026 paper.

## Repository layout

```
.
├── bash_files/                          # Paper reproduction wrappers
│   ├── pendulum/
│   │   ├── train_pendulum_monotonic.sh         # Baseline (no FPL)
│   │   └── train_pendulum_monotonic_fpl.sh     # FPL-accelerated
│   └── path_following/
│       ├── train_path_following_unicycle_monotonic_40.sh   # Baseline
│       └── train_path_following_unicycle_monotonic_fpl.sh  # FPL-accelerated
├── neural_network_lyapunov/             # Library + example systems
│   ├── monotonic_lyapunov/              # Monotonic NN Lyapunov core
│   ├── monotonic_lyapunov_init/         # Pre-training/init utilities
│   └── examples/
│       ├── pendulum/
│       ├── path_following_unicycle/
│       └── cart_pole/                   # Used for teleop demos only
├── paper_viz.py                         # Generates Figs. 1–3 of the paper
├── acc_viz.py                           # Support module for paper_viz.py
├── requirements.txt
└── setup.py                             # Writes config/setup_environments.sh
```

## Setup

### 1. Python environment

Python 3.8 is recommended (the original monotonic baseline was developed on
3.8). Create and activate an environment with your preferred tool, e.g.:

```bash
conda create -n fpl-lyap python=3.8 -y
conda activate fpl-lyap
```

Install requirements:

```bash
pip install -r requirements.txt
```

### 2. Gurobi

Stage II verification uses Gurobi (≥ 9.5). Obtain a license from
https://www.gurobi.com/ and install the Python API:

```bash
pip install gurobipy
python -c "import gurobipy"   # should succeed silently
```

### 3. PYTHONPATH

The training `bash_files/*` scripts add the repo root to `PYTHONPATH`
automatically. If you run Python files directly, either:

- Run `python setup.py` once (writes `config/setup_environments.sh`),
  then `source ./config/setup_environments.sh`; or
- Set it manually: `export PYTHONPATH="$(pwd):${PYTHONPATH}"`.

## Reproducing paper results

All commands below assume you have activated your Python environment and are
at the repository root.

### Training (Table I)

Pendulum:

```bash
# Baseline (no FPL) — Stage II only
bash bash_files/pendulum/train_pendulum_monotonic.sh

# With FPL — Stage I + Stage II
bash bash_files/pendulum/train_pendulum_monotonic_fpl.sh
```

Path-following unicycle:

```bash
# Baseline (no FPL)
bash bash_files/path_following/train_path_following_unicycle_monotonic_40.sh

# With FPL
bash bash_files/path_following/train_path_following_unicycle_monotonic_fpl.sh
```

Trained weights are written under
`neural_network_lyapunov/examples/<system>/data/`. Wall-clock time is printed
at the end of each run (measured via the `SECONDS` builtin).

### Paper figures (Figs. 1–3)

`paper_viz.py` evaluates learned controllers in closed loop and generates the
settling-time, control-effort, and phase-diagram plots used in the paper.

```bash
# Pendulum (Fig. 1a, 2a, 3a)
python paper_viz.py --env pendulum --bound_level 10 --precision float64 \
    --T 100 --dt 0.01 --eps 1e-3 --stride 1 \
    --ic_phase 10 --sample_mode_phase random --ic_seed 12345 \
    --roa_samples 5000 --ic_effort 5000 --sample_mode_effort random \
    --save_dir paper_figs/pendulum/ --gif

# Unicycle (Fig. 1b, 2b, 3b)
python paper_viz.py --env unicycle --bound_level 40 --gpu --precision float64 \
    --T 50 --dt 0.01 --eps 1e-3 --stride 1 \
    --ic_phase 10 --sample_mode_phase random --ic_seed 12345 \
    --roa_samples 5000 --ic_effort 5000 --sample_mode_effort random \
    --save_dir paper_figs/unicycle/ --gif
```

### Per-system comparison plots

Side-by-side comparisons between a controller trained without FPL and one
trained with FPL (expects both checkpoints to exist under `data/`):

```bash
python neural_network_lyapunov/examples/pendulum/pendulum_viz_compare.py \
    --bound_level 10 --save_figs

python neural_network_lyapunov/examples/path_following_unicycle/path_following_viz_compare.py \
    --bound_level 40 --save_figs

python neural_network_lyapunov/examples/cart_pole/cart_pole_viz_compare.py \
    --bound_level 100 --save_figs
```

### Interactive teleop / closed-loop rollout

Roll out a learned controller from a user-specified initial condition:

```bash
# Pendulum (FPL controller, bound 10)
python neural_network_lyapunov/examples/pendulum/teleop_pendulum.py \
    --model neural_network_lyapunov/examples/pendulum/data/pendulum_second_order_forward_relu2.pt \
    --controller NN \
    --controller-model neural_network_lyapunov/examples/pendulum/data/monotonic_bound10_fpl/monotonic_bound10_fpl_controller.pt \
    --init 3 3 --clamp-bounds --compare --umax 20 --dt 0.01

# Unicycle (FPL controller, bound 40)
python neural_network_lyapunov/examples/path_following_unicycle/teleop_path_following.py \
    --model neural_network_lyapunov/examples/path_following_unicycle/data/preprocess/path_following_unicycle_forward_model.pt \
    --clamp-bounds --controller NN \
    --controller-path neural_network_lyapunov/examples/path_following_unicycle/data/monotonic/monotonic_bound40_fpl/monotonic_bound40_fpl_controller.pt \
    --init 0.8 -0.8

# Cart-pole (demo only — uses pre-trained model, not a paper experiment)
python neural_network_lyapunov/examples/cart_pole/teleop_cartpole.py \
    --controller zero \
    --model neural_network_lyapunov/examples/cart_pole/data/preprocess/cart_pole_forward_model_3d.pt \
    --clamp-bounds --model-output accel --model-input theta \
    --x-init 0.0 3.0 0.0 0.0 --dt 0.01 --umax 30 --compare
```

## Hardware used in the paper

All paper experiments ran on CPU only: AMD Ryzen Threadripper 3960x (24C/48T),
62.7 GiB RAM, Ubuntu 20.04.6 LTS. No GPU was required.

## Citation

```bibtex
@inproceedings{abdelgawad2026fpl,
  author    = {Abdelgawad, Abdelrahman and El Mabsout, Bassel and Wang, Zili
               and Mancuso, Renato and Andersson, Sean B. and Tron, Roberto},
  title     = {Accelerating {Lyapunov}-Stable Neural Control using
               {Fulfillment Priority Logic}},
  booktitle = {American Control Conference (ACC)},
  year      = {2026}
}
```

## Acknowledgments

This work was supported in part by NSF FRR-2212051. The authors thank the BU
Robotics Lab for providing computational resources and support.

This codebase is built on top of the monotonic neural Lyapunov baseline of
[Wang et al., ACC 2024](https://doi.org/10.23919/ACC60939.2024.10644877).
