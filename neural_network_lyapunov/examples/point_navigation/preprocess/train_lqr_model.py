import torch
import numpy as np
import scipy.integrate
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules with correct paths
# Option 1: Full module path (recommended)
from neural_network_lyapunov.examples.point_navigation.point_navigation import Unicycle

# Option 2: If running from preprocess directory, use relative import
# from ..point_navigation import Unicycle
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov


def generate_dynamics_data(dt):
    """Generate (x[n], u[n]) -> x[n+1] pairs for unicycle"""
    dtype = torch.float64
    plant = Unicycle(dtype)

    # State ranges [x, y, theta]
    x_range = [-2, 2]
    y_range = [-2, 2]
    theta_range = [-np.pi, np.pi]

    # Control ranges [v, omega]
    v_range = [0, 2]
    omega_range = [-np.pi, np.pi]

    x_samples = utils.uniform_sample_in_box(
        torch.tensor([x_range[0], y_range[0], theta_range[0]], dtype=dtype),
        torch.tensor([x_range[1], y_range[1], theta_range[1]], dtype=dtype),
        1000,
    ).T

    u_samples = utils.uniform_sample_in_box(
        torch.tensor([v_range[0], omega_range[0]], dtype=dtype),
        torch.tensor([v_range[1], omega_range[1]], dtype=dtype),
        1000,
    ).T

    xu_tensors = []
    x_next_tensors = []

    for i in range(x_samples.shape[1]):
        if i % 100 == 0:
            print(f"Processing sample {i}/1000")
        for j in range(u_samples.shape[1]):
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, u_samples[:, j].detach().numpy()),
                (0, dt),
                x_samples[:, i].detach().numpy(),
            )
            xu_tensors.append(
                torch.cat((x_samples[:, i], u_samples[:, j])).reshape((1, -1))
            )
            x_next_tensors.append(torch.from_numpy(result.y[:, -1]).reshape((1, -1)))

    dataset_input = torch.cat(xu_tensors, dim=0)
    dataset_output = torch.cat(x_next_tensors, dim=0)
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(dynamics_model, model_dataset, num_epochs=20, save_dir=None):
    """Train the forward dynamics model"""
    dtype = torch.float64
    state_equilibrium = torch.zeros(3, dtype=dtype)  # [0, 0, 0]
    control_equilibrium = torch.zeros(2, dtype=dtype)  # [0, 0]

    (xu_inputs, x_next_outputs) = model_dataset[:]

    def compute_next_state(model, state_action):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(state_action) - model(
            torch.cat((state_equilibrium.to(device), control_equilibrium.to(device)))
        )

    utils.train_approximator(
        model_dataset,
        dynamics_model,
        compute_next_state,
        batch_size=500,
        num_epochs=num_epochs,
        lr=0.001,
        save_dir=save_dir,
    )


def train_lqr_control_approximator(
    controller_relu, x_equilibrium, u_equilibrium, x_lo, x_up, num_samples, K
):
    """Train controller to approximate LQR"""
    dtype = torch.float64
    states = utils.uniform_sample_in_box(x_lo, x_up, num_samples).T
    controls = (K @ (states - x_equilibrium.reshape((-1, 1)))).T + u_equilibrium

    dataset = torch.utils.data.TensorDataset(states.T, controls)

    def compute_u(model, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(x) - model(x_equilibrium.to(device)) + u_equilibrium.to(device)

    utils.train_approximator(
        dataset, controller_relu, compute_u, batch_size=50, num_epochs=500, lr=0.001
    )


def train_lqr_value_approximator(
    lyapunov_relu, V_lambda, R, x_equilibrium, x_lo, x_up, num_samples, S
):
    """Train Lyapunov function to approximate LQR value"""
    dtype = torch.float64
    states = utils.uniform_sample_in_box(x_lo, x_up, num_samples).T
    costs = 0.5 * torch.sum(
        (states - x_equilibrium.reshape((-1, 1)))
        * (torch.from_numpy(S) @ (states - x_equilibrium.reshape((-1, 1)))),
        dim=0,
    ).reshape((-1, 1))

    dataset = torch.utils.data.TensorDataset(states.T, costs)

    R.requires_grad_(True)

    def compute_v(model, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return (
            model(x)
            - model(x_equilibrium.to(device))
            + V_lambda
            * torch.norm(R @ (x - x_equilibrium).T, p=1, dim=0)
            .reshape((-1, 1))
            .to(device)
        )

    utils.train_approximator(
        dataset,
        lyapunov_relu,
        compute_v,
        batch_size=50,
        num_epochs=500,
        lr=0.001,
        additional_variable=[R],
    )
    R.requires_grad_(False)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        description="Unicycle forward model training and LQR approximation"
    )
    parser.add_argument(
        "--generate_dynamics_data", default=None, help="path to save dynamics data"
    )
    parser.add_argument(
        "--load_dynamics_data",
        type=str,
        default=None,
        help="path to load dynamics data",
    )
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--train_lqr_approximator", action="store_true")
    args = parser.parse_args()

    dtype = torch.float64
    dt = 0.01

    # Data generation
    if args.generate_dynamics_data:
        print("Generating dynamics dataset...")
        dynamics_dataset = generate_dynamics_data(dt)
        torch.save(dynamics_dataset, args.generate_dynamics_data)
        print(f"Saved to {args.generate_dynamics_data}")

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)
        print(f"Loaded dataset from {args.load_dynamics_data}")

    # Train forward model
    if args.train_forward_model:
        print("Training forward model...")
        # Network: 5 inputs (x,y,theta,v,omega) -> 3 outputs (x',y',theta')
        dynamics_relu = utils.setup_relu(
            (5, 16, 16, 8, 3), params=None, negative_slope=0.1, bias=True, dtype=dtype
        )
        save_path = os.path.join(
            dir_path, "..", "data", "preprocess", "unicycle_forward_model.pt"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        train_forward_model(
            dynamics_relu, dynamics_dataset, num_epochs=100, save_dir=save_path
        )
        torch.save(dynamics_relu, save_path)
        print(f"Saved model to {save_path}")

    # Train LQR approximators
    if args.train_lqr_approximator:
        print("Training LQR approximators...")

        # Setup plant and compute LQR gains
        plant = Unicycle(dtype)
        x_equilibrium = torch.zeros(3, dtype=dtype)
        u_equilibrium = torch.zeros(2, dtype=dtype)

        Q = np.diag([10.0, 10.0, 1.0])  # Penalize position more than orientation
        R = np.diag([1.0, 1.0])  # Control penalty
        K, S = plant.lqr_control(Q, R)
        K = torch.from_numpy(K)

        # State bounds for training
        x_lo = torch.tensor([-1.0, -1.0, -np.pi / 2], dtype=dtype)
        x_up = torch.tensor([1.0, 1.0, np.pi / 2], dtype=dtype)

        # Controller network: 3 inputs -> 2 outputs
        controller_relu = utils.setup_relu(
            (3, 20, 15, 10, 2), params=None, negative_slope=0.1, bias=True, dtype=dtype
        )

        # Lyapunov network: 3 inputs -> 1 output
        lyapunov_relu = utils.setup_relu(
            (3, 20, 15, 10, 1), params=None, negative_slope=0.01, bias=True, dtype=dtype
        )

        # R matrix for L1 norm regularization
        R = torch.cat(
            (
                torch.eye(3, dtype=dtype),
                torch.tensor([[1, -1, 0], [-1, -1, 1], [0, 1, 1]], dtype=dtype),
            ),
            dim=0,
        )
        V_lambda = 0.5

        # Train controller
        print("Training controller...")
        controller_save_path = os.path.join(
            dir_path, "..", "data", "preprocess", "unicycle_lqr_controller.pt"
        )
        os.makedirs(os.path.dirname(controller_save_path), exist_ok=True)
        train_lqr_control_approximator(
            controller_relu, x_equilibrium, u_equilibrium, x_lo, x_up, 100000, K
        )
        torch.save(controller_relu, controller_save_path)

        # Train Lyapunov
        print("Training Lyapunov function...")
        lyapunov_save_path = os.path.join(
            dir_path, "..", "data", "preprocess", "unicycle_lqr_lyapunov.pt"
        )
        R_save_path = os.path.join(
            dir_path, "..", "data", "preprocess", "unicycle_lqr_R.pt"
        )
        train_lqr_value_approximator(
            lyapunov_relu, V_lambda, R, x_equilibrium, x_lo, x_up, 100000, S
        )
        torch.save(lyapunov_relu, lyapunov_save_path)
        torch.save(R, R_save_path)

        print("Training complete!")
