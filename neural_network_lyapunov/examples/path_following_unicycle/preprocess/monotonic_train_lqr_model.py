import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as quadrotor_2d
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.r_options as r_options
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils

import torch
import numpy as np
import scipy.integrate
import gurobipy
import argparse
import os


def generate_quadrotor_dynamics_data(dt):
    """
    Generate the pairs (x[n], u[n]) -> (x[n+1])
    """
    dtype = torch.float64
    plant = quadrotor_2d.Quadrotor2D(dtype)

    theta_range = [-np.pi / 2, np.pi / 2]
    ydot_range = [-5, 5]
    zdot_range = [-5, 5]
    thetadot_range = [-2.5, 2.5]
    u_range = [-0.5, 8.5]
    # We don't need to take the grid on y and z dimension of the quadrotor,
    # since the dynamics is invariant along these dimensions.
    x_samples = torch.cat(
        (
            torch.zeros((1000, 2), dtype=torch.float64),
            utils.uniform_sample_in_box(
                torch.tensor(
                    [theta_range[0], ydot_range[0], zdot_range[0], thetadot_range[0]],
                    dtype=torch.float64,
                ),
                torch.tensor(
                    [theta_range[1], ydot_range[1], zdot_range[1], thetadot_range[1]],
                    dtype=torch.float64,
                ),
                1000,
            ),
        ),
        dim=1,
    ).T
    u_samples = utils.uniform_sample_in_box(
        torch.full((2,), u_range[0], dtype=dtype),
        torch.full((2,), u_range[1], dtype=dtype),
        1000,
    ).T

    xu_tensors = []
    x_next_tensors = []

    for i in range(x_samples.shape[1]):
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


def train_forward_model(forward_model, model_dataset, num_epochs, save_dir=None):
    # The forward model maps (theta[n], u1[n], u2[n]) to
    # (ydot[n+1]-ydot[n], zdot[n+1]-zdot[n], thetadot[n+1]-thetadot[n])
    plant = quadrotor_2d.Quadrotor2D(torch.float64)
    u_equilibrium = plant.u_equilibrium

    xu_inputs, x_next_outputs = model_dataset[:]
    network_input_data = xu_inputs[:, [2, 5, 6, 7]]  # xu_inputs[:, [2, 5, 6, 7]]
    network_output_data = x_next_outputs[:, 3:] - xu_inputs[:, 3:6]
    v_dataset = torch.utils.data.TensorDataset(network_input_data, network_output_data)

    def compute_next_v(model, theta_thetadot_u):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(theta_thetadot_u) - model(
            torch.cat((torch.tensor([0, 0], dtype=torch.float64), u_equilibrium)).to(
                device
            )
        )

    utils.train_approximator(
        v_dataset,
        forward_model,
        compute_next_v,
        batch_size=50,
        num_epochs=num_epochs,
        lr=0.001,
        save_dir=save_dir,
    )


def train_lqr_value_approximator(
    lyapunov_relu,
    V_lambda,
    R,
    x_equilibrium,
    x_lo,
    x_up,
    num_samples,
    lqr_S: torch.Tensor,
):
    """
    We train both lyapunov_relu and R such that ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
    approximates the lqr cost-to-go.
    """
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    V_samples = torch.sum(
        (x_samples.T - x_equilibrium.reshape((6, 1)))
        * (lqr_S @ (x_samples.T - x_equilibrium.reshape((6, 1)))),
        dim=0,
    ).reshape((-1, 1))
    state_value_dataset = torch.utils.data.TensorDataset(x_samples, V_samples)
    R.requires_grad_(True)

    def compute_v(model, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(R.sum())
        return (
            model(x)
            - model(x_equilibrium.to(device))
            + (
                V_lambda
                * torch.norm(
                    R.to(device) @ (x - x_equilibrium.reshape((1, 6)).to(device)).T,
                    p=1,
                    dim=0,
                ).reshape((-1, 1))
            )
        )

    utils.train_approximator(
        state_value_dataset,
        lyapunov_relu,
        compute_v,
        batch_size=50,
        num_epochs=200,
        lr=0.001,
        additional_variable=[R],
    )
    R.requires_grad_(False)


def train_lqr_control_approximator(
    controller_relu,
    x_equilibrium,
    u_equilibrium,
    x_lo,
    x_up,
    num_samples,
    lqr_K: torch.Tensor,
):
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    u_samples = (
        lqr_K @ (x_samples.T - x_equilibrium.reshape((6, 1)))
        + u_equilibrium.reshape((2, 1))
    ).T
    state_control_dataset = torch.utils.data.TensorDataset(x_samples, u_samples)

    def compute_u(model, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(x) - model(x_equilibrium.to(device)) + u_equilibrium.to(device)

    utils.train_approximator(
        state_control_dataset,
        controller_relu,
        compute_u,
        batch_size=50,
        num_epochs=50,
        lr=0.001,
    )


def train_nn_controller_approximator(
    controller_relu, target_controller_relu, x_lo, x_up, num_samples, num_epochs
):
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    target_controller_relu_output = target_controller_relu(x_samples)
    dataset = torch.utils.data.TensorDataset(x_samples, target_controller_relu_output)

    def compute_output(model, x):
        return model(x)

    utils.train_approximator(
        dataset,
        controller_relu,
        compute_output,
        batch_size=50,
        num_epochs=num_epochs,
        lr=0.001,
    )


def train_nn_lyapunov_approximator(
    lyapunov_relu,
    R,
    target_lyapunov_relu,
    target_R,
    V_lambda,
    x_equilibrium,
    x_lo,
    x_up,
    num_samples,
    num_epochs,
):
    x_samples = utils.uniform_sample_in_box(x_lo, x_up, num_samples)
    with torch.no_grad():
        target_V = (
            target_lyapunov_relu(x_samples)
            - target_lyapunov_relu(x_equilibrium)
            + V_lambda
            * torch.norm(target_R @ (x_samples - x_equilibrium).T, p=1, dim=0).reshape(
                (-1, 1)
            )
        )

    dataset = torch.utils.data.TensorDataset(x_samples, target_V)

    def compute_V(model, x):
        return (
            model(x)
            - model(x_equilibrium)
            + V_lambda
            * torch.norm(R @ (x - x_equilibrium).T, p=1, dim=0).reshape((-1, 1))
        )

    R.requires_grad_(True)
    utils.train_approximator(
        dataset,
        lyapunov_relu,
        compute_V,
        batch_size=50,
        num_epochs=num_epochs,
        lr=0.001,
        additional_variable=[R],
    )
    R.requires_grad_(False)


def simulate_quadrotor_with_controller(
    controller_relu, t_span, x_equilibrium, u_lo, u_up, x0
):
    plant = quadrotor_2d.Quadrotor2D(torch.float64)
    u_equilibrium = plant.u_equilibrium

    def dyn(t, x):
        with torch.no_grad():
            x_torch = torch.from_numpy(x)
            u_torch = (
                controller_relu(x_torch)
                - controller_relu(x_equilibrium)
                + u_equilibrium
            )
            u = torch.max(torch.min(u_torch, u_up), u_lo).detach().numpy()
        return plant.dynamics(x, u)

    result = scipy.integrate.solve_ivp(
        dyn, t_span, x0, t_eval=np.arange(start=t_span[0], stop=t_span[1], step=0.01)
    )
    return result


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        description="quadrotor 2d forward model training and lqr controller/lyapunov approximation."
    )
    parser.add_argument(
        "--generate_dynamics_data",
        default=None,  # dir_path+'/data/dynamics_data.pt',
        help="path to save dynamics data.",
    )
    parser.add_argument(
        "--load_dynamics_data",
        type=str,
        default=None,  # dir_path+'/data/dynamics_data.pt',
        help="path to the dynamics data.",
    )
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--train_lqr_approximator", action="store_true")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--training_set", type=str, default=None)
    args = parser.parse_args()
    dtype = torch.float64
    dt = 0.01

    # generate data and train for forward model
    if args.generate_dynamics_data is not None:
        model_dataset = generate_quadrotor_dynamics_data(dt)
        torch.save(model_dataset, args.generate_dynamics_data)

    if args.load_dynamics_data is not None:
        model_dataset = torch.load(args.load_dynamics_data)

    if args.train_forward_model:
        forward_model = utils.setup_relu(
            (4, 6, 6, 3), params=None, bias=True, negative_slope=0.01, dtype=dtype
        )
        train_forward_model(
            forward_model,
            model_dataset,
            num_epochs=100,
            save_dir=args.load_forward_model,
        )
        torch.save(forward_model, args.load_forward_model)

    plant = quadrotor_2d.Quadrotor2D(dtype)
    x_star = np.zeros((6,))
    u_star = plant.u_equilibrium.detach().numpy()
    lqr_Q = np.diag([10, 10, 10, 1, 1, plant.length / 2.0 / np.pi])
    lqr_R = np.array([[0.1, 0.05], [0.05, 0.1]])
    K, S = plant.lqr_control(lqr_Q, lqr_R, x_star, u_star)
    S_eig_value, S_eig_vec = np.linalg.eig(S)
    R = torch.from_numpy(S) + 0.01 * torch.eye(6, dtype=dtype)
    V_lambda = 0.9
    q_equilibrium = torch.tensor([0, 0, 0], dtype=dtype)
    u_equilibrium = plant.u_equilibrium
    x_lo = torch.tensor([-0.7, -0.7, -np.pi * 0.5, -3.75, -3.75, -2.5], dtype=dtype)
    x_up = -x_lo
    u_lo = torch.tensor([0, 0], dtype=dtype)
    u_up = torch.tensor([8, 8], dtype=dtype)

    x_equilibrium = torch.cat((q_equilibrium, torch.zeros((3,), dtype=dtype)))

    lyapunov_relu = monotonic_utils.setup_monotonic_relu(
        size_out=1,
        size_in=x_equilibrium.numel(),
        epsilon=0.005,
        size_partition=10,
        size_piecewise=3,
        params=None,
        dtype=torch.float64,
        x_eqlm=x_equilibrium,
        symm_flag=False,
        symm_v_dir_flag=False,
    )

    controller_relu = utils.setup_relu(
        (6, 6, 4, 2), params=None, negative_slope=0.01, bias=True, dtype=dtype
    )

    if args.enable_wandb:
        train_utils.wandb_config_update(
            args, lyapunov_relu, controller_relu, x_lo, x_up, u_lo, u_up
        )
    if args.train_lqr_approximator:
        lqr_path = dir_path + "/../data/preprocess/"
        controller_relu_path = lqr_path + "lqr_controller_monotonic.pt"
        lyapunov_relu_path = lqr_path + "lqr_lyapunov_monotonic.pt"
        lyapunov_R_path = lqr_path + "lqr_R_monotonic.pt"
        print(os.path.exists(lqr_path))
        # if os.path.exists(lqr_path):

        #     print("train lqr controller approximation")
        #     train_lqr_control_approximator(controller_relu, x_equilibrium,
        #                                 u_equilibrium, x_lo, x_up, 100000,
        #                                 torch.from_numpy(K))
        #     torch.save(controller_relu, controller_relu_path)
        #     print("save lqr controller approximation\n\n")

        if os.path.exists(lqr_path):
            print("train lqr lyapunov approximation")
            train_lqr_value_approximator(
                lyapunov_relu,
                V_lambda,
                R,
                x_equilibrium,
                x_lo,
                x_up,
                100000,
                torch.from_numpy(S),
            )
            torch.save(lyapunov_relu, lyapunov_relu_path)
            torch.save(R, lyapunov_R_path)
            print("save lqr lyapunov approximation")
