from __future__ import annotations
from typing import Union, Iterable, Optional
import torch
import scipy.integrate
import numpy as np
import argparse
import os

import neural_network_lyapunov.examples.point_navigation.point_navigation as point_navigation
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.r_options as r_options

# import neural_network_lyapunov.monotonic_lyapunov.custom_lyapunov as lyapunov
# import neural_network_lyapunov.monotonic_lyapunov.custom_train_lyapunov_barrier as train_lyapunov_barrier
# import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
import neural_network_lyapunov.monotonic_lyapunov_init.custom_lyapunov as lyapunov
import neural_network_lyapunov.monotonic_lyapunov_init.custom_train_lyapunov_barrier as train_lyapunov_barrier

# import neural_network_lyapunov.monotonic_lyapunov_init.monotonic_utils as monotonic_utils
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils_0615 as monotonic_utils

from neural_network_lyapunov.examples.point_navigation.preprocess.fpl import *


def rotation_matrix(theta):
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    return torch.tensor([[c_theta, -s_theta], [s_theta, c_theta]], dtype=torch.float64)


def generate_point_nav_dynamics_data(
    dt=0.01, u_lo=-10.0, u_up=10.0, x_lo=-1.0, x_up=1.0
):
    """
    Dataset for point navigation: (x,u) -> x_next.
    x = [d, theta_e], u = [omega].
    """
    dtype = torch.float64
    plant = point_navigation.Unicycle(dtype=dtype)

    def step_once(x_np, u_np):
        out = scipy.integrate.solve_ivp(
            lambda t, x_: plant.dynamics(x_, u_np), (0, dt), x_np, t_eval=[dt]
        )
        return out.y[:, -1]

    states, controls, next_states = [], [], []

    # Rollouts from a grid with a simple LQR for coverage
    Q = np.diag([2.0, 2.0, 2.0])  # penalize x,y more than theta
    R = np.diag([0.5, 0.5])  # penalize v and omega
    K, _ = plant.lqr_control(Q, R)

    x0s = np.linspace(-1.0, 1.0, 25)
    y0s = np.linspace(-1.0, 1.0, 25)
    th0s = np.linspace(-np.pi, np.pi, 25)
    counter = 0
    print(plant.v0 / dt)
    for x0 in x0s:
        for y0 in y0s:
            for th0 in th0s:
                x = np.array([x0, y0, th0], dtype=float)
                for _ in range(int(plant.v0 / dt)):
                    u = (K @ x).reshape(2)  # u = [v, omega]
                    x_next = step_once(x, u)
                    states.append(torch.tensor(x, dtype=dtype))
                    controls.append(torch.tensor(u, dtype=dtype).view(-1))
                    next_states.append(torch.tensor(x_next, dtype=dtype))
                    x = x_next
                    counter += 1
                    print(f"Generated {counter} samples", end="\r")
    # Random one-steps to fill corners
    rng = np.random.default_rng(0)
    for _ in range(10000):
        x = np.array(
            [
                rng.uniform(x_lo[0], x_up[0]),
                rng.uniform(x_lo[1], x_up[1]),
                rng.uniform(x_lo[2], x_up[2]),
            ],
            dtype=float,
        )
        u = np.array(
            [rng.uniform(u_lo[0], u_up[0]), rng.uniform(u_lo[1], u_up[1])], dtype=float
        )
        x_next = step_once(x, u)
        states.append(torch.tensor(x, dtype=dtype))
        controls.append(torch.tensor(u, dtype=dtype).view(-1))
        next_states.append(torch.tensor(x_next, dtype=dtype))

    X = torch.stack(states)  # (N, 3)
    U = torch.stack(controls)  # (N, 2)
    Xn = torch.stack(next_states)  # (N, 3)
    return torch.utils.data.TensorDataset(torch.cat((X, U), dim=1), Xn)  # ((N,3),(N,3))


def train_forward_model(
    dynamics_model,
    model_dataset,
    state_equilibrium: torch.Tensor = torch.tensor([0.0, 0.0], dtype=torch.float64),
    control_equilibrium: torch.Tensor = torch.tensor([0.0], dtype=torch.float64),
    num_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
):
    """
    Train φ_dyn so that the composed map
        f(x,u) = φ_dyn(x,u) - φ_dyn(x*,u*) + x*
    matches the true one-step next state x_next on dataset samples.
    We *do not* bake the shift into the saved model; we only use it in the loss.
    At runtime, ReLUSystemGivenEquilibrium will apply the same shift.
    """
    xu, x_next = model_dataset[
        :
    ]  # xu: (N, 5) with [x, y, theta, v, omega],  x_next: (N, 3)

    # Precompute the equilibrium (x*, u*) in the same dtype/device as xu
    def make_eq(batch_like):
        eq = torch.cat((state_equilibrium, control_equilibrium)).to(batch_like)
        return eq

    eq_once = make_eq(xu).unsqueeze(0)  # shape (1, 3)

    def fwd(phi, inp):
        # Broadcast eq to batch, then apply (1a): f = φ(x,u) - φ(x*,u*) + x*
        eq = eq_once.expand(inp.shape[0], -1).to(inp)
        pred = phi(inp) - phi(eq) + state_equilibrium.to(inp)
        return pred  # (N, 2)

    utils.train_approximator(
        model_dataset,
        dynamics_model,
        fwd,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
    )


def generate_controller_dataset():
    """
    Generate the dataset to train a controller and value function.
    Simulate the pendulum dynamics (and cost-to-go) using an energy shaping +
    LQR controller. For each state on the simulated trajectory, obtain the
    control action and cost-to-go.
    """
    plant = pendulum.Pendulum(torch.float64)
    Q = 0.1 * np.diag([1, 10])
    R = 0.1 * np.array([[1]])
    x_des = np.array([np.pi, 0])

    def state_cost_dot(y, u):
        x = y[:2]
        x_des = np.array([np.pi, 0])
        # return [xdot; cost(x, u)]
        return np.hstack(
            (plant.dynamics(x, u), (x - x_des).dot(Q @ (x - x_des)) + u.dot(R @ u))
        )

    lqr_gain = plant.lqr_control(Q, R)

    def controller(x):
        if (x - x_des).dot(Q @ (x - x_des)) > 0.1:
            u = plant.energy_shaping_control(x, x_des, 1)
        else:
            u = lqr_gain @ (x - x_des)
        return u

    def converged(t, y):
        x = y[:2]
        return np.linalg.norm(x - x_des) - 1e-2

    converged.terminal = True
    # Now take a grid of initial states, simulate them using the controller.
    theta0 = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 11)
    thetadot0 = np.linspace(-2, 2, 11)
    states = []
    controls = []
    costs = []
    dt = 0.01
    for i in range(theta0.shape[0]):
        for j in range(thetadot0.shape[0]):
            result = scipy.integrate.solve_ivp(
                lambda t, y: state_cost_dot(y, controller(y[:2])),
                (0, 15),
                np.array([theta0[i], thetadot0[j], 0]),
                t_eval=np.arange(0, 15, dt),
                events=converged,
            )
            if converged(0, result.y[:, -1]) <= 1e-4:
                states.append(result.y[:2, :])
                controls.append(
                    [controller(result.y[:2, i])[0] for i in range(result.y.shape[1])]
                )
                interval_cost = np.hstack((result.y[2, 1:] - result.y[2, :-1], 0))
                costs.append(np.cumsum(interval_cost[::-1])[::-1])
    return (
        torch.from_numpy(np.hstack(states).T),
        torch.from_numpy(np.hstack(controls).reshape((-1, 1))),
        torch.from_numpy(np.hstack(costs).reshape((-1, 1))),
    )


def train_controller_approximator(state_samples, control_samples, controller_relu, lr):
    """
    Given some state-action pairs, train a controller ϕ(x) − ϕ(x*) + u* to
    approximate these state-action pairs.
    """
    control_dataset = torch.utils.data.TensorDataset(state_samples, control_samples)

    state_equilibrium = torch.tensor([np.pi, 0], dtype=torch.float64)
    control_equilibrium = torch.tensor([0], dtype=torch.float64)

    def compute_control(model, dataset):
        return model(dataset) - model(state_equilibrium) + control_equilibrium

    utils.train_approximator(
        control_dataset,
        controller_relu,
        compute_control,
        batch_size=20,
        num_epochs=400,
        lr=lr,
    )


def train_cost_approximator(state_samples, cost_samples, cost_relu, V_lambda):
    """
    Given many state-cost pairs, train a value approximator
    ϕ(x) − ϕ(x*)+λ|x − x*|₁
    """
    cost_dataset = torch.utils.data.TensorDataset(state_samples, cost_samples)
    state_equilibrium = torch.tensor([np.pi, 0], dtype=torch.float64)

    def compute_cost(model, data):
        return (
            model(data)
            - model(state_equilibrium)
            + V_lambda
            * torch.norm(data - state_equilibrium, p=1, dim=1).reshape((-1, 1))
        )

    utils.train_approximator(
        cost_dataset, cost_relu, compute_cost, batch_size=20, num_epochs=300, lr=0.001
    )


# def pendulum_closed_loop_dynamics(plant: pendulum.Pendulum, x: np.ndarray,
#                                   controller_relu, x_equilibrium,
#                                   u_equilibrium, u_lo, u_up):
#     assert (isinstance(plant, pendulum.Pendulum))
#     u_pre_saturation = controller_relu(torch.from_numpy(x)) -\
#         controller_relu(x_equilibrium) + u_equilibrium
#     u = torch.max(torch.min(u_pre_saturation, u_up), u_lo).detach().numpy()
#     return plant.dynamics(x, u)


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="pendulum training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument(
        "--load_dynamics_data", type=str, default=None, help="path of the dynamics data"
    )
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--generate_controller_cost_data", action="store_true")
    parser.add_argument("--train_controller_approximator", action="store_true")
    parser.add_argument("--train_cost_approximator", action="store_true")
    parser.add_argument("--load_controller_cost_data", action="store_true")
    parser.add_argument(
        "--load_lyapunov_relu",
        type=str,
        default=None,
        help="path of the saved lyapunov_relu state_dict()",
    )
    parser.add_argument(
        "--load_controller_relu",
        type=str,
        default=None,  # dir_path+"/data/pendulum_controller4.pt",#None,
        help="path of the controller relu state_dict()",
    )
    parser.add_argument(
        "--pretrain_num_epochs",
        type=int,
        default=200,
        help="number of epochs in pre-training on samples.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5000,
        help="max number of iterations in searching for controller.",
    )
    parser.add_argument(
        "--search_R",
        action="store_true",
        help="search R when searching for controller.",
    )
    parser.add_argument(
        "--train_on_samples",
        action="store_true",
        help="pretrain Lyapunov controller on samples.",
    )
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--train_adversarial", action="store_true")

    parser.add_argument("--bound_level", type=int, default=1, help="bound level of x.")
    parser.add_argument(
        "--bound_level_last",
        type=int,
        default=0,
        help="bound level of x from pre-trained controller and lyapunov.",
    )
    parser.add_argument(
        "--use_fpl",
        action="store_true",
        help="use FPL for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-3,
        help="Learning rate for FPL training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for FPL training"
    )

    args = parser.parse_args()

    args.search_R = True
    search_controllerFalg = True  # True
    if not search_controllerFalg:
        args.load_controller_relu = (
            dir_path + "/data/monotonic_V06/monotonic_bound10_controller.pt"
        )
    add_l1_stateFalg = True  # True
    lyap_v_symm_flag = False
    bound_level = args.bound_level
    bound_level_x = bound_level
    bound_level_y = bound_level
    print("bound level is: ", bound_level)
    if args.bound_level_last >= 1:
        bound_level_last = args.bound_level_last  # bound_level-1
        args.load_controller_relu = (
            dir_path
            + "/data/monotonic/monotonic_bound"
            # + "/data/monotonic_roa/monotonic_bound"
            + str(bound_level_last)
            + f"/monotonic_bound{bound_level_last}_controller.pt"
        )
        args.load_lyapunov_relu = (
            dir_path
            + "/data/monotonic/monotonic_bound"
            # + "/data/monotonic_roa/monotonic_bound"
            + str(bound_level_last)
            + f"/monotonic_bound{bound_level_last}_lyapunov.pt"
        )
        args.load_lyapunov_R = (
            dir_path
            + "/data/monotonic/monotonic_bound"
            # + "/data/monotonic_roa/monotonic_bound"
            + str(bound_level_last)
            + f"/monotonic_bound{bound_level_last}_R.pt"
        )
        print("pre-trained bound level is: ", bound_level_last)
    print("pretrained lyapunov path: ", args.load_lyapunov_relu)
    # x_lo = (
    #     torch.tensor([-0.8 * bound_level_x, -0.8 * bound_level_y], dtype=torch.float64)
    #     / 40.0
    # )
    # x_up = (
    #     torch.tensor([0.8 * bound_level_x, 0.8 * bound_level_y], dtype=torch.float64)
    #     / 40.0
    # )
    # print("input bound: ", x_lo, x_up)

    # Proper bounds for [distance, bearing_error]
    x_lo = torch.tensor(
        [-bound_level * 0.025, -bound_level * 0.025, -np.pi], dtype=torch.float64
    )
    x_up = torch.tensor(
        [bound_level * 0.025, bound_level * 0.025, np.pi], dtype=torch.float64
    )  # Scale with bound level
    print("input bound: ", x_lo, x_up)

    dt = 0.01

    # Equilibrium for point navigation
    q_equilibrium = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    u_equilibrium = torch.tensor([0.0, 0.0], dtype=torch.float64)

    # (A) Generate data when requested
    if args.generate_dynamics_data:
        model_dataset = generate_point_nav_dynamics_data(
            dt=dt,
            u_lo=[-10.0, -10.0],
            u_up=[10.0, 10.0],
            x_lo=[-1.0, -1.0, -np.pi],
            x_up=[1.0, 1.0, np.pi],
        )
        torch.save(
            {"input": model_dataset[:][0], "output": model_dataset[:][1]},
            os.path.join(dir_path, "data", "point_nav_forward_data.pt"),
        )

    # (B) Or load saved data
    if args.load_dynamics_data is not None:
        data = torch.load(args.load_dynamics_data)
        model_dataset = torch.utils.data.TensorDataset(data["input"], data["output"])

    # (C) Train φ_dyn using the equilibrium-shifted loss (1a)
    if args.train_forward_model:
        dynamics_relu = utils.setup_relu(
            # (3, 8, 8, 2),
            (5, 8, 8, 3),
            params=None,
            negative_slope=0.1,
            bias=True,
            dtype=torch.float64,
        )
        train_forward_model(
            dynamics_relu,
            model_dataset,
            state_equilibrium=q_equilibrium,
            control_equilibrium=u_equilibrium,
            # num_epochs=200,
            # lr=2e-3,
            # batch_size=512,
            num_epochs=50,
            lr=2e-4,
            batch_size=512,
        )
        torch.save(
            dynamics_relu,
            os.path.join(dir_path, "data", "preprocess", "point_nav_forward_model.pt"),
        )
    else:
        dynamics_relu = torch.load(
            os.path.join(dir_path, "data", "preprocess", "point_nav_forward_model.pt"),
            map_location=torch.device("cpu"),
        ).double()

    # # (D) Build the NN forward system that *internally* applies (1a) at runtime
    # x_lo = torch.tensor([0.0, -np.pi], dtype=torch.float64)
    # x_up = torch.tensor([20.0, np.pi], dtype=torch.float64)
    # u_lo = torch.tensor([-3.0], dtype=torch.float64)
    # u_up = torch.tensor([3.0], dtype=torch.float64)
    # forward_system = relu_system.ReLUSystemGivenEquilibrium(
    #     torch.float64,
    #     x_lo,
    #     x_up,
    #     u_lo,
    #     u_up,
    #     dynamics_relu,
    #     q_equilibrium,
    #     u_equilibrium,
    #     dt=0.02,
    # )

    if args.generate_controller_cost_data:
        state_samples, control_samples, cost_samples = generate_controller_dataset()
    elif args.load_controller_cost_data:
        controller_cost_data = torch.load(
            dir_path + "/data/pendulum_controller_cost_data.pt"
        )
        state_samples = controller_cost_data["state_samples"]
        control_samples = controller_cost_data["control_samples"]
        cost_samples = controller_cost_data["cost_samples"]

    V_lambda = 0.6
    controller_relu = utils.setup_relu(
        # (2, 4, 3, 1),
        (3, 8, 8, 2),
        # (3, 4, 4, 2),
        params=None,
        negative_slope=0.1,
        bias=True,
        dtype=torch.float64,
    )
    if args.train_controller_approximator:
        train_controller_approximator(
            state_samples, control_samples, controller_relu, lr=0.001
        )
    elif args.load_controller_relu is not None:
        # controller_data = torch.load(args.load_controller_relu)
        # controller_relu = utils.setup_relu(
        #     controller_data["linear_layer_width"],
        #     params=None,
        #     negative_slope=controller_data["negative_slope"],
        #     bias=True,
        #     dtype=torch.float64)
        # controller_relu.load_state_dict(controller_data["state_dict"])
        controller_relu = torch.load(args.load_controller_relu)

    # plant = point_navigation.Point_Stabilization(torch.float64)
    plant = point_navigation.Unicycle(torch.float64)
    # lqr_gain = plant.lqr_control(np.diag([1., 10.]), np.array([[1.]]))

    dtype = torch.float64
    x_star = np.zeros((2,))
    # u_star = np.ones((1,)) * plant.v
    u_star = np.zeros((2,))
    # lqr_Q = np.diag([1, 1])
    lqr_Q = np.diag([1, 1, 1])
    # lqr_R = np.array([[1]])
    lqr_R = np.diag([1, 1])
    K, S = plant.lqr_control(lqr_Q, lqr_R)
    S_eig_value, S_eig_vec = np.linalg.eig(S)

    # R = torch.from_numpy(S) + 0.01 * torch.eye(2, dtype=torch.float64)
    R = torch.from_numpy(S) + 0.01 * torch.eye(3, dtype=torch.float64)

    # Now train the controller and Lyapunov function together
    # q_equilibrium = torch.tensor([0.0, 0.0], dtype=torch.float64)
    q_equilibrium = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    u_equilibrium = torch.tensor([0.0, 0.0], dtype=torch.float64)
    # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-10, -10], dtype=torch.float64)
    u_up = torch.tensor([10, 10], dtype=torch.float64)

    # --- ENSURE MILP IS CPU-ONLY ---
    dynamics_relu = dynamics_relu.to("cpu").double().eval()
    for p in dynamics_relu.parameters():
        p.requires_grad_(False)

    forward_system = relu_system.ReLUSystemGivenEquilibrium(
        torch.float64,
        x_lo,
        x_up,
        u_lo,
        u_up,
        dynamics_relu,
        q_equilibrium,
        u_equilibrium,
        dt,
    )
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system,
        controller_relu,
        forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy(),
    )

    # R = torch.zeros_like(R)
    # lyapunov_relu = utils.setup_relu((2, 8, 8, 6, 1),
    #                                 params=None,
    #                                 negative_slope=0.1,
    #                                 bias=True,
    #                                 dtype=torch.float64)
    lyapunov_relu = monotonic_utils.setup_monotonic_relu(
        size_out=1,
        size_in=forward_system.x_equilibrium.numel(),
        epsilon=0.01,
        size_partition=6,
        size_piecewise=4,
        params=None,
        dtype=torch.float64,
        x_eqlm=forward_system.x_equilibrium,
        provided_v=S_eig_vec,
    )

    # # Test if the closed-loop system actually stabilizes
    # print("\n=== Testing Closed-Loop Stability ===")
    # test_state = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    # for i in range(10):
    #     v_before = lyapunov_relu(test_state).item()
    #     test_state_next = closed_loop_system.step_forward(test_state)
    #     v_after = lyapunov_relu(test_state_next).item()

    #     print(f"Step {i}: state={test_state[0].tolist()}, V={v_before:.6f}")
    #     print(f"  -> next_state={test_state_next[0].tolist()}, V={v_after:.6f}")
    #     print(f"  -> V_decrease={v_before - v_after:.6f}")

    #     test_state = test_state_next

    #     if torch.norm(test_state).item() < 0.01:
    #         print("Converged to equilibrium!")
    #         break

    ###############################
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    actor = controller_relu.to(device).double()
    V = lyapunov_relu.to(device).double()

    for m in (actor, V, dynamics_relu):
        for p in m.parameters():
            p.data = p.data.double()

    ############################

    if args.train_cost_approximator:
        train_cost_approximator(state_samples, cost_samples, lyapunov_relu, V_lambda)
    elif args.load_lyapunov_relu is not None:
        # lyapunov_data = torch.load(args.load_lyapunov_relu)
        # lyapunov_relu = utils.setup_relu(
        #     lyapunov_data["linear_layer_width"],
        #     params=None,
        #     negative_slope=lyapunov_data["negative_slope"],
        #     bias=True,
        #     dtype=torch.float64)
        # lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        # V_lambda = lyapunov_data["V_lambda"]
        # R = lyapunov_data["R"]
        R = torch.load(args.load_lyapunov_R)
        lyapunov_relu = torch.load(args.load_lyapunov_relu)
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu
    )

    lyapunov_hybrid_system.add_l1_state = add_l1_stateFalg

    if args.search_R:
        R_options = r_options.SearchRwithSPDOptions(R.shape, epsilon=0.01)
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = r_options.FixedROptions(R)

    if args.enable_wandb:
        train_utils.wandb_config_update(
            args, lyapunov_relu, controller_relu, x_lo, x_up, u_lo, u_up
        )
    dut = train_lyapunov_barrier.Trainer()
    dut.add_lyapunov(
        lyapunov_hybrid_system, V_lambda, closed_loop_system.x_equilibrium, R_options
    )
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1e-5
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.5
    dut.lyapunov_derivative_epsilon = 0.001
    dut.lyapunov_derivative_eps_type = lyapunov.ConvergenceEps.ExpLower
    state_samples_all = utils.get_meshgrid_samples(
        x_lo, x_up, (51, 51, 51), dtype=torch.float64
    )
    dut.output_flag = True
    dut.search_controller = search_controllerFalg

    if args.use_fpl:
        # Create FPL trainer
        fpl_trainer = FPLMonotonicLyapunovTrainer(
            lyapunov_hybrid_system,
            closed_loop_system,
            V_lambda,
            closed_loop_system.x_equilibrium,
            R_options,
            x_lo=x_lo,
            x_up=x_up,
        )
        state_samples_all_fpl = utils.get_meshgrid_samples(
            x_lo, x_up, (15, 15, 15), dtype=torch.float64
        )
        # Train with FPL
        train_with_fpl(fpl_trainer, state_samples_all_fpl, args)

    elif args.train_on_samples:
        dut.train_lyapunov_on_samples(
            state_samples_all, num_epochs=args.pretrain_num_epochs, batch_size=64
        )
    dut.enable_wandb = args.enable_wandb

    dut.save_network_path = (
        # dir_path + "/data/monotonic_roa/monotonic_bound" + str(bound_level) + "_"
        dir_path
        + "/data/monotonic/monotonic_bound"
        + str(bound_level)
        # + "_"
    )
    if args.use_fpl:
        dut.save_network_path = (
            # dir_path + "/data/monotonic_roa/monotonic_bound" + str(bound_level) + "_"
            dir_path
            + "/data/monotonic/monotonic_bound"
            + str(bound_level)
            + "_fpl"
        )
    if args.train_adversarial:
        dut.save_network_path += "adversarial_"
        options = train_lyapunov_barrier.Trainer.AdversarialTrainingOptions()
        options.positivity_samples_pool_size = 1000
        options.derivative_samples_pool_size = 1000
        dut.add_derivative_adversarial_state = True
        dut.add_positivity_adversarial_state = True
        dut.lyapunov_positivity_mip_pool_solutions = 50
        dut.lyapunov_derivative_mip_pool_solutions = 50
        # dut.lyapunov_positivity_sample_cost_weight = 0.
        # dut.lyapunov_derivative_sample_cost_weight = 10.
        positivity_state_samples_init = utils.get_meshgrid_samples(
            x_lo, x_up, (21, 21), torch.float64
        )
        derivative_state_samples_init = positivity_state_samples_init
        result = dut.train_adversarial(
            positivity_state_samples_init, derivative_state_samples_init, options
        )
    else:
        # print(f"Learning Rate is: {dut.learning_rate}")
        # dut.learning_rate = 0.001
        # dut.learning_rate = 0.01
        dut.lyapunov_positivity_mip_cost_weight = None
        # dut.boundary_value_gap_mip_cost_weight = 0.0
        # dut.lyapunov_upper = 1.#1.#None
        dut.train(torch.empty((0, 3), dtype=torch.float64))
    pass
