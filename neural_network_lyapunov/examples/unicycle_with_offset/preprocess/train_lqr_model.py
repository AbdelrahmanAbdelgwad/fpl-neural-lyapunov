import neural_network_lyapunov.examples.unicycle_with_offset.unicycle_with_offset as unicycle
import neural_network_lyapunov.examples.unicycle_with_offset.unicycle_feedback_system as\
    unicycle_feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.r_options as r_options

import torch
import numpy as np
import scipy.integrate
import gurobipy
import argparse
import os


def generate_dynamics_data(dt):
    """
    Generate many pairs of state/action to next state.
    Notice that since the car dynamics is shift invariant, our input doesn't
    include the car position.
    """
    dtype = torch.float64
    plant = unicycle.Unicycle(dtype)
    theta_grid = torch.linspace(-1.05 * np.pi, 1.05 * np.pi, 201, dtype=dtype)
    vel_grid = torch.linspace(0, 1, 201, dtype=dtype)
    thetadot_grid = torch.linspace(-0.15 * np.pi,
                                   0.15 * np.pi,
                                   51,
                                   dtype=dtype)

    state_action_data = []
    state_next_data = []

    for i in range(theta_grid.numel()):
        print(i)
        for j in range(vel_grid.numel()):
            for k in range(thetadot_grid.numel()):
                xu = torch.tensor(
                    [0, 0, theta_grid[i], vel_grid[j], thetadot_grid[k]],
                    dtype=dtype)
                x_next = plant.next_pose(xu[:3], xu[3:], dt)
                state_action_data.append(xu.reshape((1, -1)))
                state_next_data.append(
                    torch.from_numpy(x_next).reshape((1, -1)))
    state_action_data = torch.cat(state_action_data, dim=0)
    state_next_data = torch.cat(state_next_data, dim=0)
    return torch.utils.data.TensorDataset(state_action_data, state_next_data)


def train_forward_model(dynamics_relu, dataset, num_epochs, thetadot_as_input,save_dir=None):
    """
    The dataset contains the mapping from state_action to state_next
    """
    (xu_inputs, x_next_outputs) = dataset[:]
    # The network output is just the delta_pos_x and delta_pos_y
    training_dataset = torch.utils.data.TensorDataset(
        xu_inputs, x_next_outputs[:, :2] - xu_inputs[:, :2])

    def model_output(model, state_action):
        # state_action is the concatenation of state (pos_x, pos_y, theta) and
        # the action (vel, thetadot).
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if thetadot_as_input:
            network_input = state_action[:, [2, 3, 4]]
            network_input_zero = torch.cat(
                (torch.zeros((state_action.shape[0], 1),dtype=torch.float64).to(device),
                 torch.zeros((state_action.shape[0], 1),dtype=torch.float64).to(device), 
                 torch.zeros((state_action.shape[0], 1),dtype=torch.float64).to(device)),
                dim=1)
            # network_input_zero = torch.cat(
            #     (state_action[:, 2].reshape((-1, 1)),
            #      torch.zeros((state_action.shape[0], 1),
            #                  dtype=torch.float64).to(device), state_action[:, 4].reshape(
            #                      (-1, 1))),
            #     dim=1)
        else:
            network_input = state_action[:, [2, 3]]
            network_input_zero = torch.cat(
                (state_action[:, 2].reshape((-1, 1)),
                 torch.zeros((state_action.shape[0], 1), dtype=torch.float64)).to(device),
                dim=1)
        return model(network_input) - model(network_input_zero.to(device))

    utils.train_approximator(training_dataset,
                             dynamics_relu,
                             model_output,
                             batch_size=200,
                             num_epochs=num_epochs,
                             lr=0.001,
                             save_dir=save_dir)


def train_controller_approximator(controller_relu, states, controls, lambda_u,
                                  Ru, num_epochs, lr):
    dataset = torch.utils.data.TensorDataset(states, controls)
    x_equilibrium = torch.zeros((3, ), dtype=torch.float64)
    u_equilibrium = torch.zeros((2, ), dtype=torch.float64)
    Ru.requires_grad_(True)

    def compute_u(model, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(x) - model(
            x_equilibrium.to(device)) + u_equilibrium.to(device) + lambda_u * torch.stack(
                (torch.norm(Ru @ x[:, :2].T, p=1, dim=0),
                 torch.zeros((x.shape[0], ), dtype=torch.float64))).T.to(device)

    utils.train_approximator(dataset,
                             controller_relu,
                             compute_u,
                             batch_size=30,
                             num_epochs=num_epochs,
                             lr=lr,
                             additional_variable=[Ru])
    Ru.requires_grad_(False)


def train_cost_approximator(lyapunov_relu, V_lambda, R, states, costs,
                            num_epochs, lr):
    dataset = torch.utils.data.TensorDataset(states, costs)
    x_equilibrium = torch.zeros((3, ), dtype=torch.float64)

    R.requires_grad_(True)

    def compute_v(model, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(x) - model(x_equilibrium.to(device)) + V_lambda * torch.norm(
            R @ (x - x_equilibrium).T, p=1, dim=0).reshape((-1, 1)).to(device)

    utils.train_approximator(dataset,
                             lyapunov_relu,
                             compute_v,
                             batch_size=50,
                             num_epochs=num_epochs,
                             lr=lr,
                             additional_variable=[R])
    R.requires_grad_(False)


if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description=
                                     "quadrotor 2d forward model training and lqr controller/lyapunov approximation.")
    parser.add_argument("--generate_dynamics_data",
                        default=None,#dir_path+'/../data/preprocess/dataset.pt',#None,
                        help="path to save dynamics data.")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=dir_path+'/../data/preprocess/dataset.pt',#dir_path+'/data/dynamics_data.pt',
                        help="path to the dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--train_lqr_approximator", action="store_true")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--training_set", type=str, default=None)      
    args = parser.parse_args()
    dtype = torch.float64
    dt = 0.01
    bound_level = 1
    bound_level_x = bound_level
    bound_level_y = bound_level
    bound_level_theta = bound_level
    
    # generate data and train for forward model
    if args.generate_dynamics_data:
        print("generate dynamics dataset")
        dynamics_dataset = generate_dynamics_data(dt)
        torch.save(dynamics_dataset,args.generate_dynamics_data)

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)

    thetadot_as_input = True
    if args.train_forward_model:
        print("train forward model")
        dynamics_relu = utils.setup_relu((3, 8, 8, 2),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        save_forward_dir = dir_path+'/../data/preprocess/forward_model_free.pt'
        train_forward_model(dynamics_relu,
                            dynamics_dataset,
                            num_epochs=100,
                            thetadot_as_input=thetadot_as_input,
                            save_dir=save_forward_dir)
        torch.save(dynamics_relu,save_forward_dir)

    
    V_lambda = 0.5
    
    x_lo = torch.tensor([-0.1*bound_level_x, -0.1*bound_level_y, -np.pi*bound_level_theta / 5], dtype=torch.float64)
    x_up = torch.tensor([0.1*bound_level_x, 0.1**bound_level_y, np.pi*bound_level_theta / 5], dtype=torch.float64)
    u_lo = torch.tensor([0, -0.15 * np.pi], dtype=torch.float64)
    u_up = torch.tensor([1, 0.15 * np.pi], dtype=torch.float64)
    
   
    controller_relu = utils.setup_relu((3, 15, 10, 5, 2),
                                       params=None,
                                       negative_slope=0.1,
                                       bias=True,
                                       dtype=torch.float64)
    controller_lambda_u = 4.
    controller_Ru = torch.tensor([[1, -1], [0, 1], [1, 0], [1, 1], [0.5, 0.9]],
                                 dtype=torch.float64)
    
    lyapunov_relu = utils.setup_relu((3, 15, 10, 5, 1),
                                     params=None,
                                     negative_slope=0.01,
                                     bias=True,
                                     dtype=torch.float64)
    R = torch.cat((torch.eye(3, dtype=torch.float64),
                   torch.tensor([[1, -1, 0], [-1, -1, 1], [0, 1, 1]],
                                dtype=torch.float64)),
                  dim=0)

    
    forward_system = unicycle.UnicycleReLUZeroVelModel(torch.float64, x_lo,
                                                       x_up, u_lo, u_up,
                                                       dynamics_relu, dt,
                                                       thetadot_as_input)
    # We only stabilize the horizontal position, not the orientation of the car
    Ru_options = r_options.SearchRwithSVDOptions(controller_Ru.shape,
                                                 np.array([0.1, 0.2]))
    
    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)
    if args.train_lqr_approximator:
        lqr_path = dir_path+'/../data/preprocess/'
        controller_relu_path = lqr_path + 'lqr_controller.pt'
        lyapunov_relu_path = lqr_path + 'lqr_lyapunov.pt'
        lyapunov_R_path = lqr_path + 'lqr_R.pt'
        print(os.path.exists(lqr_path))
        x_equilibrium = torch.cat(
            (q_equilibrium, torch.zeros((3, ), dtype=dtype)))
        # if os.path.exists(lqr_path):

        #     print("train lqr controller approximation")
        #     train_lqr_control_approximator(controller_relu, x_equilibrium,
        #                                 u_equilibrium, x_lo, x_up, 100000,
        #                                 torch.from_numpy(K))
        #     torch.save(controller_relu, controller_relu_path)
        #     print("save lqr controller approximation\n\n")
        
        if os.path.exists(lqr_path):
            print("train lqr lyapunov approximation")
            # lyapunov_relu = torch.load(lyapunov_relu_path)
            # R = torch.load(lyapunov_R_path)
            train_lqr_value_approximator(lyapunov_relu, V_lambda, R, x_equilibrium,
                                        x_lo, x_up, 100000, torch.from_numpy(S))
            torch.save(lyapunov_relu, lyapunov_relu_path)
            torch.save(R, lyapunov_R_path)
            print("save lqr lyapunov approximation")
