import neural_network_lyapunov.examples.third_order_strict.path_following as path_following

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
    Generate the pairs (x[n], u[n]) -> (x[n+1])
    """
    dtype = torch.float64
    plant = path_following.Third_Order_Strict(dtype)

    x_des = np.array([0, 0,0])

    def converged(t, y):
        return np.linalg.norm(y - x_des) - 1E-2

    converged.terminal = True
    lqr_Q = np.diag([2,1,2])
    lqr_R = np.array([[1]])
    lqr_gain, S = plant.lqr_control(lqr_Q, lqr_R)
    # x1_range = [-1.6,1.6]
    # x2_range = [-1.6,1.6]
    # x3_range = [-2.1,2.1]
    # u_range = [-10,10]
    
    x1_range = [-1.65,1.65]
    x2_range = [-1.65,1.65]
    x3_range = [-2.15,2.15]
    u_range = [-30,30]
    
    # We don't need to take the grid on y and z dimension of the quadrotor,
    # since the dynamics is invariant along these dimensions.
    x_samples =utils.uniform_sample_in_box(
            torch.tensor([x1_range[0], x2_range[0], x3_range[0]],dtype=torch.float64),
            torch.tensor([x1_range[1], x2_range[1], x3_range[1]],dtype=torch.float64), 1000).T
    u_samples = utils.uniform_sample_in_box(
        torch.full((1, ), u_range[0], dtype=dtype),
        torch.full((1, ), u_range[1], dtype=dtype), 1000).T

    xu_tensors = []
    x_next_tensors = []
    
    n_x = 30
    x0s = np.zeros((3,n_x))
    x0s[0,:] = np.random.rand(n_x)*(x1_range[1]-x1_range[0])+x1_range[0]
    x0s[1,:] = np.random.rand(n_x)*(x2_range[1]-x2_range[0])+x2_range[0]
    x0s[2,:] = np.random.rand(n_x)*(x3_range[1]-x3_range[0])+x3_range[0]
    
    for x0 in x0s.T:
        result = scipy.integrate.solve_ivp(
            lambda t, x: plant.dynamics(x, lqr_gain @ (x - x_des)), (0, 10),
            x0/3.,
            t_eval=np.arange(0, 10, dt),
            events=converged)
        
        xu_tensors.append(
            torch.cat((torch.from_numpy(result.y[:, :-1]), 
                       torch.from_numpy(lqr_gain @ (result.y[:, :-1] - x_des.reshape((-1, 1))))
                       )).T)
        x_next_tensors.append(
            torch.from_numpy(result.y[:, 1:]).T)

    for i in range(x_samples.shape[1]):
        print(i)
        for j in range(u_samples.shape[1]):
            result = scipy.integrate.solve_ivp(
                lambda t, x: plant.dynamics(x, u_samples[:, j].detach().numpy(
                )), (0, dt), x_samples[:, i].detach().numpy())
            xu_tensors.append(
                torch.cat((x_samples[:, i], u_samples[:, j])).reshape((1, -1)))
            x_next_tensors.append(
                torch.from_numpy(result.y[:, -1]).reshape((1, -1)))
    dataset_input = torch.cat(xu_tensors, dim=0)
    dataset_output = torch.cat(x_next_tensors, dim=0)
    
    return torch.utils.data.TensorDataset(dataset_input, dataset_output)


def train_forward_model(dynamics_model, model_dataset,num_epochs=20,save_dir=None):
    plant = path_following.Third_Order_Strict(dtype)
    state_equilibrium = torch.tensor([0., 0., 0.], dtype=torch.float64)
    control_equilibrium = torch.tensor([0.], dtype=torch.float64)
    # model_dataset contains the mapping from (x[n], u[n]) to x[n+1], but we
    # only need a mapping from (x[n], u[n]) to v[n+1]. So we regenerate a
    # dataset whose target only contains thetadot.
    (xu_inputs, x_next_outputs) = model_dataset[:]
    v_dataset = torch.utils.data.TensorDataset(
        xu_inputs, x_next_outputs)

    
    def compute_next_v(model, state_action):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return model(state_action) - model(
            torch.cat((state_equilibrium.to(device), control_equilibrium.to(device))))

    utils.train_approximator(v_dataset,
                             dynamics_model,
                             compute_next_v,
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
                        default=dir_path+'/../data/preprocess/dataset2.pt',#None,
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
    args.train_forward_model = True
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
        dynamics_relu = utils.setup_relu((4, 6, 5, 3),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        save_forward_dir = dir_path+'/../data/preprocess/third_order_forward_model2.pt'
        dynamics_relu = torch.load(dir_path+'/../data/preprocess/third_order_forward_model.pt')
        train_forward_model(dynamics_relu,
                            dynamics_dataset,
                            num_epochs=100,
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

    
    # Now train the controller and Lyapunov function together
    q_equilibrium = torch.tensor([np.pi], dtype=torch.float64)
    u_equilibrium = torch.tensor([0], dtype=torch.float64)
    # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-20], dtype=torch.float64)
    u_up = torch.tensor([20], dtype=torch.float64)
    forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
        torch.float64, x_lo, x_up, u_lo, u_up, dynamics_model, q_equilibrium,
        u_equilibrium, dt)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())
    lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
        closed_loop_system, lyapunov_relu)
    
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
