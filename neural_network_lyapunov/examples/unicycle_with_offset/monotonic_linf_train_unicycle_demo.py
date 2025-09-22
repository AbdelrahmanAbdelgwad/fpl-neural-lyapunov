import neural_network_lyapunov.examples.car.unicycle as unicycle
import neural_network_lyapunov.examples.car.unicycle_feedback_system as\
    unicycle_feedback_system
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.monotonic_lyapunov.custom_lyapunov as lyapunov
import neural_network_lyapunov.monotonic_lyapunov.custom_train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils

import neural_network_lyapunov.r_options as r_options
import argparse
import torch
import numpy as np
import os


def save_controller_model(controller_relu, x_lo, x_up, u_lo, u_up, lambda_u,
                          Ru_options, file_path):
    linear_layer_width, negative_slope, bias = utils.extract_relu_structure(
        controller_relu)
    saved_params = {
        "linear_layer_width": linear_layer_width,
        "state_dict": controller_relu.state_dict(),
        "negative_slope": negative_slope,
        "x_lo": x_lo,
        "x_up": x_up,
        "u_lo": u_lo,
        "u_up": u_up,
        "bias": bias,
        "lambda_u": lambda_u,
        "Ru": Ru_options.R(),
        "fixed_R": Ru_options.fixed_R,
    }
    Ru_params = Ru_options.extract_params()
    saved_params.update(Ru_params)
    torch.save(saved_params, file_path)


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


def train_forward_model(dynamics_relu, dataset, num_epochs, thetadot_as_input):
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
        if thetadot_as_input:
            network_input = state_action[:, [2, 3, 4]]
            network_input_zero = torch.cat(
                (state_action[:, 2].reshape((-1, 1)),
                 torch.zeros((state_action.shape[0], 1),
                             dtype=torch.float64), state_action[:, 4].reshape(
                                 (-1, 1))),
                dim=1)
        else:
            network_input = state_action[:, [2, 3]]
            network_input_zero = torch.cat(
                (state_action[:, 2].reshape((-1, 1)),
                 torch.zeros((state_action.shape[0], 1), dtype=torch.float64)),
                dim=1)
        return model(network_input) - model(network_input_zero)

    utils.train_approximator(training_dataset,
                             dynamics_relu,
                             model_output,
                             batch_size=30,
                             num_epochs=num_epochs,
                             lr=0.001)


def train_controller_approximator(controller_relu, states, controls, lambda_u,
                                  Ru, num_epochs, lr):
    dataset = torch.utils.data.TensorDataset(states, controls)
    x_equilibrium = torch.zeros((3, ), dtype=torch.float64)
    u_equilibrium = torch.zeros((2, ), dtype=torch.float64)
    Ru.requires_grad_(True)

    def compute_u(model, x):
        return model(x) - model(
            x_equilibrium) + u_equilibrium + lambda_u * torch.stack(
                (torch.norm(Ru @ x[:, :2].T, p=1, dim=0),
                 torch.zeros((x.shape[0], ), dtype=torch.float64))).T

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
        return model(x) - model(x_equilibrium) + V_lambda * torch.norm(
            R @ (x - x_equilibrium).T, p=1, dim=0).reshape((-1, 1))

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
    
    parser = argparse.ArgumentParser(description="Unicycle training demo")
    parser.add_argument("--generate_dynamics_data", action="store_true")
    parser.add_argument("--load_dynamics_data",
                        type=str,
                        default=None,
                        help="path to load the dynamics data.")
    parser.add_argument("--train_forward_model", action="store_true")
    parser.add_argument("--load_forward_model",
                        type=str,
                        default=None,
                        help="path to the forward model network.")
    parser.add_argument("--load_controller_relu",
                        type=str,
                        default=None,
                        help="path to load the controller relu.")
    parser.add_argument("--load_lyapunov_relu",
                        type=str,
                        default=None,
                        help="path to load lyapunov relu")
    parser.add_argument("--search_R", action="store_true")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=2000,
        help="max number of iterations in training the controller.")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--train_on_samples", action="store_true")
    parser.add_argument("--pretrain_num_epochs", type=int, default=20)
    parser.add_argument(
        "--train_adversarial",
        action="store_true",
        help="only do adversarial training, not bilevel optimization")
    parser.add_argument("--dp_states", type=str, default=None)
    parser.add_argument("--dp_costs", type=str, default=None)
    parser.add_argument("--dp_actions", type=str, default=None)
    parser.add_argument(
        "--traj_opt_data",
        type=str,
        default=None,
        help="path to the trajectory optimization data from many different" +
        " initial states. Will train the controller/Lyapunov to" +
        " approximate these data.")
    parser.add_argument(
        "--rrt_star_data",
        type=str,
        default=None,
        help="path to the rrt* data. Will train the controller/Lyapunov" +
        " for approximation")
    parser.add_argument("--bound_level",
                        type=int,default=1,help="bound level of x.")
    parser.add_argument("--bound_level_last",
                        type=int,default=0,help="bound level of x from pre-trained controller and lyapunov.")       
    args = parser.parse_args()
    args.search_R = True
    search_controllerFalg = True#True
    if not search_controllerFalg:
        args.load_controller_relu = dir_path+"/data/monotonic/monotonic_bound10_controller.pt"
    add_l1_stateFalg = False#True
    lyap_v_symm_flag = False
    bound_level = args.bound_level
    bound_level_x = bound_level
    bound_level_y = bound_level
    bound_level_theta = bound_level
    print("bound level is: ", bound_level)
    
    R = torch.cat((torch.eye(3, dtype=torch.float64),
                   torch.tensor([[1, -1, 0], [-1, -1, 1], [0, 1, 1]],
                                dtype=torch.float64)),
                  dim=0)
    
    if args.search_R:
        R_options = r_options.SearchRwithSVDOptions(R.shape,
                                                    np.array([0.5, 0.5, 0.5]))
        R_options.set_variable_value(R.detach().numpy())
    else:
        R_options = r_options.FixedROptions(R)
        
    if args.bound_level_last>=1:
        bound_level_last = args.bound_level_last#bound_level-1
        if search_controllerFalg:
            args.load_controller_relu=dir_path+"/data/monotonic_roa/monotonic_bound"+str(bound_level_last)+"_controller.pt"
        if args.search_R:
            args.load_R=dir_path+"/data/monotonic_roa/monotonic_bound"+str(bound_level_last)+"_R.pt"
            R = torch.load(args.load_R)
            R_options.set_variable_value(R.detach().numpy())
        args.load_lyapunov_relu=dir_path+"/data/monotonic_roa/monotonic_bound"+str(bound_level_last)+"_lyapunov.pt"
        print("pre-trained bound level is: ", bound_level_last)
   
    print("pretrained lyapunov path: ",args.load_lyapunov_relu)
    # x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
    x_lo = torch.tensor([-0.1*bound_level_x, -0.1*bound_level_y, -np.pi*bound_level_theta / 5], dtype=torch.float64)
    x_up = torch.tensor([0.1*bound_level_x, 0.1**bound_level_y, np.pi*bound_level_theta / 5], dtype=torch.float64)
    print("input bound: ",x_lo,x_up)
    dt = 0.01
    if args.generate_dynamics_data:
        dynamics_dataset = generate_dynamics_data(dt)

    if args.load_dynamics_data:
        dynamics_dataset = torch.load(args.load_dynamics_data)

    thetadot_as_input = True
    if args.train_forward_model:
        dynamics_relu = utils.setup_relu((3, 8, 8, 2),
                                         params=None,
                                         negative_slope=0.1,
                                         bias=True,
                                         dtype=torch.float64)
        train_forward_model(dynamics_relu,
                            dynamics_dataset,
                            num_epochs=100,
                            thetadot_as_input=thetadot_as_input)
    else:#if args.load_forward_model:
        dynamics_model_data = torch.load(
            dir_path + "/data/unicycle_forward_relu8.pt")
        dynamics_relu = utils.setup_relu(
            dynamics_model_data["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["negative_slope"],
            bias=dynamics_model_data["bias"],
            dtype=torch.float64)
        dynamics_relu.load_state_dict(dynamics_model_data["state_dict"])

    V_lambda = 0.5
    controller_relu = utils.setup_relu((3, 15, 10, 5, 2),
                                       params=None,
                                       negative_slope=0.1,
                                       bias=True,
                                       dtype=torch.float64)
    controller_lambda_u = 4.
    controller_Ru = torch.tensor([[1, -1], [0, 1], [1, 0], [1, 1], [0.5, 0.9]],
                                 dtype=torch.float64)
    if args.load_controller_relu:
        # controller_data = torch.load(args.load_controller_relu)
        # controller_relu = utils.setup_relu(
        #     controller_data["linear_layer_width"],
        #     params=None,
        #     negative_slope=controller_data["negative_slope"],
        #     bias=controller_data["bias"],
        #     dtype=torch.float64)
        # controller_relu.load_state_dict(controller_data["state_dict"])
        # controller_lambda_u = controller_data["lambda_u"]
        # controller_Ru = controller_data["Ru"]
        controller_relu = torch.load(args.load_controller_relu)

    u_lo = torch.tensor([0, -0.15 * np.pi], dtype=torch.float64)
    u_up = torch.tensor([1, 0.15 * np.pi], dtype=torch.float64)

    forward_system = unicycle.UnicycleReLUZeroVelModel(torch.float64, x_lo,
                                                       x_up, u_lo, u_up,
                                                       dynamics_relu, dt,
                                                       thetadot_as_input)
    # We only stabilize the horizontal position, not the orientation of the car
    Ru_options = r_options.SearchRwithSVDOptions(controller_Ru.shape,
                                                 np.array([0.1, 0.2]))
    Ru_options.set_variable_value(controller_Ru.detach().numpy())
    if args.search_R:
        Ru_options = r_options.SearchRfreeOptions(controller_Ru.shape)
        Ru_options.set_variable_value(controller_Ru.detach().numpy())
    closed_loop_system = unicycle_feedback_system.UnicycleFeedbackSystem(
        forward_system, controller_relu,
        u_lo.detach().numpy(),
        u_up.detach().numpy(), controller_lambda_u, Ru_options)
    
    # lyapunov_relu = utils.setup_relu((3, 15, 10, 5, 1),
    #                                  params=None,
    #                                  negative_slope=0.01,
    #                                  bias=True,
    #                                  dtype=torch.float64)
    lyapunov_relu = monotonic_utils.setup_monotonic_relu(size_out=1,
                    size_in=forward_system.x_equilibrium.numel(),
                    epsilon=0.05,
                    size_partition=5,
                    size_piecewise=4,
                    params=None,
                    dtype=torch.float64,
                    x_eqlm=forward_system.x_equilibrium,
                    symm_flag=lyap_v_symm_flag)
        


    if args.load_lyapunov_relu:
        # lyapunov_data = torch.load(args.load_lyapunov_relu)
        # lyapunov_relu = utils.setup_relu(
        #     lyapunov_data["linear_layer_width"],
        #     params=None,
        #     negative_slope=lyapunov_data["negative_slope"],
        #     bias=lyapunov_data["bias"],
        #     dtype=torch.float64)
        # lyapunov_relu.load_state_dict(lyapunov_data["state_dict"])
        # V_lambda = lyapunov_data["V_lambda"]
        # R = lyapunov_data["R"]
        lyapunov_relu = torch.load(args.load_lyapunov_relu)

    if args.dp_states:
        dp_states = torch.from_numpy(np.load(args.dp_states)).T
        dp_costs = torch.from_numpy(np.load(args.dp_costs)).reshape((-1, 1))
        dp_actions = torch.from_numpy(np.load(args.dp_actions))
        train_controller_approximator(controller_relu,
                                      dp_states,
                                      dp_actions,
                                      controller_lambda_u,
                                      controller_Ru,
                                      num_epochs=300,
                                      lr=0.001)
        train_cost_approximator(lyapunov_relu,
                                V_lambda,
                                R,
                                dp_states,
                                dp_costs,
                                num_epochs=1000,
                                lr=0.001)

    if args.traj_opt_data:
        traj_opt_data = torch.load(args.traj_opt_data)
        traj_opt_states = torch.cat(traj_opt_data["states"], dim=0)
        traj_opt_controls = torch.cat(traj_opt_data["controls"], dim=0)
        traj_opt_costs = torch.cat(traj_opt_data["costs"], dim=0)
        train_controller_approximator(controller_relu,
                                      traj_opt_states,
                                      traj_opt_controls,
                                      controller_lambda_u,
                                      controller_Ru,
                                      num_epochs=50,
                                      lr=0.001)
        train_cost_approximator(lyapunov_relu,
                                V_lambda,
                                R,
                                traj_opt_states,
                                traj_opt_costs,
                                num_epochs=100,
                                lr=0.001)
    if args.rrt_star_data:
        rrt_star_data = torch.load(args.rrt_star_data)
        rrt_star_states = torch.from_numpy(rrt_star_data["node_state"])
        # The root state has control 0.
        rrt_star_controls = [np.zeros((2, ))]
        for i in range(1, len(rrt_star_data["node_to_parent_u"])):
            rrt_star_controls.append(rrt_star_data["node_to_parent_u"][i][:,
                                                                          0])
        rrt_star_controls = torch.from_numpy(np.vstack(rrt_star_controls))
        rrt_star_costs = torch.from_numpy(rrt_star_data["node_cost_to_root"])
        train_controller_approximator(controller_relu,
                                      rrt_star_states,
                                      rrt_star_controls,
                                      num_epochs=50,
                                      lr=0.001)
        train_cost_approximator(lyapunov_relu,
                                V_lambda,
                                R,
                                rrt_star_states,
                                rrt_star_costs,
                                num_epochs=100,
                                lr=0.001)
    

    lyap = lyapunov.LyapunovDiscreteTimeHybridSystem(closed_loop_system,
                                                     lyapunov_relu)


    if args.enable_wandb:
        train_utils.wandb_config_update(args, lyapunov_relu, controller_relu,
                                        x_lo, x_up, u_lo, u_up)

    dut = train_lyapunov_barrier.Trainer()
    dut.add_lyapunov(lyap, V_lambda, closed_loop_system.x_equilibrium,
                     R_options)
    dut.lyapunov_positivity_mip_pool_solutions = 1
    dut.lyapunov_derivative_mip_pool_solutions = 1
    dut.lyapunov_derivative_convergence_tol = 1E-6
    dut.max_iterations = args.max_iterations
    dut.lyapunov_positivity_epsilon = 0.1
    dut.lyapunov_derivative_epsilon = 0.002
    # Only want to stabilize the horizontal position of the car, not the
    # orientation.
    state_samples_all = utils.get_meshgrid_samples(x_lo,
                                                   x_up, (31, 31, 31),
                                                   dtype=torch.float64)
    dut.output_flag = True
    dut.search_controller = search_controllerFalg
    if args.train_on_samples:
        dut.train_lyapunov_on_samples(state_samples_all,
                                      num_epochs=args.pretrain_num_epochs,
                                      batch_size=50)
    dut.enable_wandb = args.enable_wandb
    dut.save_network_path = dir_path + "/data/monotonic_roa/monotonic_bound" + str(bound_level) + "_"
    
    if args.train_adversarial:
        dut.save_network_path += "adversarial_"
        options = train_lyapunov_barrier.Trainer.AdversarialTrainingOptions()
        options.positivity_samples_pool_size = 10000
        options.derivative_samples_pool_size = 10000
        options.num_epochs_per_mip = 10
        dut.add_derivative_adversarial_state = True
        dut.add_positivity_adversarial_state = True
        dut.lyapunov_positivity_mip_pool_solutions = 100
        dut.lyapunov_derivative_mip_pool_solutions = 200
        positivity_state_samples_init = utils.get_meshgrid_samples(
            x_lo, x_up, (20, 20, 20), dtype=torch.float64)
        derivative_state_samples_init = positivity_state_samples_init
        result = dut.train_adversarial(positivity_state_samples_init,
                                       derivative_state_samples_init, options)
    else:
        dut.lyapunov_positivity_mip_cost_weight = None
        dut.boundary_value_gap_mip_cost_weight = 0.0
        # dut.train(torch.empty((0, 3), dtype=torch.float64))
               
        dut.linf_expand_mip_cost_weight = None#0.001
        dut.linf_margin_mip_cost_weight = None #0.01
        dut.optimizer = "Adam"#"custom_optimizer"#"Adam""AdaBound"
        if dut.optimizer != "custom_optimizer":
            dut.linf_expand_mip_cost_weight = None
        # dut.learning_rate = 1e-4
        dut.optimzer_scheduling = False
        for i in range(2,7):
            dut.lyapunov_upper = 0.5*i#1.#None
            print("ROA value: ",dut.lyapunov_upper)
            dut.train(torch.empty((0, 3), dtype=torch.float64))
        
        
        ### recursive training
        # num_of_trials = 6
        # for j in range(num_of_trials):
        #     dut.max_iterations = args.max_iterations
        #     dut.linf_expand_mip_cost_weight = None#0.001
        #     dut.linf_margin_mip_cost_weight = None #0.01
        #     dut.optimizer = "Adam"#"custom_optimizer"#"Adam""AdaBound"
        #     if dut.optimizer != "custom_optimizer":
        #         dut.linf_expand_mip_cost_weight = None
        #     # dut.learning_rate = 1e-4
        #     dut.optimzer_scheduling = False
        #     for i in range(5,7):
        #         dut.lyapunov_upper = 0.5*i#1.#None
        #         print("ROA value: ",dut.lyapunov_upper)
        #         dut.train(torch.empty((0, 2), dtype=torch.float64))
                
        #     if j<num_of_trials-1:
        #         dut.max_iterations = 10
        #         dut.lyapunov_upper = 3.0
        #         dut.linf_expand_mip_cost_weight = 0.001#0.001
        #         dut.linf_margin_mip_cost_weight = None
        #         dut.optimizer = "custom_optimizer"#"custom_optimizer"#"Adam"
        #         if dut.optimizer != "custom_optimizer":
        #             dut.linf_expand_mip_cost_weight = 0.
        #         # dut.learning_rate = 5*1e-4
        #         dut.optimzer_scheduling = False
        #         dut.train_lineSearch(torch.empty((0, 2), dtype=torch.float64))
                # dut.train(torch.empty((0, 2), dtype=torch.float64))
        
        
        # dut.lyapunov_upper = 3.0
        # dut.linf_expand_mip_cost_weight = 0.001#0.001
        # dut.linf_margin_mip_cost_weight = None
        # dut.optimizer = "custom_optimizer"#"custom_optimizer"#"Adam"
        # if dut.optimizer != "custom_optimizer":
        #     dut.linf_expand_mip_cost_weight = 0.
        # # dut.learning_rate = 5*1e-4
        # dut.optimzer_scheduling = False
        # dut.train(torch.empty((0, 2), dtype=torch.float64))
        
        
        # dut.lyapunov_upper = 3.0
        # dut.linf_expand_mip_cost_weight = 0.001#0.001
        # dut.linf_margin_mip_cost_weight = None
        # # dut.learning_rate = 5*1e-4
        # dut.optimzer_scheduling = False
        # dut.train_lineSearch(torch.empty((0, 2), dtype=torch.float64))
        
        # expand_mip_cost_weight = None#0.001
        # dut.optimizer = "Adam"#"custom_optimizer"#"Adam""AdaBound"
        # dut.linf_margin_mip_cost_weight = 0.01
        # if dut.optimizer != "custom_optimizer":
        #     dut.linf_expand_mip_cost_weight = None
        # # dut.learning_rate = 1e-4
        # dut.optimzer_scheduling = False
        # dut.lyapunov_upper = 2.#1.#None
        # print("ROA value: ",dut.lyapunov_upper)
        # dut.train(torch.empty((0, 2), dtype=torch.float64))
    pass
