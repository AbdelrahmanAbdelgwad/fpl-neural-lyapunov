# %%
import sys

sys.path.insert(
    0,
    "/home/abdelrahman/projects/Neural_Lyapunov_Control/neural-network-lyap-control-roa/",
)
from neural_network_lyapunov.examples.pendulum.visualizations.pendulum_visualization_utils import *
import neural_network_lyapunov.utils as utils

dt = 0.01
bound_level = 10  # highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
add_l1_state = True

x_lo = torch.tensor(
    [np.pi - 0.1 * bound_level_x * np.pi, -0.5 * bound_level_y], dtype=torch.float64
)
x_up = torch.tensor(
    [np.pi + 0.1 * bound_level_x * np.pi, 0.5 * bound_level_y], dtype=torch.float64
)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "data/tedrake/tedrake_bound"
load_prefix = load_lossSeq_prefix + str(bound_level)
# dir_path = os.path.dirname(os.path.realpath(__file__))#+"/.."
dir_path = "/home/abdelrahman/projects/Neural_Lyapunov_Control/neural-network-lyap-control-roa/neural_network_lyapunov/examples/pendulum/"  # os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "/data/pendulum_second_order_forward_relu2.pt"
print(load_dynamics_relu)
load_lyapunov_relu = dir_path + load_prefix + "_lyapunov.pt"
load_controller_relu = dir_path + load_prefix + "_controller.pt"
load_loss_history = dir_path + load_prefix + "_loss_history.npy"
load_R = dir_path + load_prefix + "_R.pt"

dynamics_model_data = torch.load(load_dynamics_relu)
dynamics_model = utils.setup_relu(
    dynamics_model_data["linear_layer_width"],
    params=None,
    negative_slope=dynamics_model_data["negative_slope"],
    bias=True,
    dtype=torch.float64,
)
dynamics_model.load_state_dict(dynamics_model_data["state_dict"])
lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)
V_lambda = 0.6
R = torch.load(load_R)

forward_system, closed_loop_system, lyapunov_hybrid_system = setup_closed_loop_system(
    x_lo, x_up, dt, dynamics_model, controller_relu, lyapunov_relu, monotonic_flag=False
)
lyapunov_hybrid_system.add_l1_state = add_l1_state


# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = (
    lambda x_next: lyapunov_hybrid_system.lyapunov_value(
        x_next, forward_system.x_equilibrium, V_lambda=V_lambda, R=R
    )
    .detach()
    .numpy()
)
plot_loss_history(load_loss_history)
# %%
plot_loss_historySeq(dir_path + load_lossSeq_prefix, [2, 4, 6, 8, 10])
# plt.title("Bound Level "+str(bound_level)+" Converged in "+str(loss_history_dict['total_training_time'])+" s")
# %%
plot_traj(
    lyapunov_hybrid_system,
    lyap_func,
    x_lo,
    x_up,
    N_x=8,
    dt=0.01,
    display_violation_states=True,
)
# %%
lyapunov_relu_init = utils.setup_relu(
    (2, 8, 8, 6, 1), params=None, negative_slope=0.1, bias=True, dtype=torch.float64
)
num_samples = [80, 80]
lyap_func_init = lambda x_next: lyapunov_relu_init(x_next).detach().numpy()
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80])
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80])

plot_lyapunov_contour(
    x_lo,
    x_up,
    x_des,
    monotonic_NN=lyapunov_relu_init,
    num_samples=[80, 80],
    V_lambda=V_lambda,
    add_l1_state=add_l1_state,
    R=R,
    display_v=False,
    mode="Initial",
)
plot_lyapunov_contour(
    x_lo,
    x_up,
    x_des,
    monotonic_NN=lyapunov_relu,
    num_samples=[80, 80],
    V_lambda=V_lambda,
    add_l1_state=add_l1_state,
    R=R,
    display_v=False,
    mode="Trained",
)
# %%

import gurobipy

x_lo_larger = torch.tensor([0.0, -10.0], dtype=torch.float64)
x_up_larger = torch.tensor([np.pi + np.pi, 10.0], dtype=torch.float64)


def converged(t, y):
    dist = np.round(np.linalg.norm(y - x_des), 3) - 1e-3
    # print(dist)
    return dist


converged.terminal = True


def compute_roa(all_points):
    """Compute the largest ROA as a set of states in a discretization."""

    def converged(t, y):
        dist = np.round(np.linalg.norm(y - x_des), 3) - 1e-3
        # print(dist)
        return dist

    converged.terminal = True

    def pendulum_closed_loop_dynamics(
        plant: pendulum.Pendulum,
        x: np.ndarray,
        controller_relu,
        x_equilibrium,
        u_equilibrium,
        u_lo,
        u_up,
    ):
        assert isinstance(plant, pendulum.Pendulum)
        u_pre_saturation = (
            controller_relu(torch.from_numpy(x))
            - controller_relu(x_equilibrium)
            + u_equilibrium
        )
        u = torch.max(torch.min(u_pre_saturation, u_up), u_lo).detach().numpy()
        # print(u,x)
        return plant.dynamics(x, u)

    plant = pendulum.Pendulum(torch.float64)
    lqr_gain, lqr_s = plant.lqr_control(np.diag([1.0, 10.0]), np.array([[1.0]]))

    # Now train the controller and Lyapunov function together
    q_equilibrium = torch.tensor([np.pi], dtype=torch.float64)
    u_equilibrium = torch.tensor([0], dtype=torch.float64)
    # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-20], dtype=torch.float64)
    u_up = torch.tensor([20], dtype=torch.float64)

    # Forward-simulate all trajectories from initial points in the discretization
    states = []
    controls = []
    next_states = []
    vList = []
    v_nextList = []
    vDiffList = []

    tspan = 600
    for idx in range(all_points.shape[0]):
        print("idx: ", idx)
        x0 = all_points[idx]
        result = scipy.integrate.solve_ivp(
            lambda t, x: pendulum_closed_loop_dynamics(
                plant,
                x,
                controller_relu,
                torch.from_numpy(x_des),
                u_equilibrium,
                u_lo,
                u_up,
            ),
            (0, tspan),
            x0,
            t_eval=np.arange(0, tspan, dt),
            events=converged,
        )
        states.append(result.y[:, :-1])
        controls.append(
            torch.from_numpy(lqr_gain @ (result.y[:, :-1] - x_des.reshape((-1, 1))))
        )
        # print(controls[-1])
        next_states.append(result.y[:, 1:])
        lyap_val = lyap_func(torch.from_numpy(result.y).T)
        vList.append(lyap_val[:-1])
        v_nextList.append(lyap_val[1:])
        vDiffList.append(v_nextList[-1] + (0.001 - 1) * vList[-1])
    return states, vList, vDiffList


# Number of states along each dimension
num_samples = [80, 80]
x_lo_np = x_lo.detach().numpy()
x_up_np = x_up.detach().numpy()
(pos_mesh, vel_mesh) = np.meshgrid(
    np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
    np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]),
)
x_samples = torch.tensor(
    np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))
).t()
# v_pred = lyapunov_value(x_samples,torch.tensor(x_des),lyapunov_relu,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
# v_pred = v_pred.detach().numpy()
v_pred = lyapunov_value(
    x_samples,
    torch.tensor(x_des),
    lyapunov_relu,
    V_lambda=V_lambda,
    add_l1_state=add_l1_state,
    R=R,
)
v_pred = v_pred.detach().numpy()
random.seed(10)
idxList = random.sample(range(0, num_samples[0] * num_samples[1]), 20)
x_samples_selected = x_samples[idxList, :]
states, vList, vDiffList = compute_roa(x_samples_selected)


# %%
from matplotlib.colors import ListedColormap


def get_boundary_levelset(
    x_lo_larger, x_up_larger, lyapunov_hybrid_system, forward_system, V_lambda, R
):
    milp, x = lyapunov_hybrid_system._construct_milp_for_roa_boundary(
        V_lambda, R, forward_system.x_equilibrium
    )
    milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
    milp.gurobi_model.optimize()
    x_sol = torch.tensor([v.x for v in x]).detach().numpy()

    return milp.gurobi_model.ObjVal, x_sol


roa_level, x_sol = get_boundary_levelset(
    x_lo_larger, x_up_larger, lyapunov_hybrid_system, forward_system, V_lambda, R
)
roa_map = v_pred <= roa_level
roa_map = roa_map.reshape(num_samples)
Z = v_pred.reshape(num_samples[0], num_samples[1])


def binary_cmap(color="red", alpha=1.0):
    """Construct a binary colormap."""
    if color == "red":
        color_code = (1.0, 0.0, 0.0, alpha)
    elif color == "green":
        color_code = (0.0, 1.0, 0.0, alpha)
    elif color == "blue":
        color_code = (0.0, 0.0, 1.0, alpha)
    else:
        color_code = color
    transparent_code = (1.0, 1.0, 1.0, 0.0)
    return ListedColormap([transparent_code, color_code])


# plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max*2, omega_max])))

alpha = 1
colors = [None] * 4
colors[0] = (0, 158 / 255, 115 / 255)  # ROA - bluish-green
colors[1] = (230 / 255, 159 / 255, 0)  # NN  - orange
colors[2] = (0, 114 / 255, 178 / 255)  # LQR - blue
colors[3] = (240 / 255, 228 / 255, 66 / 255)  # SOS - yellow


fig, ax1 = plt.subplots(constrained_layout=True)

for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]

    ax1.plot(trajectories[0, :], trajectories[1, :])
    ax1.plot(trajectories[0, 0], trajectories[1, 0], "go")
    ax1.plot(trajectories[0, -1], trajectories[1, -1], "rx")
    # ax2.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx])),vList[idx])
    # ax3.plot(np.linspace(0, len(vDiffList[idx])*dt,len(vDiffList[idx])),vDiffList[idx])
z = roa_map.reshape(num_samples)
X = x_samples[:, 0].reshape(num_samples)
Y = x_samples[:, 1].reshape(num_samples)
# print(z)


contour_linewidth = 2
contour_font = 14
if roa_level is not None:
    CS5 = ax1.contour(X, Y, Z, [roa_level], colors=("r",), linewidths=(2,))
    ax1.clabel(CS5, fmt="%2.1f", colors="k", fontsize=14)
contour_linewidth = 1
contour_font = 10
# CS3 = ax1.contourf(X, Y, Z,
#             origin='lower',
#             extend='both',
#             alpha=0.3)
CS4 = ax1.contour(
    X, Y, Z, colors=("k",), linewidths=(contour_linewidth,), origin="lower", alpha=0.2
)
ax1.clabel(CS4, fmt="%2.1f", colors="k", fontsize=contour_font)
# Notice that the colorbar gets all the information it
# needs from the ContourSet object, CS3.
# fig.colorbar(CS3)
ax1.set_xlim([x_lo_np[0], x_up_np[0]])
ax1.set_ylim([x_lo_np[1], x_up_np[1]])
# plt.title('Contour Plot of Trained Lyapunov Function and Trajectories')
plt.xlabel("${\Theta}$")
plt.ylabel("${\dot{\Theta}}$")
plt.rcParams.update({"font.size": 16})
plt.tight_layout()
plt.savefig(
    "/home/zw2445/Documents/neural-network-lyapunov/plots/report/pendulum_tedrake_bound"
    + str(bound_level)
    + "_traj.png",
    dpi=300,
)
# plot lyapunov value
fig = plt.figure(dpi=300)
plt.xlabel("t")
plt.ylabel("V(t)")
for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]
    plt.plot(np.linspace(0, len(vList[idx]) * dt, len(vList[idx])), vList[idx])
plt.rcParams.update({"font.size": 16})
plt.tight_layout()
plt.savefig(
    "/home/zw2445/Documents/neural-network-lyapunov/plots/report/pendulum_tedrake_bound"
    + str(bound_level)
    + "_lyap.png",
    dpi=300,
)
# %%
