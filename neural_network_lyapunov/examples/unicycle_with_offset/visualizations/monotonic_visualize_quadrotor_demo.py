#%%
import sys  
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')

import neural_network_lyapunov.examples.quadrotor2d.quadrotor_2d as\
    quadrotor_2d
from neural_network_lyapunov.examples.quadrotor2d.visualizations.quadrotor2d_visualization_utils import *
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
v = monotonic_utils.generate_partition_space(9,dtype=torch.float64,size_in=6).detach().numpy()
print(v)
x_des = np.zeros((6, ))
ind = [0,1]
for v_i in range(v.shape[0]):
    plt.arrow(x=x_des[ind[0]], y=x_des[ind[1]], dx=v[v_i,ind[0]], dy=v[v_i,ind[1]], width=.01,head_width=0.1) 
plt.xlabel('x')
plt.ylabel('y')
#%%
dt = 0.01
bound_level = 1 # highest level 5 to give largest analysis region
add_l1_state = True

dtype = torch.float64
x_lo = torch.tensor([-0.1*bound_level, -0.1*bound_level,
                        -np.pi * 0.1*bound_level, -0.5*bound_level,
                        -0.5*bound_level, -0.3*bound_level],
                    dtype=dtype)
x_up = -x_lo
# x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "/../data/monotonic/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
dir_path = os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "/../data/second_order_forward_relu6.pt"
load_lyapunov_relu = dir_path+ load_prefix+"_lyapunov.pt"
load_controller_relu = dir_path+load_prefix+"_controller.pt"
load_loss_history = dir_path+load_prefix+"_loss_history.npy"
load_R = dir_path+load_prefix+"_R.pt"

dynamics_model_data = torch.load(load_dynamics_relu)
dynamics_model = utils.setup_relu(
    dynamics_model_data["linear_layer_width"],
    params=None,
    negative_slope=dynamics_model_data["negative_slope"],
    bias=True,
    dtype=torch.float64)
dynamics_model.load_state_dict(dynamics_model_data["state_dict"])
lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)
V_lambda = 0.6
R = torch.load(load_R)

forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=False)
lyapunov_hybrid_system.add_l1_state = add_l1_state
# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
def plot_loss_history(load_loss_history):
    loss_history_dict = np.load(load_loss_history,allow_pickle='TRUE').item()

    # loss_history_dict['loss_history'] = [x.item() for x in loss_history_dict['loss_history']]
    fig, ax = plt.subplots()
    loss_history_dict['total_training_time'] = round(loss_history_dict['total_training_time'],1)
    textstr = '\n'.join((
        r'$\mathrm{time}: %.1f \mathrm{ min}$' % (loss_history_dict['total_training_time']/60., ),
        r'$\mathrm{iterations}: %.f$' % (len(loss_history_dict['loss_history']), )))
    ax.plot(loss_history_dict['loss_history'][200:-1])

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
plot_loss_history(load_loss_history)
#%%
plot_loss_historySeq(dir_path+load_lossSeq_prefix,[2,4,6,8,10])
# plt.title("Bound Level "+str(bound_level)+" Converged in "+str(loss_history_dict['total_training_time'])+" s")
#%%
plot_traj(lyapunov_hybrid_system,lyap_func,x_lo,x_up,N_x=8,dt=0.01,display_violation_states=True)
#%%
lyapunov_relu_init = utils.setup_relu((2, 8, 8, 6, 1),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True,
                                    dtype=torch.float64)
num_samples = [80,80]
lyap_func_init = lambda x_next: lyapunov_relu_init(x_next).detach().numpy()
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80])
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80])

plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,mode='Initial')
plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,mode='Trained')
#%%

import gurobipy
x_lo_larger =  torch.tensor([0., -10.], dtype=torch.float64)
x_up_larger =  torch.tensor([np.pi + np.pi,10.], dtype=torch.float64)
rho = lyapunov_hybrid_system.compute_region_of_attraction(V_lambda, R,forward_system.x_equilibrium,
                                        None, x_lo_larger, x_up_larger)

print(rho)
milp1, _, _, _, _ = lyapunov_hybrid_system._construct_milp_for_roa(V_lambda, R, 
    forward_system.x_equilibrium, x_lo_larger, x_up_larger, True)
milp1.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
milp1.gurobi_model.optimize()

print(milp1.gurobi_model.ObjVal)

milp, x = lyapunov_hybrid_system._construct_milp_for_roa_boundary(
    V_lambda, R, forward_system.x_equilibrium)
milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
milp.gurobi_model.optimize()

x_sol = torch.tensor([v.x for v in x]).detach().numpy()
print(milp.gurobi_model.ObjVal,x_sol)
# %%
ax2 = plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[milp.gurobi_model.ObjVal],mode='Trained')
ax2.plot(x_sol[0],x_sol[1],'bo')
# %%
# %%
#%%
def compute_roa(all_points, lyapunov_hybrid_system, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""

    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states = all_points
        for t in range(1, horizon):
            print(end_states.shape)
            end_states = lyapunov_hybrid_system.system.step_forward(end_states)
    else:
        trajectories = np.empty((all_points.shape[0], 2, horizon))
        trajectories[:, :, 0] = all_points.detach().numpy()
        for t in range(1, horizon):
            # print(trajectories.shape,trajectories[:, :, t - 1].shape)
            trajectories[:, :, t] = lyapunov_hybrid_system.system.step_forward(torch.tensor(trajectories[:, :, t - 1])).detach().numpy()
        end_states =torch.tensor(trajectories[:, :, -1])

    if equilibrium is None:
        equilibrium = np.zeros((1, 2))

    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states.detach().numpy() - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories

# Number of states along each dimension
num_samples = [251,251]
x_lo_np = x_lo.detach().numpy()
x_up_np = x_up.detach().numpy()
(pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
x_des = np.array([[np.pi, 0]])
roa,trajectories = compute_roa(x_samples, lyapunov_hybrid_system, horizon=1500, tol=1e-1, equilibrium=x_des, no_traj=False)

#%%
from matplotlib.colors import ListedColormap
theta_max = np.deg2rad(180)                     # angular position [rad]
omega_max = np.deg2rad(360)
def binary_cmap(color='red', alpha=1.):
    """Construct a binary colormap."""
    if color == 'red':
        color_code = (1., 0., 0., alpha)
    elif color == 'green':
        color_code = (0., 1., 0., alpha)
    elif color == 'blue':
        color_code = (0., 0., 1., alpha)
    else:
        color_code = color
    transparent_code = (1., 1., 1., 0.)
    return ListedColormap([transparent_code, color_code])

fig = plt.figure(figsize=(5, 5), frameon=True)
plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max*2, omega_max])))


ax = plt.subplot(121)
alpha = 1
colors = [None] * 4
colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
colors[1] = (230/255, 159/255, 0)       # NN  - orange
colors[2] = (0, 114/255, 178/255)       # LQR - blue
colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow

# True ROA
z = roa.reshape(num_samples)
print(z)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)

# %%
