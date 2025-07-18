#%%
import sys  
sys.path.insert(0, '/usr3/graduate/zw2445/Lyapunov/neural-network-lyapunov')
from neural_network_lyapunov.examples.pendulum.visualizations.pendulum_visualization_utils import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.colors import ListedColormap
import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import random
#%%
dt = 0.01
bound_level = 9#  highest level 5 to give largest analysis region
roa_level = 3.
bound_level_x = bound_level
bound_level_y = bound_level
V_lambda = 0.8
add_l1_state = True#True
lyap_v_symm_flag = False

x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/pendulum/'#os.path.dirname(os.path.realpath(__file__))#+"/.."

load_lossSeq_prefix = "/data/monotonic_roa_0913/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
dir_path = os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "/data/pendulum_second_order_forward_relu2.pt"
load_lyapunov_relu = dir_path+ load_prefix+"_lyapunov.pt"
load_controller_relu = dir_path+load_prefix+"_controller.pt"
load_loss_history = dir_path+load_prefix+"_loss_history.npy"
load_R = dir_path+load_prefix+"_R.pt"
load_prefix = load_lossSeq_prefix +str(3)
load_L = dir_path+load_prefix+"_L.pt"

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

R = torch.load(load_R)
L = torch.ones(1,dtype=torch.float64)

resultList, l_inf_rect = get_levelSet_region(lyapunov_relu,lyapunov_upper=roa_level,add_l1_state=add_l1_state,R=R)
l_inf_bound_lo = torch.tensor([x_des[0]-l_inf_rect[0], x_des[1]-l_inf_rect[1]], dtype=torch.float64)
l_inf_bound_up = torch.tensor([x_des[0]+l_inf_rect[0], x_des[1]+l_inf_rect[1]], dtype=torch.float64)

forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(l_inf_bound_lo,l_inf_bound_up,dt,dynamics_model,controller_relu,lyapunov_relu)
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()

#%%
roa_level_tmp = 3.
final_l,f_c,x_sol = Linf_box_bisection_search(lyapunov_hybrid_system,forward_system,V_lambda,roa_level_tmp, R=R)
v = lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
v =  v + V_lambda * np.linalg.norm(R.detach().numpy() @ (x_sol[0] - x_des), ord=1) if add_l1_state else v
linf = final_l*np.linalg.norm(np.array(x_sol[0])-x_des,ord=np.inf)
print("roa_level: ",roa_level_tmp,
    "\nfinal_l: ",final_l, 
    "\nx_star: ",x_sol[0],
    "\nlyapunov at x_star: ",v,
    "\nl_inf with l_star at x_star: ",linf,
    "\nmilp optimal gap: ",f_c,
    "\nnumerical gap at x_star: ",v-linf)
bound_level_x = 10
bound_level_y = bound_level_x
x_lo_whole = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up_whole = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
N_x = 6

ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=x_sol[0])
#%%
spread_level = 0.1
spread_level_x = spread_level
spread_level_y = spread_level
x0s = []

x0s.append(np.array([np.pi + 0.1*spread_level_x, 0.1*spread_level_y]))
x0s.append(np.array([np.pi + 0.1*spread_level_x, -0.1*spread_level_y]))
x0s.append(np.array([np.pi - 0.1*spread_level_x, 0.1*spread_level_y]))
x0s.append(np.array([np.pi - 0.1*spread_level_x, -0.1*spread_level_y]))

spread_level = 10
spread_level_x = spread_level
spread_level_y = spread_level
x0s.append(np.array([np.pi + 0.1*spread_level_x, 0.1*spread_level_y]))
x0s.append(np.array([np.pi + 0.1*spread_level_x, -0.1*spread_level_y]))
x0s.append(np.array([np.pi - 0.1*spread_level_x, 0.1*spread_level_y]))
x0s.append(np.array([np.pi - 0.1*spread_level_x, -0.1*spread_level_y]))

x0s.append(np.array([np.pi - 0.5, 1.5]))
x0s.append(np.array([np.pi + 0.5, 1.5]))
x0s.append(np.array([np.pi + 1., 1.]))
x0s.append(np.array([np.pi, 0.]))
x_des = np.array([np.pi, 0])


def converged(t, y):
    dist = np.round(np.linalg.norm(y - x_des),3) - 1E-3
    # print(dist)
    return dist
converged.terminal = True

def pendulum_closed_loop_dynamics(plant: pendulum.Pendulum, x: np.ndarray,
                                  controller_relu, x_equilibrium,
                                  u_equilibrium, u_lo, u_up):
    assert (isinstance(plant, pendulum.Pendulum))
    u_pre_saturation = controller_relu(torch.from_numpy(x)) -\
        controller_relu(x_equilibrium) + u_equilibrium
    u = torch.max(torch.min(u_pre_saturation, u_up), u_lo).detach().numpy()
    # print(u,x)
    return plant.dynamics(x, u)

plant = pendulum.Pendulum(torch.float64)
lqr_gain = plant.lqr_control(np.diag([1., 10.]), np.array([[1.]]))

# Now train the controller and Lyapunov function together
q_equilibrium = torch.tensor([np.pi], dtype=torch.float64)
u_equilibrium = torch.tensor([0], dtype=torch.float64)
# x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
# x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
u_lo = torch.tensor([-20], dtype=torch.float64)
u_up = torch.tensor([20], dtype=torch.float64)


states = []
controls = []
next_states = []
vList = []
v_nextList = []
vDiffList = []

tspan = 600
for x0 in x0s:
    result = scipy.integrate.solve_ivp(
                lambda t, x: pendulum_closed_loop_dynamics(plant,x,controller_relu, torch.from_numpy(x_des),
                                  u_equilibrium, u_lo, u_up), (0, tspan), x0,
                t_eval=np.arange(0, tspan, dt),
                events=converged)
    states.append(result.y[:, :-1])
    controls.append(
        torch.from_numpy(lqr_gain @ (result.y[:, :-1] - x_des.reshape(
            (-1, 1)))))
    # print(controls[-1])
    next_states.append(result.y[:, 1:])
    lyap_val = lyap_func(torch.from_numpy(result.y).T)
    vList.append(lyap_val[:-1])
    v_nextList.append(lyap_val[1:])
    vDiffList.append(v_nextList[-1]+(0.001-1)*vList[-1])

plt.figure(figsize=(5, 11), dpi=80)
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.set_xlabel('$\Theta$')
ax1.set_ylabel('$\dot{\Theta}$')
ax2.set_xlabel('t')
ax2.set_ylabel('V')
# ax2.set_yscale('log')
ax3.set_xlabel('t')
ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])') 
for i in range(len(x0s)):
    # if states[i].shape[1]>1:
    ax1.plot(states[i][0],states[i][1])
    ax1.plot(states[i][0,0],states[i][1,0],'ro')
    ax1.plot(states[i][0,-1],states[i][1,-1],'gx')
    ax2.plot(np.linspace(0, len(vList[i])*dt,len(vList[i])),vList[i])
    ax3.plot(np.linspace(0, len(vDiffList[i])*dt,len(vDiffList[i])),vDiffList[i])
#%%
print(len(vList))
#%%
def compute_roa(all_points):
    """Compute the largest ROA as a set of states in a discretization."""

    # Forward-simulate all trajectories from initial points in the discretization
    states = []
    controls = []
    next_states = []
    vList = []
    v_nextList = []
    vDiffList = []

    tspan = 60
    for idx in range(all_points.shape[0]):
        x0 = all_points[idx]
        result = scipy.integrate.solve_ivp(
                    lambda t, x: pendulum_closed_loop_dynamics(plant,x,controller_relu, torch.from_numpy(x_des),
                                    u_equilibrium, u_lo, u_up), (0, tspan), x0,
                    t_eval=np.arange(0, tspan, dt),
                    events=converged)
        states.append(result.y[:, :-1])
        controls.append(
            torch.from_numpy(lqr_gain @ (result.y[:, :-1] - x_des.reshape(
                (-1, 1)))))
        # print(controls[-1])
        next_states.append(result.y[:, 1:])
        lyap_val = lyap_func(torch.from_numpy(result.y).T)
        vList.append(lyap_val[:-1])
        v_nextList.append(lyap_val[1:])
        vDiffList.append(v_nextList[-1]+(0.001-1)*vList[-1])
    return states,vList,vDiffList

# Number of states along each dimension
num_samples = [50,50]
x_lo = torch.tensor([np.pi - 0.1*10*np.pi, -0.5*10], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*10*np.pi, 0.5*10], dtype=torch.float64)
x_lo_np = x_lo.detach().numpy()
x_up_np = x_up.detach().numpy()
(pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
x_des = np.array([[np.pi, 0]])
states,vList,vDiffList = compute_roa(x_samples)
#%%
roa_map = np.zeros((x_samples.shape[0],))
for i in range(roa_map.shape[0]):
    roa_map[i] = np.linalg.norm(states[i][:,-1] - x_des)
roa_map = roa_map.reshape(num_samples)

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
# plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max*2, omega_max])))
plot_limits = np.column_stack((x_lo_np, x_up_np))

alpha = 1
colors = [None] * 4
colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
colors[1] = (230/255, 159/255, 0)       # NN  - orange
colors[2] = (0, 114/255, 178/255)       # LQR - blue
colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow


fig=plt.figure(figsize=(5, 11), dpi=80)
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
ax1.set_xlabel('$\Theta$')
ax1.set_ylabel('$\dot{\Theta}$')
ax2.set_xlabel('t')
ax2.set_ylabel('V')
# ax2.set_yscale('log')
ax3.set_xlabel('t')
ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])') 
idxList = random.sample(range(0, num_samples[0]*num_samples[1]), 10)


for idx in idxList:
    trajectories = states[idx]
    
    ax1.plot(trajectories[0,:],trajectories[1,:])
    ax1.plot(trajectories[0,0],trajectories[1,0],'ro')
    ax1.plot(trajectories[0,-1],trajectories[1,-1],'gx')
    ax2.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx])),vList[idx])
    ax3.plot(np.linspace(0, len(vDiffList[idx])*dt,len(vDiffList[idx])),vDiffList[idx])
z = roa_map.reshape(num_samples)
X = x_samples[:,0].reshape(num_samples)
Y =  x_samples[:,1].reshape(num_samples)
# print(z)
# ax1.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
# ax1.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)
ax1.contour(X, Y, z, origin='lower',  colors=(colors[0],), linewidths=1)
ax1.imshow(z.T, origin='lower', extent=plot_limits.ravel(),cmap=binary_cmap(colors[0]), alpha=alpha)

fig.tight_layout()
# def compute_roa(all_points, lyapunov_hybrid_system, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
#     """Compute the largest ROA as a set of states in a discretization."""

#     # Forward-simulate all trajectories from initial points in the discretization
#     if no_traj:
#         end_states = all_points
#         for t in range(1, horizon):
#             print(end_states.shape)
#             end_states = lyapunov_hybrid_system.system.step_forward(end_states)
#     else:
#         trajectories = np.empty((all_points.shape[0], 2, horizon))
#         lyaps = np.empty((all_points.shape[0], 1, horizon))
#         trajectories[:, :, 0] = all_points.detach().numpy()
#         for t in range(1, horizon):
#             # print(trajectories.shape,trajectories[:, :, t - 1].shape)
#             trajectories[:, :, t] = lyapunov_hybrid_system.system.step_forward(torch.tensor(trajectories[:, :, t - 1])).detach().numpy()
#             lyaps[:,0,t-1] = lyap_func(torch.tensor(trajectories[:, :, t - 1]))
#         end_states =torch.tensor(trajectories[:, :, -1])

#     if equilibrium is None:
#         equilibrium = np.zeros((1, 2))

#     # Compute an approximate ROA as all states that end up "close" to 0
#     dists = np.linalg.norm(end_states.detach().numpy() - equilibrium, ord=2, axis=1, keepdims=True).ravel()
#     roa = (dists <= tol)
#     if no_traj:
#         return roa
#     else:
#         return roa, trajectories,lyaps

# # Number of states along each dimension
# num_samples = [251,251]
# x_lo = torch.tensor([np.pi - 0.1*10*np.pi, -0.5*10], dtype=torch.float64)
# x_up = torch.tensor([np.pi + 0.1*10*np.pi, 0.5*10], dtype=torch.float64)
# x_lo_np = x_lo.detach().numpy()
# x_up_np = x_up.detach().numpy()
# (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
#                         np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
# x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
# x_des = np.array([[np.pi, 0]])
# roa,trajectories,lyaps = compute_roa(x_samples, lyapunov_hybrid_system, horizon=6000, tol=1e-1, equilibrium=x_des, no_traj=False)


# end_states =torch.tensor(trajectories[:, :, -1])
# dists = np.linalg.norm(end_states.detach().numpy() - x_des, ord=2, axis=1, keepdims=True).ravel()
# lyaps_end_states = lyaps[:,:,-1]
# lyaps_diff = lyaps[:,:,1:]+(0.001-1)*lyaps[:,:,:-1]
# print(dists)
# print(len(lyaps_diff[0][0]))
#%%

# theta_max = np.deg2rad(180)                     # angular position [rad]
# omega_max = np.deg2rad(360)
# def binary_cmap(color='red', alpha=1.):
#     """Construct a binary colormap."""
#     if color == 'red':
#         color_code = (1., 0., 0., alpha)
#     elif color == 'green':
#         color_code = (0., 1., 0., alpha)
#     elif color == 'blue':
#         color_code = (0., 0., 1., alpha)
#     else:
#         color_code = color
#     transparent_code = (1., 1., 1., 0.)
#     return ListedColormap([transparent_code, color_code])
# # plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max*2, omega_max])))
# plot_limits = np.column_stack((x_lo_np, x_up_np))

# alpha = 1
# colors = [None] * 4
# colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
# colors[1] = (230/255, 159/255, 0)       # NN  - orange
# colors[2] = (0, 114/255, 178/255)       # LQR - blue
# colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow


# fig=plt.figure(figsize=(5, 11), dpi=80)
# ax1 = plt.subplot(311)
# ax2 = plt.subplot(312)
# ax3 = plt.subplot(313)
# ax1.set_xlabel('$\Theta$')
# ax1.set_ylabel('$\dot{\Theta}$')
# ax2.set_xlabel('t')
# ax2.set_ylabel('V')
# # ax2.set_yscale('log')
# ax3.set_xlabel('t')
# ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])') 
# idxList = random.sample(range(0, 251*251), 10)
# roa = dists<0.6*1e-3
# z = roa.reshape(num_samples)
# X = x_samples[:,0].reshape(num_samples)
# Y =  x_samples[:,1].reshape(num_samples)
# # print(z)
# # ax1.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
# # ax1.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)
# ax1.contour(X, Y, z, origin='lower',  colors=(colors[0],), linewidths=1)
# ax1.imshow(z.T, origin='lower', extent=plot_limits.ravel(),cmap=binary_cmap(colors[0]), alpha=alpha)

# for idx in idxList:
#     ax1.plot(trajectories[idx,0,:],trajectories[idx,1,:])
#     ax1.plot(trajectories[idx,0,0],trajectories[idx,1,0],'ro')
#     ax1.plot(trajectories[idx,0,-1],trajectories[idx,1,-1],'gx')
#     ax2.plot(np.linspace(0, len(lyaps[idx][0])*dt,len(lyaps[idx][0])),lyaps[idx][0])
#     ax3.plot(np.linspace(0, len(lyaps_diff[idx][0])*dt,len(lyaps_diff[idx][0])),lyaps_diff[idx][0])
# fig.tight_layout()
#%%
# # Neural network
x_des = np.array([np.pi, 0])
x_lo_larger =  torch.tensor([0., -10.], dtype=torch.float64)
x_up_larger =  torch.tensor([np.pi + np.pi,10.], dtype=torch.float64)
x_sol,roa_level = find_roa(lyapunov_hybrid_system,forward_system,x_lo_larger, x_up_larger,V_lambda=V_lambda, R=R)
ax2 = plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level],mode='Trained')
ax2.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
ax2.plot(x_sol[0],x_sol[1],'bo')
#%%
# Plot some trajectories
N_traj = 11
skip = int(num_samples[0]/ N_traj)
nindex = np.prod(num_samples)
sub_idx = np.arange(nindex).reshape(num_samples)
sub_idx = sub_idx[::skip, ::skip].ravel()
sub_trajectories = trajectories[sub_idx, :, :]
sub_states = x_samples[sub_idx]
for n in range(sub_trajectories.shape[0]):
    x = sub_trajectories[n, 0, :] * np.rad2deg(x_up_np[0])
    y = sub_trajectories[n, 1, :] * np.rad2deg(x_up_np[1])
    ax.plot(x, y, 'k--', linewidth=0.25)

# dx_dt = (tf_future_states.eval({tf_states: sub_states}) - sub_states) / dt
# dx_dt = dx_dt / np.linalg.norm(dx_dt, ord=2, axis=1, keepdims=True)
# ax.quiver(sub_states[:, 0] * np.rad2deg(theta_max), sub_states[:, 1] * np.rad2deg(omega_max), dx_dt[:, 0], dx_dt[:, 1], 
#           scale=None, pivot='mid', headwidth=3, headlength=6, color='k')

ax.set_aspect(x_up_np[0] / x_up_np[1] / 1.2)
ax.set_xlim(plot_limits[0])
ax.set_ylim(plot_limits[1])
ax.set_xlabel(r'angle [deg]')
ax.set_ylabel(r'angular velocity [deg/s]')
ax.xaxis.set_ticks(np.arange(-180, 181, 60))
ax.yaxis.set_ticks(np.arange(-360, 361, 120))

proxy = [plt.Rectangle((0,0), 1, 1, fc=c) for c in colors]    
legend = ax.legend(proxy, [r'$\mathcal{S}_\pi$', r'NN', r'LQR', r'SOS'], loc='upper right')
legend.get_frame().set_alpha(1.)


# %%
