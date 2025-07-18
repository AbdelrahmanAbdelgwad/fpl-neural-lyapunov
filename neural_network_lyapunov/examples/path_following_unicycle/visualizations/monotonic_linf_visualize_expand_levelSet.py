#%%
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')
from neural_network_lyapunov.examples.path_following_unicycle.visualizations.unicycle_visualization_utils_linf import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.patches import Rectangle
import neural_network_lyapunov.monotonic_lyapunov.custom_lyapunov as lyapunov
import neural_network_lyapunov.monotonic_lyapunov.custom_train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils

import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip

dt = 0.01
bound_level = 10 # highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
bound_level_theta = bound_level
add_l1_state = True

dtype = torch.float64
bound_level_tmp = 40
x_lo = torch.tensor([-0.8*bound_level_tmp, -0.8*bound_level_tmp], dtype=torch.float64)/40.
x_up = torch.tensor([0.8*bound_level_tmp, 0.8*bound_level_tmp], dtype=torch.float64)/40.
# x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "data/monotonic_roa_0913/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/path_following_unicycle/'#os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "data/preprocess/path_following_unicycle_forward_model.pt"
load_lyapunov_relu = dir_path+ load_prefix+"_lyapunov.pt"
load_controller_relu = dir_path+load_prefix+"_controller.pt"

load_loss_history = dir_path+load_prefix+"_loss_history.npy"
load_R = dir_path+load_prefix+"_R.pt"
load_controller_Ru = dir_path+load_prefix+"_Ru.pt"

dynamics_model = torch.load(load_dynamics_relu, map_location=torch.device('cpu'))

lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)
L = torch.ones(1,dtype=torch.float64)
V_lambda = 0.6
R = torch.load(load_R)

def setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=True):
    
    dtype = torch.float64
    x_star = np.zeros((1,2))
    plant = path_following.Path_Following(torch.float64)
    q_equilibrium = torch.tensor([0.,0.], dtype=torch.float64)
    u_equilibrium = torch.tensor([plant.v], dtype=torch.float64)
    # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-10], dtype=torch.float64)
    u_up = torch.tensor([10], dtype=torch.float64)
    forward_system = relu_system.ReLUSystemGivenEquilibrium(
        torch.float64, x_lo, x_up, u_lo, u_up, dynamics_model, q_equilibrium,
        u_equilibrium, dt)
    closed_loop_system = feedback_system.FeedbackSystem(
        forward_system, controller_relu, forward_system.x_equilibrium,
        forward_system.u_equilibrium,
        u_lo.detach().numpy(),
        u_up.detach().numpy())

    if monotonic_flag:
        lyapunov_hybrid_system = custom_lyapunov.LyapunovDiscreteTimeHybridSystem(
            closed_loop_system, lyapunov_relu)
    else:
        lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            closed_loop_system, lyapunov_relu)
    # print('network value at eqlm: ',lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_star,dtype=torch.float64)))
    return forward_system,closed_loop_system,lyapunov_hybrid_system


forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=True)
lyapunov_hybrid_system.add_l1_state = add_l1_state
# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
x_des = np.zeros((2, ))
v_eqlm = lyap_func(torch.tensor(x_des,dtype=dtype))
print('Lyapunov at x_eqlm: ',v_eqlm)

#%%
plot_loss_history(load_loss_history)

#%%
# x_lo_whole = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
# x_up_whole = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
roa_level_tmp = 1.4
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

x_lo_whole = x_lo
x_up_whole = x_up
N_x = 6

ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=x_sol[0],report_flag=True)
ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=True,x_star=x_sol[0])


#%%
# roa_level = 1.4
# def solve_optimization(lyapunov_hybrid_system,x_equilibrium,V_lambda,roa_level,l,mip_pool_solutions=1,outputFlag=False,R=None):
#     milp_return = lyapunov_hybrid_system._construct_milp_for_roa_expand(x_equilibrium,
#                                                                     V_lambda,lyapunov_lower=0., 
#                                                                     lyapunov_upper=roa_level,
#                                                                     L = l, R=R)

#     milp = milp_return[0]
#     milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, outputFlag)
#     milp.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
#     milp.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions,mip_pool_solutions)
#     milp.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber,100)           
#     milp.gurobi_model.optimize()

#     # for solution_number in range(np.min((3, milp.gurobi_model.solCount))):
#     #         x_sol.append([v.xn for v in milp_return[1]])
#     x_sol = torch.tensor([v.x for v in milp_return[1]])
#     x_sol = []
#     x_sol_f = []
#     for solution_number in range(np.min((mip_pool_solutions,milp.gurobi_model.solCount))):
#         milp.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber, solution_number)
#         if  milp.gurobi_model.PoolObjVal >= -1000.:
#             x_sol.append(
#                 [v.xn for v in milp_return[1]])
#             x_sol_f.append(milp.gurobi_model.PoolObjVal)
#     return milp.gurobi_model.ObjVal, x_sol,x_sol_f

# l_tmp = final_l
# f_c, x_sol,x_sol_f= solve_optimization(lyapunov_hybrid_system,forward_system.x_equilibrium,V_lambda,roa_level,torch.ones_like(L)*l_tmp,
#                                R=R,mip_pool_solutions=50,outputFlag=True)
# print(x_sol,x_sol_f)
# linf = l_tmp*np.linalg.norm(np.array(x_sol[0])-x_des,ord=np.inf)
# x_star = torch.tensor(x_sol[0]).to(torch.float64)
# v_opt = lyapunov_relu.forward(x_star).to(torch.float64).detach().numpy()
# v_opt =  v_opt + V_lambda * np.linalg.norm(R.detach().numpy() @ (x_star.detach().numpy() - x_des), ord=1) if add_l1_state else v

# x_star_linf = roa_level/(torch.ones_like(L)*l_tmp)
# x_star = torch.tensor([x_des[0]+x_star_linf,x_des[1]-x_star_linf])
# v_star = lyapunov_relu.forward(x_star).to(torch.float64).detach().numpy()
# v_star =  v_star + V_lambda * np.linalg.norm(R.detach().numpy() @ (x_star.detach().numpy() - x_des), ord=1) if add_l1_state else v
# print(f_c,v_opt[0]-linf,v_opt,v_star)

# idx_max_norm = 0
# best_norm = np.linalg.norm(np.array(x_sol[0])-x_des)
# for i in range(1,len(x_sol)):
#     i_norm = np.linalg.norm(np.array(x_sol[i])-x_des)
#     # print(i_norm,idx_max_norm)
#     if i_norm>=best_norm:
#         best_norm = i_norm
#         idx_max_norm = i

#%%

# %%
def converged(t, y):
    dist = np.round(np.linalg.norm(y - x_des),3) - 1E-3
    # print(dist)
    return dist
converged.terminal = True


def compute_roa(all_points):
    """Compute the largest ROA as a set of states in a discretization."""
    def converged(t, y):
        dist = np.round(np.linalg.norm(y - x_des),3) - 1E-3
        # print(dist)
        return dist
    converged.terminal = True

    def pf_closed_loop_dynamics(plant: path_following.Path_Following, x: np.ndarray,
                                  controller_relu, x_equilibrium,
                                  u_equilibrium, u_lo, u_up):
        assert (isinstance(plant, path_following.Path_Following))
        u_pre_saturation = controller_relu(torch.from_numpy(x)) -\
            controller_relu(x_equilibrium) + u_equilibrium
        u = torch.max(torch.min(u_pre_saturation, u_up), u_lo).detach().numpy()
        # print(u,x)
        return plant.dynamics(x, u)

    plant = path_following.Path_Following(torch.float64)
    lqr_Q = np.diag([1,1])
    lqr_R = np.array([[1]])
    lqr_gain, S = plant.lqr_control(lqr_Q, lqr_R)
    S_eig_value, S_eig_vec = np.linalg.eig(S)

    # Now train the controller and Lyapunov function together
    q_equilibrium = torch.tensor([0.,0.], dtype=torch.float64)
    u_equilibrium = torch.tensor([plant.v], dtype=torch.float64)
    # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-10], dtype=torch.float64)
    u_up = torch.tensor([10], dtype=torch.float64)
    
    
    # Forward-simulate all trajectories from initial points in the discretization
    states = []
    controls = []
    next_states = []
    vList = []
    v_nextList = []
    vDiffList = []

    tspan = 300
    for idx in range(all_points.shape[0]):
        print('idx: ',idx)
        x0 = all_points[idx]
        result = scipy.integrate.solve_ivp(
                    lambda t, x: pf_closed_loop_dynamics(plant,x,controller_relu, torch.from_numpy(x_des),
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


#%%
# Number of states along each dimension
num_samples = [80,80]
x_lo_np = x_lo.detach().numpy()
x_up_np = x_up.detach().numpy()
(pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
v_pred = v_pred = lyapunov_value(x_samples,torch.tensor(x_des),lyapunov_relu,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
v_pred = v_pred.detach().numpy()
idxList = random.sample(range(0, num_samples[0]*num_samples[1]), 20)
x_samples_selected = x_samples[idxList,:]
states,vList,vDiffList = compute_roa(x_samples_selected)


#%%
from matplotlib.colors import ListedColormap
roa_level = [roa_level_tmp]
roa_map = v_pred<=roa_level_tmp
roa_map = roa_map.reshape(num_samples)
Z = v_pred.reshape(num_samples[0], num_samples[1])

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

alpha = 1
colors = [None] * 4
colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
colors[1] = (230/255, 159/255, 0)       # NN  - orange
colors[2] = (0, 114/255, 178/255)       # LQR - blue
colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow


fig, ax1 = plt.subplots(constrained_layout=True)

for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]
    
    ax1.plot(trajectories[0,:],trajectories[1,:])
    ax1.plot(trajectories[0,0],trajectories[1,0],'go')
    ax1.plot(trajectories[0,-1],trajectories[1,-1],'rx')
    # ax2.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx])),vList[idx])
    # ax3.plot(np.linspace(0, len(vDiffList[idx])*dt,len(vDiffList[idx])),vDiffList[idx])
z = roa_map.reshape(num_samples)
X = x_samples[:,0].reshape(num_samples)
Y =  x_samples[:,1].reshape(num_samples)
# print(z)


contour_linewidth = 2
contour_font = 14
if roa_level is not None:
    CS5 = ax1.contour(X, Y, Z, [roa_level],
                colors=('r',),
                linewidths=(2,))
    ax1.clabel(CS5, fmt='%2.1f', colors='k', fontsize=14)
contour_linewidth = 1
contour_font = 10
# CS3 = ax1.contourf(X, Y, Z, 
#             origin='lower',
#             extend='both',
#             alpha=0.3)
CS4 = ax1.contour(X, Y, Z, 
                colors=('k',),
                linewidths=(contour_linewidth,),
                origin='lower',
                alpha = 0.2)
ax1.clabel(CS4, fmt='%2.1f', colors='k', fontsize=contour_font)
# Notice that the colorbar gets all the information it
# needs from the ContourSet object, CS3.
# fig.colorbar(CS3)
ax1.set_xlim([x_lo_np[0],x_up_np[0]])
ax1.set_ylim([x_lo_np[1],x_up_np[1]])
# plt.title('Contour Plot of Trained Lyapunov Function and Trajectories')
plt.xlabel('${x_e}$')
plt.ylabel('${\Theta_e}$')
# %%
fig=plt.figure(figsize=(5, 5), dpi=80)

plt.xlabel('t')
plt.ylabel('V(t)') 
for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]
    plt.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx])),vList[idx])
    
# %%
