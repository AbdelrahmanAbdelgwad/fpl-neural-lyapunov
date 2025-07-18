#%%
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')
from neural_network_lyapunov.examples.pendulum.visualizations.monotonic_init_pendulum_visualization_utils import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.patches import Rectangle
dt = 0.01
bound_level = 10#  highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
V_lambda = 0.8
add_l1_state = True#True
lyap_v_symm_flag = False

x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "/data/monotonic_V08/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
# dir_path = os.path.dirname(os.path.realpath(__file__))#+"/.."
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/pendulum/'#os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "/data/pendulum_second_order_forward_relu2.pt"
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
lyapunov_relu_loaded = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)
# for param_tensor in lyapunov_relu_loaded[2].state_dict():
#     print(param_tensor, "\t", lyapunov_relu_loaded[2].state_dict()[param_tensor].size())
# print(lyapunov_relu_loaded[0].__dir__(),lyapunov_relu_loaded[0].load_state_dict)
lyapunov_relu = monotonic_utils.setup_monotonic_relu(size_out=lyapunov_relu_loaded[0].size_out,
                size_in=lyapunov_relu_loaded[0].size_in,
                epsilon=0.01,
                size_partition=5,
                size_piecewise=4,
                params=None,
                dtype=torch.float64,
                x_eqlm=torch.tensor(x_des))
for i in [0,2]:
    lyapunov_relu[i].F_1 = lyapunov_relu_loaded[i].F_1
    lyapunov_relu[i].F_2 = lyapunov_relu_loaded[i].F_2
    lyapunov_relu[i].F_3 = lyapunov_relu_loaded[i].F_3
    lyapunov_relu[i].G = lyapunov_relu_loaded[i].G
    lyapunov_relu[i].v = lyapunov_relu_loaded[i].v
lyapunov_relu[0].b_input = lyapunov_relu_loaded[0].b_input
lyapunov_relu[2].a_input = lyapunov_relu_loaded[2].a_input
R = torch.load(load_R)

forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu)
lyapunov_hybrid_system.add_l1_state = add_l1_state
# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
plot_loss_history(load_loss_history)
#%%
plot_loss_historySeq(dir_path+load_lossSeq_prefix,[2,4,6,8,10])
# plt.title("Bound Level "+str(bound_level)+" Converged in "+str(loss_history_dict['total_training_time'])+" s")
#%%
def get_boundary_levelset(x_lo_larger,x_up_larger,lyapunov_hybrid_system,forward_system,V_lambda, R):
    milp, x = lyapunov_hybrid_system._construct_milp_for_roa_boundary(
        V_lambda, R, forward_system.x_equilibrium)
    milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
    milp.gurobi_model.optimize()
    x_sol = torch.tensor([v.x for v in x]).detach().numpy()

    return milp.gurobi_model.ObjVal,x_sol

x_lo_larger =  torch.tensor([0., -5.], dtype=torch.float64)
x_up_larger =  torch.tensor([np.pi + np.pi,5.], dtype=torch.float64)
milp_obj,x_sol=get_boundary_levelset(x_lo_larger,x_up_larger,lyapunov_hybrid_system,forward_system,V_lambda, R)
print("level set value at boundary: ", milp_obj, " at point: ",x_sol)
print("Lyapunov NN directions ",lyapunov_relu[0].v)
#%%
def plot_levelSet_Linf_region(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,roa_level_at_bound=None,
                         mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,report_flag=False,fig_name=None,roa_color='r'):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
    v_pred = lyapunov_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,0].reshape(num_samples[0], num_samples[1])
    Y =  x_samples[:,1].reshape(num_samples[0], num_samples[1])
    Z = v_pred.reshape(num_samples[0], num_samples[1])
    
    origin = 'lower'
    # cmap = "viridis"#pltcm.get_cmap("viridis").copy()

    fig2, ax2 = plt.subplots(constrained_layout=True,figsize=(3.54,3.54), dpi=600)
    CS3 = ax2.contourf(X, Y, Z, 
                    origin=origin,
                    extend='both',vmin=0,alpha=0.4,cmap=plt.cm.coolwarm)
    # Our data range extends outside the range of levels; make
    # data below the lowest contour level yellow, and above the
    # highest level cyan:
    # CS3.cmap.set_under('yellow')
    # CS3.cmap.set_over('cyan')

    contour_linewidth = 2
    contour_font = 18 
    # if x_star is not None:
    #     v = monotonic_NN.forward(torch.tensor(x_star).to(torch.float64)).detach().numpy()
    #     linf = l*np.linalg.norm(np.array(x_star)-x_des,ord=np.inf)
    

        # contour_font = 10
    # CS4 = ax2.contour(X, Y, Z, 
    #                 colors=('k',),
    #                 linewidths=(1,),
    #                 origin=origin,alpha=0.5)
    # ax2.clabel(CS4, fmt='%2.1f', colors='b', fontsize=contour_font,alpha=0.5)
    
    plt.rcParams.update({'font.size': 15})
    if roa_level is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=(roa_color,),
                    linewidths=(3,), label='$V^{-1}(r)$')
        ax2.clabel(CS5, fmt='%2.1f', colors='k', fontsize=contour_font)
        contour_linewidth = 2
        # contour_font = 10
    if roa_level_at_bound is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level_at_bound,
                    colors=('g',),
                    linewidths=(3,))
        ax2.clabel(CS5, fmt='%2.1f', colors='k', fontsize=contour_font)
    # Notice that the colorbar gets all the information it
    # needs from the ContourSet object, CS3.
    if not report_flag:
        fig2.colorbar(CS3)
    if display_v:
        v = monotonic_NN[0].v.detach().numpy()
        scale = min((x_up_np[0]-x_lo_np[0])/2.,(x_up_np[1]-x_lo_np[1])/2.)-0.5
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        for v_i in range(v.shape[0]):
            ax2.text(x_des[0]+v[v_i,0]*scale,x_des[1]+v[v_i,1]*scale, str(v_i), fontsize=contour_font,
                bbox=props)
            plt.arrow(x=x_des[0], y=x_des[1], dx=v[v_i,0]*scale, dy=v[v_i,1]*scale, width=.01,head_width=0.1) 
        if add_l1_state:
            props = dict(boxstyle='round', facecolor='yellow', alpha=0.5)
            R_np = R.detach().numpy()
            for v_i in range(R_np.shape[0]):
                v_norm = np.linalg.norm(R_np[v_i,:])
                ax2.text(x_des[0]+R_np[v_i,0]*scale/v_norm,x_des[1]+R_np[v_i,1]*scale/v_norm, str(v_i+v.shape[0]), fontsize=contour_font,
                    bbox=props)
                plt.arrow(x=x_des[0], y=x_des[1], dx=R_np[v_i,0]*scale/v_norm, dy=R_np[v_i,1]*scale/v_norm, width=.01,head_width=0.1) 

    if not report_flag:
        plt.legend(loc="upper right")
        plt.title(mode + ' NN Optimized on Level Set $V^{-1}$(r)')
    ax2.set_xlim([x_lo[0],x_up[0]])
    ax2.set_ylim([x_lo[1],x_up[1]])   
    ax2.xaxis.set_ticks(np.arange(x_lo[0],x_up[0]+0.1, 2))
    ax2.yaxis.set_ticks(np.arange(x_lo[1],x_up[1]+0.1, 2))
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/'+fig_name, bbox_inches='tight')
    return ax2

bound_level_x = 10
bound_level_y = bound_level_x
x_lo_whole = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up_whole = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
N_x = 6
if bound_level <= 2:
    milp_obj_tmp = None
    roa_color = 'b'
else:
    milp_obj_tmp = [milp_obj]
    roa_color = 'r'
ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[milp_obj],
    roa_level_at_bound=None,mode='Trained',report_flag=True, roa_color=roa_color,plot_setBox=False,fig_name='pendulum_bound'+str(bound_level)+'_contour.pdf')

#%% find region of attraction of trained model

def converged(t, y):
    dist = np.round(np.linalg.norm(y - x_des),3) - 1E-3
    # print(dist)
    return dist
converged.terminal = True


def compute_roa(all_points):
    """Compute the largest ROA as a set of states in a discretization."""
    def converged(t, y):
        dist = np.round(np.linalg.norm(y - x_des),3) - 1E-2
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
    lqr_gain, lqr_s = plant.lqr_control(np.diag([1., 10.]), np.array([[1.]]))

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

    tspan = 400
    for idx in range(all_points.shape[0]):
        print('idx: ',idx)
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
num_samples = [80,80]
x_lo_np = x_lo_whole.detach().numpy()
x_up_np = x_up_whole.detach().numpy()
(pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
# v_pred = lyapunov_value(x_samples,torch.tensor(x_des),lyapunov_relu,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
# v_pred = v_pred.detach().numpy()
v_pred = lyapunov_value(x_samples,torch.tensor(x_des),lyapunov_relu,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
v_pred = v_pred.detach().numpy()
random.seed(10)
idxList = random.sample(range(0, num_samples[0]*num_samples[1]), 20)
x_samples_selected = x_samples[idxList,:]
states,vList,vDiffList = compute_roa(x_samples_selected)


#%%
import matplotlib
from matplotlib.colors import ListedColormap
import scienceplots
roa_level = [milp_obj]
roa_map = v_pred<=milp_obj
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

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fig, ax1 = plt.subplots(constrained_layout=True,figsize=(3.5*1.4, 2.7*1.4), dpi=600)
ax1.autoscale(tight=True)
for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]
    
    ax1.plot(trajectories[0,:],trajectories[1,:])
    ax1.plot(trajectories[0,0],trajectories[1,0],'go')
    ax1.plot(trajectories[0,-1],trajectories[1,-1],'ro',markersize=10)
    # ax2.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx])),vList[idx])
    # ax3.plot(np.linspace(0, len(vDiffList[idx])*dt,len(vDiffList[idx])),vDiffList[idx])
z = roa_map.reshape(num_samples)
X = x_samples[:,0].reshape(num_samples)
Y =  x_samples[:,1].reshape(num_samples)
# print(z)


contour_linewidth = 2
contour_font = 15
if roa_level is not None:
    CS5 = ax1.contour(X, Y, Z, [roa_level],
                colors=('r',),
                linewidths=(2,))
    ax1.clabel(CS5, fmt='%2.1f', colors='k', fontsize=14)
    # CS5 = ax1.contour(X, Y, Z, [milp_obj_tmp],
    #             colors=('g',),
    #             linewidths=(2,))
    # ax1.clabel(CS5, fmt='%2.1f', colors='k', fontsize=14)
    
contour_linewidth = 1
contour_font = 15
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
ax1.set_xlim([x_lo[0],x_up[0]])
ax1.set_ylim([x_lo[1],x_up[1]])   
ax1.xaxis.set_ticks(np.arange(x_lo[0],x_up[0]+0.1, 2))
ax1.yaxis.set_ticks(np.arange(x_lo[1],x_up[1]+0.1, 2))
# plt.title('Contour Plot of Trained Lyapunov Function and Trajectories')
plt.xlabel('${\Theta}$')
plt.ylabel('${\dot{\Theta}}$')
plt.rcParams.update({'font.size': 18})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/pendulum_bound'+str(bound_level)+'_traj.pdf', bbox_inches='tight')
# plot lyapunov value
#%%
fig, ax1 = plt.subplots(constrained_layout=True,figsize=(3.5*2, 2.625*2), dpi=600)
ax1.autoscale(tight=True)
plt.xlabel('t')
plt.ylabel('V(t)') 
for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]
    plt.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx]))[0:6000],vList[idx][0:6000])
plt.rcParams.update({'font.size': 15})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/pendulum_bound'+str(bound_level)+'_lyap.pdf', bbox_inches='tight')
    
# %%
