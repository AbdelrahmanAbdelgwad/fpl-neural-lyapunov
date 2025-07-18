#%%
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')
from neural_network_lyapunov.examples.pendulum.visualizations.pendulum_visualization_utils import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.patches import Rectangle
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
dt = 0.01
bound_level = 25#  highest level 5 to give largest analysis region
roa_level = 3.
bound_level_x = 10
bound_level_y = 10
V_lambda = 0.6
add_l1_state = True#True
lyap_v_symm_flag = False

x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "/data/monotonic_roa_0913/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
# dir_path = os.path.dirname(os.path.realpath(__file__))#+"/.."
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/pendulum/'#os.path.dirname(os.path.realpath(__file__))#+"/.."

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

resultList, l_inf_rect = get_levelSet_region(lyapunov_relu,lyapunov_upper=roa_level,add_l1_state=add_l1_state,R=R,V_lambda=V_lambda)
l_inf_bound_lo = torch.tensor([x_des[0]-l_inf_rect[0], x_des[1]-l_inf_rect[1]], dtype=torch.float64)
l_inf_bound_up = torch.tensor([x_des[0]+l_inf_rect[0], x_des[1]+l_inf_rect[1]], dtype=torch.float64)

forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(l_inf_bound_lo,l_inf_bound_up,dt,dynamics_model,controller_relu,lyapunov_relu)
    
# lyapunov_relu = monotonic_utils.setup_monotonic_relu(size_out=1,
#                 size_in=forward_system.x_equilibrium.numel(),
#                 epsilon=0.05,
#                 size_partition=7,
#                 size_piecewise=3,
#                 params=None,
#                 dtype=torch.float64,
#                 x_eqlm=forward_system.x_equilibrium,
#                 symm_flag=False)
# forward_system,closed_loop_system,lyapunov_hybrid_system=\
#     setup_closed_loop_system(l_inf_bound_lo,l_inf_bound_up,dt,dynamics_model,controller_relu,lyapunov_relu)
    
lyapunov_hybrid_system.add_l1_state = add_l1_state
# lyapunov_hybrid_system.add_levelSet_linf_L(L)
# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()

print(R)
#%%
def get_boundary_levelset(x_lo_larger,x_up_larger,lyapunov_hybrid_system,forward_system,V_lambda, R):
    milp, x = lyapunov_hybrid_system._construct_milp_for_roa_boundary(
        V_lambda, R, forward_system.x_equilibrium,x_lo_larger.numpy(),x_up_larger.numpy())
    milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
    milp.gurobi_model.optimize()
    x_sol = torch.tensor([v.x for v in x]).detach().numpy()

    return milp.gurobi_model.ObjVal,x_sol

x_lo_larger =  torch.tensor([0., -5.], dtype=torch.float64)
x_up_larger =  torch.tensor([np.pi + np.pi,5.], dtype=torch.float64)
milp_obj,x_sol=get_boundary_levelset(x_lo_larger,x_up_larger,lyapunov_hybrid_system,forward_system,V_lambda, R)
print("level set value at boundary: ", milp_obj, " at point: ",x_sol)
print("Lyapunov NN directions ",lyapunov_relu[0].v)

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
#%%
plot_loss_history(load_loss_history)

#%%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# x_lo_whole = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
# x_up_whole = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
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

    fig2, ax2 = plt.subplots(constrained_layout=True,figsize=(3.5*1.5, 2.7*1.5), dpi=600)
    ax2.autoscale(tight=True)
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
    
    plt.rcParams.update({'font.size': 20})
    if roa_level is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=(roa_color,),
                    linewidths=(3,), label='$V^{-1}(r)$')
        manual_locations = [(1.2,-3.2)]
        ax2.clabel(CS5, fmt='%2.1f', colors='k', fontsize=contour_font)#, manual=manual_locations)
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
    if plot_setBox:
        resultList, l_inf_rect = get_levelSet_region(monotonic_NN,lyapunov_upper=roa_level[0],add_l1_state=add_l1_state,R=R,V_lambda=V_lambda)
        v = monotonic_NN[0].v.detach().numpy()
        for i in range(len(resultList)): 
            transformed_vector = x_des + resultList[i]#*v[i,:]
            if i==0:
                plt.plot(transformed_vector[0],transformed_vector[1],'ro', label='$m_i^{-1}(r)$')
            else:
                plt.plot(transformed_vector[0],transformed_vector[1],'ro')
            # print(v[i,:],resultList[i],transformed_vector)
        print(l_inf_rect)   
        rect_bottom_left_corner = (x_des[0]-l_inf_rect[0],
                                   x_des[1]-l_inf_rect[1])    
        ax2.add_patch(Rectangle(rect_bottom_left_corner, 
                                l_inf_rect[0]*2,l_inf_rect[1]*2,
                                fc ='g', 
                                alpha=0.3,
                                ec ='g',
                                lw = 5, label = 'set box'))
    linf_bottom_left_corner = (x_des[0]-l,
                                x_des[1]-l)  
    ax2.add_patch(Rectangle(linf_bottom_left_corner, 
                            2.*l,2.*l,
                            alpha=0.5,
                            ec ='w',
                            lw = 3, label = 'L_inf box'))
    if x_star is not None:
        plt.plot(x_star[0],x_star[1],'rD', label='$x^*$',markersize=10)
    if x_star_list is not None:
        for i in range(len(x_star_list)):
            plt.plot(x_star_list[i][0],x_star_list[i][1],'rD',markersize=10)
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
        plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/'+fig_name, bbox_inches='tight',dpi=600)
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
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],
    roa_level_at_bound=milp_obj_tmp,mode='Trained',report_flag=True, roa_color=roa_color,
    l=linf/final_l,plot_setBox=False,x_star=x_sol[0],fig_name='pendulum_linf_bound'+str(bound_level)+'_contour.pdf')
# ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
#     V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],
#     roa_level_at_bound=None,mode='Trained',report_flag=True,
#     l=linf/final_l,plot_setBox=True,x_star=x_sol[0],fig_name=None)


# %%
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
contour_font = 16
if roa_level is not None:
    CS5 = ax1.contour(X, Y, Z, [roa_level],
                colors=('r',),
                linewidths=(2,))
    manual_locations = [(1.2,-3.2),(5.2,4)]
    ax1.clabel(CS5, fmt='%2.1f', colors='k', fontsize=14,manual=manual_locations)
    CS5 = ax1.contour(X, Y, Z, [milp_obj_tmp],
                colors=('g',),
                linewidths=(2,))
    # ax1.clabel(CS5, fmt='%2.1f', colors='k', fontsize=14)
    
contour_linewidth = 1
contour_font = 12
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
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/pendulum_linf_bound'+str(bound_level)+'_traj.pdf', bbox_inches='tight')
# plot lyapunov value
#%%
fig, ax1 = plt.subplots(constrained_layout=True,figsize=(3.5*1.5, 2.7*1.5), dpi=600)
ax1.autoscale(tight=True)
plt.xlabel('t')
plt.ylabel('V(t)') 
for idx in range(x_samples_selected.shape[0]):
    trajectories = states[idx]
    plt.plot(np.linspace(0, len(vList[idx])*dt,len(vList[idx]))[0:6000],vList[idx][0:6000])
plt.rcParams.update({'font.size': 20})
ax1.set_xlim([-2,62])
ax1.xaxis.set_ticks([0,20,40,60])
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/pendulum_linf_bound'+str(bound_level)+'_lyap.pdf', bbox_inches='tight')
    
#%%
def plot_lqr_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy() 
    x_dim = len(x_lo_np)
    xList = np.zeros((x_dim,N_x))
    np.random.seed(0)
    np.random.seed(0)
    for i in range(x_dim):
        xList[i,:] = np.random.rand(N_x)*(x_up_np[i]-x_lo_np[i])+x_lo_np[i]
    # for i in range(N_x):
    #     x_tmp = xList[:,i]
    #     # print(torch.tensor(x_tmp,dtype=torch.float64).shape)
    #     if lyap_func(torch.tensor(x_tmp,dtype=torch.float64))[0]>0.0032:
    #         xList[:,i] = 0.
            
    plt.figure(figsize=(10, 10), dpi=80)
    
    fig,((ax2,ax3),(ax1,ax4)) = plt.subplots(2, 2)
    
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    # ax2.set_yscale('log')
    ax3.set_xlabel('t')
    ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])')

    for i in range(N_x):
        x=torch.tensor(xList[:,i],dtype=torch.float64)/50.
        x_nextList = [x]
        x_next_for_v = x_nextList[-1]
        # x_next_for_v = torch.cat(
        #         (lyapunov_hybrid_system.system.step_forward(x_nextList[-1])[0:2],
        #          torch.zeros((1),dtype=torch.float64)),dim=0)
        vList = [lyap_func(x_next_for_v)[0]]
        # print(lyap_func(x_nextList[-1]))
        vDiffList = [0]
        tol = 1*1E-3
        def converged(y):
            return np.linalg.norm(y.detach().numpy() + x_des) 
        i = 0

        plant = pendulum.Pendulum(torch.float64)
        K, S = plant.lqr_control(np.diag([1., 10.]), np.array([[1.]]))

        # Now train the controller and Lyapunov function together
        q_star = torch.tensor([np.pi], dtype=torch.float64)
        u_star= torch.tensor([0], dtype=torch.float64)
        # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
        # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
        u_lo = torch.tensor([-20], dtype=torch.float64)
        u_up = torch.tensor([20], dtype=torch.float64)
        lyap_func = lambda x_next: np.transpose(x_next.detach().numpy())@S@x_next.detach().numpy()
        while True:
            # print(lyapunov_hybrid_system.system.step_forward(x_nextList[-1])[0:2].shape)
            dtype = torch.float64
            u = torch.tensor(K,dtype=dtype) @ (x_nextList[-1]-torch.tensor(x_des,dtype=dtype)) + torch.tensor(u_star,dtype=dtype)
            # print(K.shape,x_nextList[-1].shape,u.shape)
            x_next = forward_system.step_forward(x_nextList[-1].reshape(-1),u.reshape(-1))#lyapunov_hybrid_system.system.step_forward(x_nextList[-1])
            # print(x_next,closed_loop_system.compute_u(x_nextList[-1]))
            x_next_for_v = x_next
            # print(x_next)
            # x_next_for_v = torch.cat(
            #     (lyapunov_hybrid_system.system.step_forward(x_nextList[-1])[0:2],
            #      torch.zeros((1),dtype=torch.float64)),dim=0)
            # if np.linalg.norm(x_next.detach().numpy())>2:
            #     break
            x_nextList.append(x_next)
            vList.append(lyap_func(x_next_for_v)[0])
            vDiffList.append(vList[-1]+(0.001-1)*vList[-2])
            
            if display_violation_states:
                if vDiffList[-1]>5*1E-4:
                    print('\n',x_nextList[-2],vDiffList[-1])
            i+=1
            if converged(x_next)<tol or i>8000:
                print(i,vDiffList[-1],vList[-10:-1])
                break
            
        x_nextMat = torch.cat(x_nextList, dim=0).detach().numpy().reshape(len(x_nextList),x_dim)
        ax2.plot(np.linspace(0, len(vList)*dt,len(vList)),vList)

        ax3.plot(np.linspace(0, len(vDiffList)*dt,len(vDiffList)),vDiffList)
        
        dim_x = 0
        dim_y = 1
        ax1.plot(x_nextMat[0,dim_x],x_nextMat[0,dim_y],'ro')
        ax1.plot(x_nextMat[:,dim_x],x_nextMat[:,dim_y])#
        ax1.plot(x_nextMat[-1,dim_x],x_nextMat[-1,dim_y],'gx')
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$\theta$")
        ax1.plot(x_des[dim_x],x_des[dim_y],'bx')

    fig.tight_layout()
plot_lqr_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True)

# %%
def plot_levelSet_Linf_region(x_lo,x_up,x_des,num_samples=[80,80],roa_level=None,fig_name=None):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
    plant = pendulum.Pendulum(torch.float64)
    K, S = plant.lqr_control(0.1*np.diag([1., 10.]), 0.1*np.array([[1.]]))

    lyap_func =  lambda x_next: S[0,0]*x_next[:,0]*x_next[:,0] +\
        S[1,1]*x_next[:,1]*x_next[:,1] +\
        2.*S[0,1]*x_next[:,0]*x_next[:,1] 
    v_pred = lyap_func(x_samples-x_des)
    # v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,0].reshape(num_samples[0], num_samples[1])
    Y =  x_samples[:,1].reshape(num_samples[0], num_samples[1])
    Z = v_pred.reshape(num_samples[0], num_samples[1])
    
    origin = 'lower'
    # cmap = "viridis"#pltcm.get_cmap("viridis").copy()

    fig2, ax2 = plt.subplots(constrained_layout=True)
    CS3 = ax2.contourf(X, Y, Z, 
                    origin=origin,
                    extend='both',vmin=0,alpha=0.6)
    # Our data range extends outside the range of levels; make
    # data below the lowest contour level yellow, and above the
    # highest level cyan:
    # CS3.cmap.set_under('yellow')
    # CS3.cmap.set_over('cyan')

    contour_linewidth = 2
    contour_font = 16
    # if x_star is not None:
    #     v = monotonic_NN.forward(torch.tensor(x_star).to(torch.float64)).detach().numpy()
    #     linf = l*np.linalg.norm(np.array(x_star)-x_des,ord=np.inf)
    

        # contour_font = 10
    CS4 = ax2.contour(X, Y, Z, 
                    colors=('k',),
                    linewidths=(1,),
                    origin=origin,alpha=0.5)
    ax2.clabel(CS4, fmt='%2.1f', colors='b', fontsize=contour_font)
    
    if roa_level is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=('r',),
                    linewidths=(3,), label='$V^{-1}(r)$')
        ax2.clabel(CS5, fmt='%2.1f', colors='k', fontsize=contour_font)
        contour_linewidth = 2
        # contour_font = 10
    
    ax2.set_xlim([x_lo[0],x_up[0]])
    ax2.set_ylim([x_lo[1],x_up[1]])   
    ax2.xaxis.set_ticks(np.arange(x_lo[0],x_up[0]+0.1, 2))
    ax2.yaxis.set_ticks(np.arange(x_lo[1],x_up[1]+0.1, 2))
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/'+fig_name, dpi=300)
    return ax2

bound_level_x = 10
bound_level_y = bound_level_x
x_lo_whole = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up_whole = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
N_x = 6
ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,num_samples=[80,80],roa_level=[1,3,5])
# %%
