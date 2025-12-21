#%%
import sys  
sys.path.insert(0, '/usr3/graduate/zw2445/Lyapunov/neural-network-lyapunov')
from neural_network_lyapunov.examples.pendulum.monotonic_init_pendulum_visualization_utils import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.patches import Rectangle
dt = 0.01
bound_level = 10#  highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
V_lambda = 0.6
add_l1_state = True#True
lyap_v_symm_flag = False

x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "/data/monotonic_V06/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
dir_path = os.path.dirname(os.path.realpath(__file__))#+"/.."

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
lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)

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
plot_traj(lyapunov_hybrid_system,lyap_func,x_lo,x_up,N_x=8,dt=0.01,display_violation_states=True)

#%%
lyapunov_relu_init = monotonic_utils.setup_monotonic_relu(size_out=1,
                    size_in=forward_system.x_equilibrium.numel(),
                    epsilon=0.05,
                    size_partition=5,
                    size_piecewise=3,
                    params=None,
                    dtype=torch.float64,
                    x_eqlm=forward_system.x_equilibrium)
num_samples = [80,80]
lyap_func_init = lambda x_next: lyapunov_relu_init(x_next).detach().numpy()
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80])
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80])

plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,mode='Initial')
plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,mode='Trained')
plot_l1_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],V_lambda=V_lambda,R=R,display_v=True)
#%% plot monotonic function on each direction
plot_monotonic_network_functions(monotonic_NN=lyapunov_relu_init,plot_flag=True)
plot_monotonic_network_functions(monotonic_NN=lyapunov_relu,plot_flag=True)
#%% analyse monotonic property on each direction
monotonic_layer_direction_visualization(monotonic_NN=lyapunov_relu,plot_flag=True)
#%% find region of attraction of trained model

x_lo_larger =  torch.tensor([0., -10.], dtype=torch.float64)
x_up_larger =  torch.tensor([np.pi + np.pi,10.], dtype=torch.float64)
x_sol,roa_level = find_roa(lyapunov_hybrid_system,forward_system,x_lo, x_up,V_lambda=V_lambda, R=R)
ax2 = plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level],mode='Trained')
ax2.plot(x_sol[0],x_sol[1],'bo')
# %%
# %% analyse relationship between region of attraction and last b's 
x_lo = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
x_up = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
controller_relu = utils.setup_relu((2, 3, 2, 1),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True,
                                    dtype=torch.float64)
lyapunov_relu_init = modified_utils.setup_monotonic_relu(size_out=1,
                    size_in=forward_system.x_equilibrium.numel(),
                    epsilon=0.05,
                    size_partition=5,
                    size_piecewise=4,
                    params=None,
                    dtype=torch.float64,
                    x_eqlm=forward_system.x_equilibrium,
                    symm_flag=True)
forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu_init)
lyapunov_hybrid_system.add_l1_state = False
num_samples = [80,80]
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80])
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80])

x_lo_whole = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
x_up_whole = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
x_sol,roa_level = find_roa(lyapunov_hybrid_system,forward_system,x_lo_whole, x_up_whole,V_lambda=V_lambda, R=None)
ax2 = plot_lyapunov_contour(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=False,R=R,display_v=True,roa_level=[roa_level],mode='Initial',plot_bLast=True)
ax2.plot(x_sol[0],x_sol[1],'bo')

#%% Illustrate the points on each direction where m_i(x) = r
lyapunov_relu_init = modified_utils.setup_monotonic_relu(size_out=1,
                size_in=forward_system.x_equilibrium.numel(),
                epsilon=0.05,
                size_partition=5,
                size_piecewise=4,
                params=None,
                dtype=torch.float64,
                x_eqlm=forward_system.x_equilibrium,
                symm_flag=False)

roa_level = 0.8
x_lo_whole = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
x_up_whole = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
ax2 = plot_levelSet_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=False,R=R,display_v=True,roa_level=[roa_level])

plot_monotonic_network_functions(monotonic_NN=lyapunov_relu_init,plot_flag=True)

# %%
x_lo_whole = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
x_up_whole = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
ax2 = plot_levelSet_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=False,R=R,display_v=True,roa_level=[1.])
# %%
def relu(x):
    return np.maximum(x,0)
def plot_1D_l_inf_monotonic():
    m = lambda x,a,b: np.dot(a, relu(x-b)) + np.dot(a, relu(-x-b)) 
    l_inf = lambda x,l: l*abs(x)
    x = np.linspace(-1,1,150)
    a = np.array([2.,2.4,1.6])
    b = np.array([0.,0.4,0.8])
    l=2.2
    plt.axhline(y=0.8, label='f(x)=0.8')
    i = 0
    for pt in list(x):
        if i == 0:
            plt.plot(pt, m(pt,a,b),'ro', markersize=1, label='$V_1(x)$')
            plt.plot(pt,l_inf(pt,l),'go', markersize=1, label='$2.2|x|$')
        else:
            plt.plot(pt, m(pt,a,b),'ro', markersize=1)
            plt.plot(pt,l_inf(pt,l),'go', markersize=1)
        i+=1
    a = np.array([0.5,0.5,0.4])
    l = 0.9
    i = 0
    for pt in list(x):
        if i==0:
            plt.plot(pt, m(pt,a,b),'rx', markersize=2.5, label='$V_2(x)$')
            plt.plot(pt,l_inf(pt,l),'gx', markersize=2.5, label='$0.9|x|$')
        else:
            plt.plot(pt, m(pt,a,b),'rx', markersize=2.5)
            plt.plot(pt,l_inf(pt,l),'gx', markersize=2.5)
        i+=1
    plt.legend(loc="upper right")
    plt.title('Illustration of $V(x)\leq l|x|$ in $V^{-1}(0.8)$')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
plot_1D_l_inf_monotonic()
# %%
