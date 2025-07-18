#%%
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')
from neural_network_lyapunov.examples.cart_pole.visualizations.unicycle_visualization_utils import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.colors import ListedColormap
import neural_network_lyapunov.examples.cart_pole.cart_pole as\
    cart_pole
#%%
dt = 0.01
bound_level = 1 # highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
bound_level_theta = bound_level
add_l1_state = True

dtype = torch.float64
x_lo = torch.tensor([-1*bound_level,-np.pi/6*bound_level,
                         -1.5*bound_level,-1*bound_level], dtype=torch.float64)/100.
x_up = torch.tensor([1*bound_level,np.pi/6*bound_level,
                        1.5*bound_level,1*bound_level], dtype=torch.float64)/100.
# x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "data/tedrake/tedrake_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/cart_pole/'#os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "data/preprocess/cart_pole_forward_model.pt"
load_lyapunov_relu = dir_path+ load_prefix+"_lyapunov.pt"
load_controller_relu = dir_path+load_prefix+"_controller.pt"

load_loss_history = dir_path+load_prefix+"_loss_history.npy"
load_R = dir_path+load_prefix+"_R.pt"

dynamics_model = torch.load(load_dynamics_relu, map_location=torch.device('cpu'))

lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)
V_lambda = 0.5
R = torch.load(load_R)

def setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=True):
    
    dtype = torch.float64
    x_star = np.zeros((1,2))
    plant = cart_pole.Cart_Pole(torch.float64)
    q_equilibrium = torch.tensor([0.,0.], dtype=torch.float64)
    u_equilibrium = torch.tensor([0.], dtype=torch.float64)
    # x_lo = torch.tensor([np.pi - 0.1 * np.pi, -0.5], dtype=torch.float64)
    # x_up = torch.tensor([np.pi + 0.1 * np.pi, 0.5], dtype=torch.float64)
    u_lo = torch.tensor([-10.], dtype=torch.float64)
    u_up = torch.tensor([10.], dtype=torch.float64)
    thetadot_as_input = True
    forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
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
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=False)
lyapunov_hybrid_system.add_l1_state = add_l1_state
# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
x_des = np.zeros((4, ))
v_eqlm = lyap_func(torch.tensor(x_des,dtype=dtype))
print('Lyapunov at x_eqlm: ',v_eqlm)
#%%
tmp_dir_path= dir_path+ 'data/preprocess/dataset.pt'
data = torch.load(tmp_dir_path)
model_dataset = data#torch.utils.data.TensorDataset(data["input"],
                     #                           data["output"])
print("loaded dynamics data.")
#%%
plant = cart_pole.Cart_Pole(torch.float64)
for i in range(100):
    ip,op = model_dataset[i*1000]
    state_equilibrium = torch.tensor([0., 0.,0.,0.], dtype=torch.float64)
    control_equilibrium = torch.tensor([0.], dtype=torch.float64)
    network_input_zero = torch.cat((state_equilibrium, control_equilibrium))
    dynamic_output = dynamics_model(ip)-dynamics_model(network_input_zero)
    gt = op[2::]
    print((dynamic_output-gt).detach().numpy())
# %%
for i in range(100):
    ip,op = model_dataset[i*1000]
    dynamic_output = forward_system.step_forward(ip[0:4],ip[4::])
    print((dynamic_output-op).detach().numpy())

# %%
jacobian = torch.autograd.functional.jacobian(dynamics_model, network_input_zero)
A,B = plant.linearized_dynamics(state_equilibrium,control_equilibrium)
# %%
print('Jacobian w.r.t x: \n',jacobian[:,0:4], '\n',A[[2,3],:])
print('Jacobian w.r.t u: \n',jacobian[:,4], '\n',B[[2,3]])
# %%
