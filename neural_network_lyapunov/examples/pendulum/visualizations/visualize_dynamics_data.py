#%%
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov')
from neural_network_lyapunov.examples.pendulum.pendulum_visualization_utils import *
import neural_network_lyapunov.utils as utils
dt = 0.01
bound_level = 10 # highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
add_l1_state = True

x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "/data/tedrake/tedrake_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
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
lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)
V_lambda = 0.6
R = torch.load(load_R)

forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=False)

#%%
tmp_dir_path= dir_path+ '/data/dynamics_dataset.pt'
data = torch.load(tmp_dir_path)
model_dataset = torch.utils.data.TensorDataset(data["input"],
                                                data["output"])
print("loaded dynamics data.")
# %%
for i in range(10):
    ip,op = model_dataset[i*100]
    dynamic_output = lyapunov_hybrid_system.system.step_forward(ip[0:-1])
    print((dynamic_output-op).detach().numpy())
# %%
print(model_dataset[0:100:100000])

# %%
