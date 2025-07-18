#%%
import sys  
sys.path.insert(0, '/usr3/graduate/zw2445/Lyapunov/neural-network-lyapunov')
from neural_network_lyapunov.examples.pendulum.pendulum_visualization_utils import *
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
from matplotlib.patches import Rectangle
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
dt = 0.01
bound_level = 8#  highest level 5 to give largest analysis region
roa_level = 3.
bound_level_x = bound_level
bound_level_y = bound_level
V_lambda = 0.6
add_l1_state = True#True
lyap_v_symm_flag = False

x_lo = torch.tensor([np.pi - 0.1*bound_level_x*np.pi, -0.5*bound_level_y], dtype=torch.float64)
x_up = torch.tensor([np.pi + 0.1*bound_level_x*np.pi, 0.5*bound_level_y], dtype=torch.float64)
x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "/data/monotonic_roa/monotonic_bound"
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


#%%
# x_lo_whole = torch.tensor([np.pi - np.pi, -5], dtype=torch.float64)
# x_up_whole = torch.tensor([np.pi + np.pi, 5], dtype=torch.float64)
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

# ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
#     V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level_tmp],mode='Trained',
#     l=linf/final_l,plot_setBox=False,x_star=x_sol[0])

#%%
def cal_v_linf(x_sol,l):
    M = torch.eye(2,dtype=torch.float64)
    M = torch.vstack((M,torch.sum(M,dim=0)))
    v = lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
    v =  v + V_lambda * np.linalg.norm(R.detach().numpy() @ (x_sol[0]- x_des), ord=1) if add_l1_state else v
    linf = l*np.linalg.norm(M.detach().numpy() @ (np.array(x_sol[0])-x_des),ord=np.inf)
    print(l,M.detach().numpy() @ (np.array(x_sol[0])-x_des))
    return v, linf
def Linf_box_bisection_search(lyapunov_hybrid_system,forward_system,V_lambda,roa_level,bisection_search=True,R=None):
    def solve_optimization(lyapunov_hybrid_system,x_equilibrium,V_lambda,roa_level,l,mip_pool_solutions=1):
        milp_return = lyapunov_hybrid_system._construct_milp_for_generalized_roa_expand(x_equilibrium,
                                                                        V_lambda,lyapunov_lower=0., 
                                                                        lyapunov_upper=roa_level,
                                                                        L = l, R=R)

        milp = milp_return[0]
        milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.PoolSearchMode, 2)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.PoolSolutions,mip_pool_solutions)
        milp.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber,3)           
        milp.gurobi_model.optimize()
    
        # for solution_number in range(np.min((3, milp.gurobi_model.solCount))):
        #         x_sol.append([v.xn for v in milp_return[1]])
        x_sol = torch.tensor([v.x for v in milp_return[1]])
        x_sol = []
        for solution_number in range(np.min((mip_pool_solutions,milp.gurobi_model.solCount))):
            milp.gurobi_model.setParam(gurobipy.GRB.Param.SolutionNumber, solution_number)
            if  milp.gurobi_model.PoolObjVal >= -1000.:
                x_sol.append(
                    [v.xn for v in milp_return[1]])
        return milp.gurobi_model.ObjVal, x_sol
    if bisection_search:
        a = 1#1e-3
        b = 20
        L = torch.empty(1,
                        dtype=torch.float64,
                        requires_grad=True)
        f_a,_ = solve_optimization(lyapunov_hybrid_system,forward_system.x_equilibrium,V_lambda,roa_level,torch.ones_like(L)*a)
        f_b,_ = solve_optimization(lyapunov_hybrid_system,forward_system.x_equilibrium,V_lambda,roa_level,torch.ones_like(L)*b)
        # a<b, f(a)>0, f(b)<0
        tol = 1e-6
        max_iter = 100
        i = 0 

        while i < max_iter:
            c = (a+b)/2
            f_c, x_sol= solve_optimization(lyapunov_hybrid_system,forward_system.x_equilibrium,V_lambda,roa_level,torch.ones_like(L)*c)
            # print(f_c,x_sol)
            v,linf = cal_v_linf(x_sol,c)
            print(v,linf,f_c)
            if (b-a)/2. < tol and np.absolute(f_c)<tol:
                final_l = c
                break
            if np.sign(f_c) == np.sign(f_a):
                a = c
                f_a = f_c
            else:
                b = c
                f_b = f_c
            i += 1
    else:
        return None
    return final_l, f_c, x_sol
  
roa_level_tmp = 3.
final_l,f_c,x_sol = Linf_box_bisection_search(lyapunov_hybrid_system,forward_system,V_lambda,roa_level_tmp, R=R)
# f_expand_c, x_expand_sol= solve_linf_expand_optimization(lyapunov_hybrid_system,forward_system.x_equilibrium,V_lambda,
#                                            roa_level,torch.ones_like(L)*final_l,
#                                            mip_pool_solutions=50,outputFlag=True)
# ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
#     V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level_tmp],mode='Trained',
#     l=linf/final_l,plot_setBox=False,x_star=x_expand_sol[0],x_star_list=x_expand_sol)
#%%
M = torch.eye(2,dtype=torch.float64)
M = torch.vstack((M,torch.sum(M,dim=0)))
v = lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
v =  v + V_lambda * np.linalg.norm(R.detach().numpy() @ (x_sol[0]- x_des), ord=1) if add_l1_state else v
linf = final_l*np.linalg.norm(M.detach().numpy() @ (np.array(x_sol[0])-x_des),ord=np.inf)
print(v,linf,x_sol[0],final_l)
# %%
def linf_value(x,x_equilibrium,lyapunov_relu,L=0.8,M=None):
    if x.shape == (x_equilibrium.numel(), ):
        # A single state.
        l1_value = L * torch.norm(M @ (x - x_equilibrium), p=torch.inf) 
    else:
        # A batch of states.
        assert (x.shape[1] == x_equilibrium.numel())
        l1_value = L * torch.norm(M @ (x - x_equilibrium).T, p=torch.inf,
                                        dim=0)
    return l1_value



# %%
def plot_levelSet_Linf_region(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,x_star_list=None):
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

    fig2, ax2 = plt.subplots(constrained_layout=True)
    CS3 = ax2.contourf(X, Y, Z, 
                    origin=origin,
                    extend='both')
    # Our data range extends outside the range of levels; make
    # data below the lowest contour level yellow, and above the
    # highest level cyan:
    # CS3.cmap.set_under('yellow')
    # CS3.cmap.set_over('cyan')

    contour_linewidth = 2
    contour_font = 14
    # if x_star is not None:
    #     v = monotonic_NN.forward(torch.tensor(x_star).to(torch.float64)).detach().numpy()
    #     linf = l*np.linalg.norm(np.array(x_star)-x_des,ord=np.inf)
    
    if roa_level is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=('r',),
                    linewidths=(2,), label='$V^{-1}(r)$')
        ax2.clabel(CS5, fmt='%2.1f', colors='w', fontsize=14)
        contour_linewidth = 1
        contour_font = 10
    CS4 = ax2.contour(X, Y, Z, 
                    colors=('k',),
                    linewidths=(contour_linewidth,),
                    origin=origin)
    ax2.clabel(CS4, fmt='%2.1f', colors='w', fontsize=contour_font)
    # Notice that the colorbar gets all the information it
    # needs from the ContourSet object, CS3.
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
    v_pred = linf_value(x_samples,torch.tensor(x_des),monotonic_NN,L=l,M=M)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    Z = v_pred.reshape(num_samples[0], num_samples[1])
    origin = 'lower'
    # cmap = "viridis"#pltcm.get_cmap("viridis").copy()
    CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=('r',),
                    linewidths=(2,))
    ax2.clabel(CS5, fmt='%2.1f', colors='k', fontsize=14)
    
    if x_star is not None:
        plt.plot(x_star[0],x_star[1],'rD', label='$x^*$')
    if x_star_list is not None:
        for i in range(len(x_star_list)):
            plt.plot(x_star_list[i][0],x_star_list[i][1],'rD',)
    plt.legend(loc="upper right")
    plt.title(mode + ' NN Optimized on Level Set $V^{-1}$(r)')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2
ax2 = plot_levelSet_Linf_region(x_lo_whole,x_up_whole,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[roa_level_tmp],mode='Trained',
    l=final_l,plot_setBox=False,x_star=x_sol[0])
# %%
