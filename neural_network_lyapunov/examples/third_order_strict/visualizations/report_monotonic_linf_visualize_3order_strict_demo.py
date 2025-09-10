#%%
import sys  
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')


from neural_network_lyapunov.examples.third_order_strict.visualizations.unicycle_visualization_utils_linf import *
import neural_network_lyapunov.examples.third_order_strict.path_following as\
    path_following
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.monotonic_lyapunov.custom_lyapunov as custom_lyapunov
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as modified_utils


bound_level = 9 # highest level 5 to give largest analysis region
add_l1_state = True

dtype = torch.float64
bound_level_tmp = 50
x_lo = torch.tensor([-0.03*bound_level_tmp,-0.03*bound_level_tmp,-0.04*bound_level_tmp], dtype=torch.float64)
x_up = torch.tensor([0.03*bound_level_tmp,0.03*bound_level_tmp,0.04*bound_level_tmp], dtype=torch.float64)

# x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
V_lambda = 0.6
def setup_closed_loop_system(x_lo,x_up,dir_path,load_prefix,bound_level=1,monotonic_flag=True,add_l1_state=True):
    load_dynamics_relu = dir_path + "data/preprocess/third_order_forward_model.pt"
    load_lyapunov_relu = dir_path+ load_prefix+"_lyapunov.pt"
    load_controller_relu = dir_path+load_prefix+"_controller.pt"

    load_loss_history = dir_path+load_prefix+"_loss_history.npy"
    load_R = dir_path+load_prefix+"_R.pt"
    load_controller_Ru = dir_path+load_prefix+"_Ru.pt"

    dynamics_model = torch.load(load_dynamics_relu, map_location=torch.device('cpu'))

    lyapunov_relu = torch.load(load_lyapunov_relu)
    controller_relu = torch.load(load_controller_relu)
    R = torch.load(load_R)
    
    dt = 0.01
    x_star = np.zeros((1,2))
    plant = path_following.Third_Order_Strict(torch.float64)
    q_equilibrium = torch.tensor([0.,0.,0.], dtype=torch.float64)
    u_equilibrium = torch.tensor([0.], dtype=torch.float64)
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
    lyapunov_hybrid_system.add_l1_state = add_l1_state
    # print('network value at eqlm: ',lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_star,dtype=torch.float64)))
    return forward_system,closed_loop_system,lyapunov_hybrid_system,R


load_prefix = "data/monotonic_roa_0913/monotonic_bound" +str(bound_level)
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/third_order_strict/'#os.path.dirname(os.path.realpath(__file__))#+"/.."
forward_system,closed_loop_system,lyapunov_hybrid_system,R =\
    setup_closed_loop_system(x_lo,x_up,dir_path,load_prefix,bound_level=bound_level,monotonic_flag=True,add_l1_state=add_l1_state)

load_prefix_init = "data/monotonic_roa_0913/monotonic_bound" +str(1)
dir_path_init = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/third_order_strict/'#os.path.dirname(os.path.realpath(__file__))#+"/.."
forward_system_init,closed_loop_system_init,lyapunov_hybrid_system_init,R_init =\
    setup_closed_loop_system(x_lo,x_up,dir_path_init,load_prefix_init,bound_level=bound_level,monotonic_flag=True,add_l1_state=add_l1_state)


# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
x_des = np.zeros((3, ))
v_eqlm = lyap_func(torch.tensor(x_des,dtype=dtype))
print('Lyapunov at x_eqlm: ',v_eqlm)
x_sol, roa = find_roa(lyapunov_hybrid_system,forward_system,x_lo, x_up,V_lambda=V_lambda, R=R)
print('Region of Attraction: ',roa)
# print(lyapunov_hybrid_system.lyapunov_relu[0].v)
# #%%
# plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True)
# milp1, x_max,x_min = lyapunov_hybrid_system._construct_milp_project_states(forward_system.x_equilibrium,V_lambda, lyapunov_upper=1., R=R)
#%%

def get_boundary_levelset(x_lo_larger,x_up_larger,lyapunov_hybrid_system,forward_system,V_lambda, R):
    milp, x = lyapunov_hybrid_system._construct_milp_for_roa_boundary(
        V_lambda, R, forward_system.x_equilibrium,x_lo_larger.numpy(),x_up_larger.numpy())
    milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
    milp.gurobi_model.optimize()
    x_sol = torch.tensor([v.x for v in x]).detach().numpy()

    return milp.gurobi_model.ObjVal,x_sol

x_lo_larger =  torch.tensor([-1.5,-1.5,-2.], dtype=torch.float64)
x_up_larger =  torch.tensor([1.5,1.5,2.], dtype=torch.float64)
milp_obj,x_sol=get_boundary_levelset(x_lo_larger,x_up_larger,lyapunov_hybrid_system,forward_system,V_lambda, R)
print("level set value at boundary: ", milp_obj, " at point: ",x_sol)
print("Lyapunov NN directions ",lyapunov_hybrid_system.lyapunov_relu[0].v)

roa_level_tmp =2.#milp.gurobi_model.ObjVal
final_l,f_c,x_sol = Linf_box_bisection_search(lyapunov_hybrid_system,forward_system,V_lambda,roa_level_tmp, R=R)
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
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
#%%
import matplotlib.ticker as tck
from scipy.spatial import ConvexHull, convex_hull_plot_2d
def plot_convex_hull(x_points,z_points,zdir,zs,colorSet,alpha,markerSize,mode):
    points =  np.concatenate((x_points[...,np.newaxis], 
                            z_points[...,np.newaxis]),axis=1)
    # print(X.detach().numpy()[indices_roa].shape,points.shape)
    hull = ConvexHull(points)
    # print(points[hull.vertices])
    if mode == 'Init':
        for simplex in hull.simplices:
            #  if points[simplex[0],0]!=points[simplex[1],0] and points[simplex[0],1]!=points[simplex[1],1]:
            plt.plot(points[simplex, 0], points[simplex, 1], colorSet, lw=2, 
                zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
    else:
        for simplex in hull.simplices:
            if points[simplex[0],0]!=points[simplex[1],0] and points[simplex[0],1]!=points[simplex[1],1]:
                plt.plot(points[simplex, 0], points[simplex, 1], colorSet, lw=2, 
                    zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
            else:
                plt.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=2, 
                    zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
        
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,2]
    axis_labels = [r'$x_1$',r'$x_2$',r'$x_3$']
    (pos_mesh, vel_mesh,acc_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]),
                        np.linspace(x_lo_np[2], x_up_np[2], num_samples[2]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1)))).t()
    
    v_pred = lyapunov_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,dim_check[0]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Y =  x_samples[:,dim_check[1]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Z  =  x_samples[:,dim_check[2]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    indices = v_pred<=roa_level[0]
    
    ax.scatter(X.detach().numpy()[indices], 
            Y.detach().numpy()[indices], 
            Z.detach()[indices].numpy(), 
            s = 0.5,alpha=alpha_set, marker='o',
            color=scatter_color,label=label_name)
    markerSize = 0.4
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.01*roa_level[0],v_pred>=0.95*roa_level[0]) 
        # indices_roa =  v_pred<=1.0*roa_level[0]
        markerSize = 2.
        alphaSet = 0.1
        x_points = X.detach().numpy()[indices_roa]
        y_points = Y.detach().numpy()[indices_roa]
        z_points = Z.detach().numpy()[indices_roa]
        # ax.plot(x_points,z_points, 
        #         'b+', zdir='y', zs=x_up_np[1],alpha=alphaSet,markersize=markerSize)
        # ax.plot(y_points, z_points, 
        #         'g+', zdir='x', zs=x_lo_np[0],alpha=alphaSet,markersize=markerSize)
        # ax.plot(x_points, y_points, 
        #         'k+', zdir='z', zs=x_lo_np[2],alpha=alphaSet,markersize=markerSize)
        
        # ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'r-', lw=2, 
        #         zdir='y', zs=x_up_np[1],alpha=alphaSet,markersize=markerSize)
        plot_convex_hull(x_points,z_points,'y',x_up_np[1],scatter_color+'-',0.5,2,mode)
        plot_convex_hull(y_points, z_points,'x',x_lo_np[0],scatter_color+'-',0.5,2,mode)
        plot_convex_hull(x_points, y_points,'z',x_lo_np[2],scatter_color+'-',0.5,2,mode)
        
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 1.))
    # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 1.))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 1.))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]+0.1])
    ax.set_ylim([x_lo[dim_check[1]]-0.1,x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.yaxis.labelpad=12
    ax.xaxis.labelpad=8
    ax.zaxis.labelpad=2
    if plot_setBox:
        resultList, l_inf_rect = get_levelSet_region(monotonic_NN,lyapunov_upper=roa_level[0],add_l1_state=add_l1_state,R=R,V_lambda=V_lambda)
        v = monotonic_NN[0].v.detach().numpy()
        for i in range(len(resultList)): 
            transformed_vector = x_des + resultList[i]#*v[i,:]
            if i==0:
                plt.plot(transformed_vector[0],transformed_vector[1],transformed_vector[2],'ro', label='$m_i^{-1}(r)$')
            else:
                plt.plot(transformed_vector[0],transformed_vector[1],transformed_vector[2],'ro')
            # print(v[i,:],resultList[i],transformed_vector)
        print(l_inf_rect)   
        rect_bl = (x_des[0]-l_inf_rect[0],
                                    x_des[1]-l_inf_rect[1],
                                    x_des[2]-l_inf_rect[2]) 
    if Linf_fit_box:  
        linf_bottom_left_corner = (x_des[0]-l,
                                    x_des[1]-l)  
        # ax.add_patch(Rectangle(linf_bottom_left_corner, 
        #                         2.*l,2.*l,
        #                         alpha=0.3,
        #                         ec ='w',
        #                         lw = 2, label = 'L_inf box'))
        if x_star is not None:
            plt.plot(x_star[0],x_star[1],'rD', label='$x^*$')
        if x_star_list is not None:
            for i in range(len(x_star_list)):
                plt.plot(x_star_list[i][0],x_star_list[i][1],'rD',)   
    # ax.legend()
    plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    ax.dist = 12
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    return ax
numSamples = 70
# %matplotlib notebook
fig = plt.figure(figsize=(3.5*1.8, 2.7*1.8), dpi=600)
ax = fig.add_subplot(projection='3d')

# fig, ax1 = plt.subplots(constrained_layout=True,figsize=(3.5*1.5, 2.7*1.5), dpi=600)
ax.autoscale(tight=True)

# figure = plt.figure(figsize=(4,4.5), facecolor='w')
# ax = figure.gca(projection='3d')


v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final',alpha_set=0.03,add_projection=True)
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Init',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.7,add_projection=True)

# plt.show()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/3rd_order_linf_bound1_bound'+str(bound_level)+'_contour_final.png', bbox_inches='tight')

#%%
numSamples = 20
ax = plt.figure().add_subplot(projection='3d')
# figure = plt.figure(figsize=(4,4.5), facecolor='w')
# ax = figure.gca(projection='3d')
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.03,add_projection=True)
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/3rd_order_linf_bound'+str(bound_level)+'_contour_init.png', dpi=300)
#%%
import plotly.graph_objects as go
# ax = plt.figure().add_subplot(projection='3d')
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,2]
    axis_labels = [r'$x_1$',r'$x_2$',r'$x_3$']
    (pos_mesh, vel_mesh,acc_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]),
                        np.linspace(x_lo_np[2], x_up_np[2], num_samples[2]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1)))).t()
    
    v_pred = lyapunov_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,dim_check[0]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Y =  x_samples[:,dim_check[1]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Z  =  x_samples[:,dim_check[2]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    indices = v_pred<=roa_level[0]
    
    fig= go.Figure(data=go.Isosurface(
            x=list(X),
            y=list(Y),
            z=list(Z),
            value=list(v_pred),
            isomin=0.,
            isomax=roa_level[0],
            opacity=0.7,
            colorscale='viridis',
            surface_count=10, # number of isosurfaces, 2 by default: only min and max
            colorbar_nticks=5, # colorbar ticks correspond to isosurface values
            caps=dict(x_show=False, y_show=False)
        ))

    fig.update_layout(
        margin=dict(t=0, l=0, b=0))
    fig.show()

    # ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 1.))
    # # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    # ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 1.))
    # ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 1.))
    # # ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    # # ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    # # ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    # ax.set_xlabel(axis_labels[0])
    # ax.set_ylabel(axis_labels[1])
    # ax.set_zlabel(axis_labels[2])
    # if plot_setBox:
    #     resultList, l_inf_rect = get_levelSet_region(monotonic_NN,lyapunov_upper=roa_level[0],add_l1_state=add_l1_state,R=R,V_lambda=V_lambda)
    #     v = monotonic_NN[0].v.detach().numpy()
    #     for i in range(len(resultList)): 
    #         transformed_vector = x_des + resultList[i]#*v[i,:]
    #         if i==0:
    #             plt.plot(transformed_vector[0],transformed_vector[1],transformed_vector[2],'ro', label='$m_i^{-1}(r)$')
    #         else:
    #             plt.plot(transformed_vector[0],transformed_vector[1],transformed_vector[2],'ro')
    #         # print(v[i,:],resultList[i],transformed_vector)
    #     print(l_inf_rect)   
    #     rect_bl = (x_des[0]-l_inf_rect[0],
    #                                 x_des[1]-l_inf_rect[1],
    #                                 x_des[2]-l_inf_rect[2]) 
    # if Linf_fit_box:  
    #     linf_bottom_left_corner = (x_des[0]-l,
    #                                 x_des[1]-l)  
    #     # ax.add_patch(Rectangle(linf_bottom_left_corner, 
    #     #                         2.*l,2.*l,
    #     #                         alpha=0.3,
    #     #                         ec ='w',
    #     #                         lw = 2, label = 'L_inf box'))
    #     if x_star is not None:
    #         plt.plot(x_star[0],x_star[1],'rD', label='$x^*$')
    #     if x_star_list is not None:
    #         for i in range(len(x_star_list)):
    #             plt.plot(x_star_list[i][0],x_star_list[i][1],'rD',)   
    # # ax.legend()
    # # plt.subplots_adjust(right=0.15)
    # # plt.tight_layout()
    # ax.dist = 12
    return ax
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[50,50,50],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final',alpha_set=0.06,add_projection=False)

# import pyvista
# import numpy as np

# def make_cube():
#     x = np.linspace(-0.5, 0.5, 25)
#     grid = pyvista.StructuredGrid(*np.meshgrid(x, x, x))
#     surf = grid.extract_surface().triangulate()
#     surf.flip_normals()
#     return surf

# # Create example PolyData meshes for boolean operations
# sphere = pyvista.Sphere(radius=0.65, center=(0, 0, 0))
# cube = make_cube()

# # Perform a boolean difference
# boolean = cube.boolean_difference(sphere)
# boolean.plot(color='darkgrey', smooth_shading=True, split_sharp_edges=True)
#%% projection on axis
def plot_lyapunov_2d_axis_project(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,2]
    axis_labels = [r'$x_1$',r'$x_2$',r'$x_3$']
    (pos_mesh, vel_mesh,acc_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]),
                        np.linspace(x_lo_np[2], x_up_np[2], num_samples[2]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1)))).t()
    
    v_pred = lyapunov_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,dim_check[0]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Y =  x_samples[:,dim_check[1]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Z  =  x_samples[:,dim_check[2]]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    indices = v_pred<=roa_level[0]
    
    return X,Y,Z,indices

    
    
NUM_samples_cnt = 200
X,Y,Z,indices = plot_lyapunov_2d_axis_project(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[NUM_samples_cnt,NUM_samples_cnt,NUM_samples_cnt],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[milp_obj],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final')
X,Y,Z,indices_linf = plot_lyapunov_2d_axis_project(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[NUM_samples_cnt,NUM_samples_cnt,NUM_samples_cnt],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final')

x_lo_np = x_lo.detach().numpy()
x_up_np = x_up.detach().numpy()

#%%
def using_hist2d(ax, x1, x2, bins=(20, 20),cmap_custom="Blues"):
    # https://stackoverflow.com/a/20105673/3015186

    # Answer by askewchan
    ax.hist2d(x1, x2, bins, cmap=cmap_custom, alpha=0.4)#, norm = LogNorm(),cmin=1)
    
from matplotlib.colors import ListedColormap
import matplotlib.pylab as pl
# modify existing Reds colormap with a linearly fading alpha
red = pl.cm.Reds  # original colormap
fading_red = red(np.arange(red.N)) # extract colors
fading_red[:, -1] = np.linspace(0, 1, red.N) # modify alpha
fading_red = ListedColormap(fading_red) # convert to colormap

cmap_custom2 = plt.cm.jet#fading_red
cmap_custom1 = pl.cm.Blues
fig, ax = plt.subplots()
bin_size = 50
using_hist2d(ax, np.array(X[indices]), np.array(Y[indices]), 
             bins=(bin_size, bin_size), cmap_custom=cmap_custom1) #plt.cm.jet
using_hist2d(ax, np.array(X[indices_linf]), np.array(Y[indices_linf]), 
             bins=(bin_size, bin_size), cmap_custom=cmap_custom2) #plt.cm.jet
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
ax.set_xlim([x_lo_np[0],x_up_np[0]])
ax.set_ylim([x_lo_np[1],x_up_np[1]])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/3order_linf_bound'+str(bound_level)+'_contour_2d_1.png', dpi=300)
fig, ax = plt.subplots()
using_hist2d(ax, np.array(X[indices]), np.array(Z[indices]), 
             bins=(bin_size, bin_size), cmap_custom=cmap_custom1) #plt.cm.jet
using_hist2d(ax, np.array(X[indices_linf]), np.array(Z[indices_linf]), 
             bins=(bin_size, bin_size), cmap_custom=cmap_custom2) #plt.cm.jet
plt.xlabel('$x_1$')
plt.ylabel('$x_3$')
ax.set_xlim([x_lo_np[0],x_up_np[0]])
ax.set_ylim([x_lo_np[2],x_up_np[2]])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/order_linf_bound'+str(bound_level)+'_contour_2d_2.png', dpi=300)
fig, ax = plt.subplots()
using_hist2d(ax, np.array(Y[indices]), np.array(Z[indices]), 
             bins=(bin_size, bin_size), cmap_custom=cmap_custom1) #plt.cm.jet
using_hist2d(ax, np.array(Y[indices_linf]), np.array(Z[indices_linf]), 
             bins=(bin_size, bin_size), cmap_custom=cmap_custom2) #plt.cm.jet
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
ax.set_xlim([x_lo_np[1],x_up_np[1]])
ax.set_ylim([x_lo_np[2],x_up_np[2]])
plt.rcParams.update({'font.size': 16})
plt.tight_layout()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/order_linf_bound'+str(bound_level)+'_contour_2d_3.png', dpi=300)
#%%
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80])
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80])
def plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,R=None,display_v=False,roa_level=None,mode=None,plot_bLast=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    # check_dim = [2,3]
    # (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[check_dim[0]], x_up_np[check_dim[0]], num_samples[0]),
    #                         np.linspace(x_lo_np[check_dim[1]], x_up_np[check_dim[1]], num_samples[1]))
    # x_mesh = 0.1 * np.ones_like(pos_mesh)
    # thetha_mesh = -0.0 * np.ones_like(pos_mesh)
    # x_samples = torch.tensor(np.vstack((np.reshape(x_mesh, -1),np.reshape(thetha_mesh, -1),
    #                                     np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
   
    check_dim = [0,2]
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[check_dim[0]], x_up_np[check_dim[0]], num_samples[0]),
                            np.linspace(x_lo_np[check_dim[1]], x_up_np[check_dim[1]], num_samples[1]))
    x_mesh = 0. * np.ones_like(pos_mesh)
    thetha_mesh = 0. * np.ones_like(pos_mesh)
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(x_mesh, -1),
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(thetha_mesh, -1)))).t()
    v_pred = lyapunov_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,check_dim[0]].reshape(num_samples[0], num_samples[1])
    Y =  x_samples[:,check_dim[1]].reshape(num_samples[0], num_samples[1])
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
    if roa_level is not None:
        
        CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=('r',),
                    linewidths=(2,))
        ax2.clabel(CS5, fmt='%.2f', colors='w', fontsize=14)
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
        v = monotonic_NN[0].v.detach().numpy()[:,check_dim]
        scale = min((x_up_np[check_dim[0]]-x_lo_np[check_dim[0]])/2.,(x_up_np[check_dim[1]]-x_lo_np[check_dim[1]])/2.)-0.5
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        for v_i in range(v.shape[0]):
            ax2.text(x_des[check_dim[0]]+v[v_i,0]*scale,x_des[check_dim[1]]+v[v_i,1]*scale, str(v_i), fontsize=contour_font,
                    bbox=props)
            plt.arrow(x=x_des[check_dim[0]], y=x_des[check_dim[1]], dx=v[v_i,0]*scale, dy=v[v_i,1]*scale, width=.01,head_width=0.1) 
    if not add_l1_state and plot_bLast:
        v = monotonic_NN[0].v.detach().numpy()
        b = monotonic_NN[0].b.detach().numpy()
        for i in range(v.shape[0]): 
            b_last = v[i,:]*b[i,-1]
            transformed_vector = x_des + b_last
            plt.plot(transformed_vector[0],transformed_vector[1],'ro')
            
    plt.title('Contour Plot of '+mode+' Lyapunov Function')
    plt.xlabel('$\dot{x}$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2

# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80],\
#     V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,mode='Initial')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=True,roa_level=[0.07],mode='Trained')
#%%


# %%
ax2 = plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[milp.gurobi_model.ObjVal],mode='Trained')
ax2.plot(x_sol[0],x_sol[1],'bo')
