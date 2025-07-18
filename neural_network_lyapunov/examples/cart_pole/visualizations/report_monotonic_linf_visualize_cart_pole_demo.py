#%%
import sys  
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')


from neural_network_lyapunov.examples.cart_pole.visualizations.report_cartPole_visualization_utils import *
import neural_network_lyapunov.examples.cart_pole.cart_pole as\
    cart_pole
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.monotonic_lyapunov.custom_lyapunov as custom_lyapunov
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as modified_utils


bound_level = 10 # highest level 5 to give largest analysis region
add_l1_state = True

dtype = torch.float64
bound_level_tmp = 100
x_lo = torch.tensor([-1*bound_level_tmp,-np.pi/6*bound_level_tmp,
                        -1.*bound_level_tmp,-1*bound_level_tmp], dtype=torch.float64)/100.
x_up = torch.tensor([1*bound_level_tmp,np.pi/6*bound_level_tmp,
                        1.*bound_level_tmp,1*bound_level_tmp], dtype=torch.float64)/100.# x_des = np.array([np.pi, 0])
# x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
V_lambda = 0.5
load_prefix = "data/monotonic_roa_0913/monotonic_bound" +str(bound_level)
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/cart_pole/'#os.path.dirname(os.path.realpath(__file__))#+"/.."
forward_system,closed_loop_system,lyapunov_hybrid_system,R =\
    setup_closed_loop_system(x_lo,x_up,dir_path,load_prefix,bound_level=bound_level,monotonic_flag=True,add_l1_state=add_l1_state)

load_prefix_init = "data/monotonic_roa_0913/monotonic_bound" +str(1)
dir_path_init = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/cart_pole/'#os.path.dirname(os.path.realpath(__file__))#+"/.."
forward_system_init,closed_loop_system_init,lyapunov_hybrid_system_init,R_init =\
    setup_closed_loop_system(x_lo,x_up,dir_path_init,load_prefix_init,bound_level=bound_level,monotonic_flag=True,add_l1_state=add_l1_state)


# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
x_des = np.zeros((4, ))
v_eqlm = lyap_func(torch.tensor(x_des,dtype=dtype))
print('Lyapunov at x_eqlm: ',v_eqlm)
x_sol, roa = find_roa(lyapunov_hybrid_system,forward_system,x_lo, x_up,V_lambda=V_lambda, R=R)
print('Region of Attraction: ',roa)
# print(lyapunov_hybrid_system.lyapunov_relu[0].v)
# #%%
# plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True)
# milp1, x_max,x_min = lyapunov_hybrid_system._construct_milp_project_states(forward_system.x_equilibrium,V_lambda, lyapunov_upper=1., R=R)
#%%
roa_level_tmp =0.07#milp.gurobi_model.ObjVal
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
def plot_convex_hull(x_points,z_points,zdir,zs,colorSet,alpha,markerSize):
    points =  np.concatenate((x_points[...,np.newaxis], 
                            z_points[...,np.newaxis]),axis=1)
    # print(X.detach().numpy()[indices_roa].shape,points.shape)
    hull = ConvexHull(points)
    # print(points[hull.vertices])
    for simplex in hull.simplices:
        #  if points[simplex[0],0]!=points[simplex[1],0] and points[simplex[0],1]!=points[simplex[1],1]:
        plt.plot(points[simplex, 0], points[simplex, 1], colorSet, lw=2, 
            zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
        # if points[simplex[0],0]!=points[simplex[1],0] and points[simplex[0],1]!=points[simplex[1],1]:
        #     plt.plot(points[simplex, 0], points[simplex, 1], colorSet, lw=2, 
        #         zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
        # else:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=2, 
        #         zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,2]#,3]
    axis_labels = [r'$x$',r'$\Theta$',r'$\dot{x}$'] #,r'$\dot{\Theta}$'
    (pos_mesh, vel_mesh,acc_mesh,jerk_mesh) = np.meshgrid(
                            np.linspace(x_lo_np[dim_check[0]], x_up_np[dim_check[0]], num_samples[0]),
                            np.linspace(x_lo_np[dim_check[1]], x_up_np[dim_check[1]], num_samples[1]),
                            np.linspace(x_lo_np[dim_check[2]], x_up_np[dim_check[2]], num_samples[2]),
                            0.*np.ones_like(np.linspace(x_lo_np[2], x_up_np[2], num_samples[2])))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1),
                                        np.reshape(jerk_mesh, -1)))).t()
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
            s = 1.,alpha=alpha_set, marker='x',
            color=scatter_color,label=label_name)
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.01*roa_level[0],v_pred>=0.9*roa_level[0]) 
        markerSize = 2.
        alphaSet = 0.8
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
        plot_convex_hull(x_points,z_points,'y',x_up_np[1],scatter_color+'-',0.5,2)
        plot_convex_hull(y_points, z_points,'x',x_lo_np[0],scatter_color+'-',0.5,2)
        plot_convex_hull(x_points, y_points,'z',x_lo_np[2],scatter_color+'-',0.5,2)
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=12
    ax.zaxis.labelpad=10
            
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 0.5))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 0.5))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 0.5))
    ax.zaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
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
    # plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    ax.dist = 12
    # plt.show()
    plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_3d_1.png', dpi=600)
    return ax

numSamples = 90
ax = plt.figure().add_subplot(projection='3d')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final', alpha_set=0.003,add_projection=True)
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.1,add_projection=True)

#%%
import matplotlib.ticker as tck
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,3]#,3]
    axis_labels = [r'$x$',r'$\Theta$',r'$\dot{\Theta}$'] #,r'$\dot{\Theta}$',r'$\dot{x}$'
    (pos_mesh, vel_mesh,acc_mesh,jerk_mesh) = np.meshgrid(
                            np.linspace(x_lo_np[dim_check[0]], x_up_np[dim_check[0]], num_samples[0]),
                            np.linspace(x_lo_np[dim_check[1]], x_up_np[dim_check[1]], num_samples[1]),
                            0.*np.ones_like(np.linspace(x_lo_np[2], x_up_np[2], num_samples[2])),
                            np.linspace(x_lo_np[dim_check[2]], x_up_np[dim_check[2]], num_samples[2]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1),
                                        np.reshape(jerk_mesh, -1)))).t()
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
            s = 1.,alpha=alpha_set, marker='x',
            color=scatter_color,label=label_name)
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.01*roa_level[0],v_pred>=0.9*roa_level[0]) 
        markerSize = 2.
        alphaSet = 0.8
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
        plot_convex_hull(x_points,z_points,'y',x_up_np[1],scatter_color+'-',0.5,2)
        plot_convex_hull(y_points, z_points,'x',x_lo_np[0],scatter_color+'-',0.5,2)
        plot_convex_hull(x_points, y_points,'z',x_lo_np[2],scatter_color+'-',0.5,2)
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=12
    ax.zaxis.labelpad=10
            
            
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 0.5))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 0.5))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 0.5))
    ax.zaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
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
    # plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    ax.dist = 12
    return ax

numSamples = 90
ax = plt.figure().add_subplot(projection='3d')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final', alpha_set=0.003,add_projection=True)
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.1,add_projection=True)
# plt.show()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_3d_2.png', dpi=600)
#%%
import matplotlib.ticker as tck
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [3,1,2]#,3]
    axis_labels = [r'$\dot{\Theta}$',r'$\Theta$', r'$\dot{x}$'] #,r'$\dot{\Theta}$'
    (pos_mesh, vel_mesh,acc_mesh,jerk_mesh) = np.meshgrid(
                            0.*np.ones_like(np.linspace(x_lo_np[2], x_up_np[2], num_samples[2])),
                            np.linspace(x_lo_np[dim_check[1]], x_up_np[dim_check[1]], num_samples[1]),
                            np.linspace(x_lo_np[dim_check[2]], x_up_np[dim_check[2]], num_samples[2]),
                            np.linspace(x_lo_np[dim_check[0]], x_up_np[dim_check[0]], num_samples[0]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1),
                                        np.reshape(jerk_mesh, -1)))).t()
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
            s = 1.,alpha=alpha_set, marker='x',
            color=scatter_color,label=label_name)
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.01*roa_level[0],v_pred>=0.9*roa_level[0]) 
        markerSize = 2.
        alphaSet = 0.8
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
        plot_convex_hull(x_points,z_points,'y',x_up_np[1],scatter_color+'-',0.5,2)
        plot_convex_hull(y_points, z_points,'x',x_lo_np[0],scatter_color+'-',0.5,2)
        plot_convex_hull(x_points, y_points,'z',x_lo_np[2],scatter_color+'-',0.5,2)
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=12
    ax.zaxis.labelpad=10
            
            
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 0.5))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 0.5))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 0.5))
    ax.zaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
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
    # plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    ax.dist = 12
    return ax

numSamples = 90
ax = plt.figure().add_subplot(projection='3d')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final', alpha_set=0.003,add_projection=True)
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.1,add_projection=True)
# plt.show()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_3d_3.png', dpi=600)
#%%
import matplotlib.ticker as tck
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,3,2]#,3]
    axis_labels = [r'$x$',r'$\dot{\Theta}$',r'$\dot{x}$'] #,r'$\dot{\Theta}$'
    (pos_mesh, vel_mesh,acc_mesh,jerk_mesh) = np.meshgrid(
                            np.linspace(x_lo_np[dim_check[0]], x_up_np[dim_check[0]], num_samples[0]),
                            0.*np.ones_like(np.linspace(x_lo_np[2], x_up_np[2], num_samples[2])),
                            np.linspace(x_lo_np[dim_check[2]], x_up_np[dim_check[2]], num_samples[2]),
                            np.linspace(x_lo_np[dim_check[1]], x_up_np[dim_check[1]], num_samples[1]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1),
                                        np.reshape(jerk_mesh, -1)))).t()
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
            s = 1.,alpha=alpha_set, marker='x',
            color=scatter_color,label=label_name)
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.01*roa_level[0],v_pred>=0.9*roa_level[0]) 
        markerSize = 2.
        alphaSet = 0.8
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
        plot_convex_hull(x_points,z_points,'y',x_up_np[1],scatter_color+'-',0.5,2)
        plot_convex_hull(y_points, z_points,'x',x_lo_np[0],scatter_color+'-',0.5,2)
        plot_convex_hull(x_points, y_points,'z',x_lo_np[2],scatter_color+'-',0.5,2)
    ax.yaxis.labelpad=10
    ax.xaxis.labelpad=12
    ax.zaxis.labelpad=10
            
            
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 0.5))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 0.5))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 0.5))
    ax.zaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
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
    # plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    ax.dist = 12
    return ax

numSamples = 90
ax = plt.figure().add_subplot(projection='3d')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final', alpha_set=0.003,add_projection=True)
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[numSamples,numSamples,numSamples],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.1,add_projection=True)
# plt.show()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_3d_4.png', dpi=600)
#%%
import matplotlib.ticker as tck
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,2]
    axis_labels = [r'$x$',r'$\Theta$',r'$\dot{x}$']
    (pos_mesh, vel_mesh,acc_mesh,jerk_mesh) = np.meshgrid(
                            np.linspace(x_lo_np[dim_check[0]], x_up_np[dim_check[0]], num_samples[0]),
                            np.linspace(x_lo_np[dim_check[1]], x_up_np[dim_check[1]], num_samples[1]),
                            np.linspace(x_lo_np[dim_check[2]], x_up_np[dim_check[2]], num_samples[2]),
                            0.*np.ones_like(np.linspace(x_lo_np[2], x_up_np[2], num_samples[2])))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1),
                                        np.reshape(jerk_mesh, -1)))).t()
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
            s = 1.5,alpha=alpha_set, marker='x',
            color=scatter_color,label=label_name)
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.001*roa_level[0],v_pred>=0.999*roa_level[0]) 
        markerSize = 2.
        alphaSet = 0.8
        ax.plot(X.detach().numpy()[indices_roa], Z.detach().numpy()[indices_roa], 
                'b+', zdir='y', zs=x_up_np[dim_check[1]],alpha=alphaSet,markersize=markerSize)
        # ax.contourf(X.detach().numpy()[indices_roa], Y.detach().numpy()[indices_roa], 
        #             Z.detach().numpy()[indices_roa],
        #             v_pred[indices_roa].flatten(), zdir='z', offset=5,
        #               levels=v_pred[indices_roa].flatten(),cmap='jet')
        ax.plot(Y.detach().numpy()[indices_roa], Z.detach().numpy()[indices_roa], 
                'g+', zdir='x', zs=x_lo_np[dim_check[0]],alpha=alphaSet,markersize=markerSize)
        ax.plot(X.detach().numpy()[indices_roa], Y.detach().numpy()[indices_roa], 
                'k+', zdir='z', zs=x_lo_np[dim_check[2]],alpha=alphaSet,markersize=markerSize)

    ax.yaxis.labelpad=8
    ax.xaxis.labelpad=8
    ax.zaxis.labelpad=8
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, np.pi/6.))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 0.5))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 0.5))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
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
    # plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    ax.dist = 12
    return ax

ax = plt.figure().add_subplot(projection='3d')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[50,50,50],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final')
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[50,50,50],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.02)
# plt.show()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_3d_2.png', dpi=600)
#%%
import matplotlib.ticker as tck
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         x_star_list=None,Linf_fit_box=False,ax=None,scatter_color='b',
                         label_name='roa init.',alpha_set=0.005,add_projection=False):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    dim_check = [0,1,3]
    axis_labels = [r'$x$',r'$\Theta$',r'$\dot{x}$']
    (pos_mesh, vel_mesh,acc_mesh,jerk_mesh) = np.meshgrid(
                            np.linspace(x_lo_np[dim_check[0]], x_up_np[dim_check[0]], num_samples[0]),
                            np.linspace(x_lo_np[dim_check[1]], x_up_np[dim_check[1]], num_samples[1]),
                            np.linspace(x_lo_np[dim_check[2]], x_up_np[dim_check[2]], num_samples[2]),
                            0.*np.ones_like(np.linspace(x_lo_np[2], x_up_np[2], num_samples[2])))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1),
                                        np.reshape(jerk_mesh, -1)))).t()
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
            s = 1.5,alpha=alpha_set, marker='x',
            color=scatter_color,label=label_name)
    if add_projection:
        indices_roa =  np.logical_and(v_pred<=1.001*roa_level[0],v_pred>=0.999*roa_level[0]) 
        markerSize = 2.
        alphaSet = 0.8
        ax.plot(X.detach().numpy()[indices_roa], Z.detach().numpy()[indices_roa], 
                'b+', zdir='y', zs=x_up_np[dim_check[1]],alpha=alphaSet,markersize=markerSize)
        # ax.contourf(X.detach().numpy()[indices_roa], Y.detach().numpy()[indices_roa], 
        #             Z.detach().numpy()[indices_roa],
        #             v_pred[indices_roa].flatten(), zdir='z', offset=5,
        #               levels=v_pred[indices_roa].flatten(),cmap='jet')
        ax.plot(Y.detach().numpy()[indices_roa], Z.detach().numpy()[indices_roa], 
                'g+', zdir='x', zs=x_lo_np[dim_check[0]],alpha=alphaSet,markersize=markerSize)
        ax.plot(X.detach().numpy()[indices_roa], Y.detach().numpy()[indices_roa], 
                'k+', zdir='z', zs=x_lo_np[dim_check[2]],alpha=alphaSet,markersize=markerSize)

    ax.yaxis.labelpad=8
    ax.xaxis.labelpad=8
    ax.zaxis.labelpad=8
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, np.pi/6.))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 0.5))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 0.5))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
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
    # plt.subplots_adjust(right=0.15)
    # plt.tight_layout()
    
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    ax.dist = 12
    return ax

ax = plt.figure().add_subplot(projection='3d')
v = lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_sol[0]).to(torch.float64)).detach().numpy()
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system.lyapunov_relu,num_samples=[50,50,50],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='r',
                         label_name='roa final')
ax = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_hybrid_system_init.lyapunov_relu,num_samples=[50,50,50],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R_init,display_v=False,roa_level=[roa_level_tmp],mode='Trained',
    l=linf/final_l,plot_setBox=False,x_star=None,Linf_fit_box=True,ax=ax,scatter_color='b',
                         label_name='roa init.',alpha_set=0.02)
plt.show()
plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_3d_2.png', dpi=600)

#%%
#
def plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    x_dim = len(x_lo_np)
    xList = np.zeros((x_dim,200))
    np.random.seed(0)
    dim_check = [0,1,2,3]
    num_samples = [30]*4
    for i in range(x_dim):
        xList[i,:] = np.random.rand(xList.shape[1])*(x_up_np[i]-x_lo_np[i])+x_lo_np[i]
    xList = xList.T
    v_pred = lyapunov_value(torch.tensor(xList),torch.tensor(x_des),lyapunov_hybrid_system.lyapunov_relu,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    indices = v_pred<=roa_level_tmp*10.
    
    xList = xList[indices]
    xList = xList[random.sample(range(xList.shape[0]), N_x),:].T
    # for i in range(N_x):
    #     x_tmp = xList[:,i]
    #     # print(torch.tensor(x_tmp,dtype=torch.float64).shape)
    #     if lyap_func(torch.tensor(x_tmp,dtype=torch.float64))[0]>0.24:
    #         xList[:,i] = 0.
            
    # plt.figure(figsize=(3, 3), dpi=80)
    
    fig,(ax1,ax4,ax2) = plt.subplots(1, 3,figsize=(11, 3.5),
                                     gridspec_kw={'width_ratios': [1,1, 1]})
    plt.rcParams.update({'font.size': 16})
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    # ax2.set_yscale('log')

    for i in range(N_x):
        x=torch.tensor(xList[:,i],dtype=torch.float64)
        x_nextList = [x]
        x_next_for_v = x_nextList[-1]
        # x_next_for_v = torch.cat(
        #         (lyapunov_hybrid_system.system.step_forward(x_nextList[-1])[0:2],
        #          torch.zeros((1),dtype=torch.float64)),dim=0)
        vList = [lyap_func(x_next_for_v)[0]]
        # print(lyap_func(x_nextList[-1]))
        vDiffList = [0]
        tol = 1*1E-2
        def converged(y):
            return np.linalg.norm(y.detach().numpy() - x_des) 
        i = 0
        while True:
            # print(lyapunov_hybrid_system.system.step_forward(x_nextList[-1])[0:2].shape)
            x_next = lyapunov_hybrid_system.system.step_forward(x_nextList[-1])
            # print(x_next,closed_loop_system.compute_u(x_nextList[-1]))
            x_next_for_v = x_next
            # print(x_next)
            # x_next_for_v = torch.cat(
            #     (lyapunov_hybrid_system.system.step_forward(x_nextList[-1])[0:2],
            #      torch.zeros((1),dtype=torch.float64)),dim=0)
            if np.linalg.norm(x_next.detach().numpy())>2:
                break
            x_nextList.append(x_next)
            vList.append(lyap_func(x_next_for_v)[0])
            vDiffList.append(vList[-1]+(0.001-1)*vList[-2])
            
            if display_violation_states:
                if vDiffList[-1]>5*1E-4:
                    print('\n',x_nextList[-2],vDiffList[-1])
            i+=1
            # print(i)
            if converged(x_next)<tol or i>5000:
                print(i,vDiffList[-1],vList[-10:-1])
                break
            
        x_nextMat = torch.cat(x_nextList, dim=0).detach().numpy().reshape(len(x_nextList),x_dim)
        ax2.plot(np.linspace(0, len(vList)*dt,len(vList)),vList)

        # ax3.plot(np.linspace(0, len(vDiffList)*dt,len(vDiffList)),vDiffList)
        

        ax1.plot(np.linspace(0, len(x_nextMat[:,0])*dt,len(x_nextMat[:,0])),x_nextMat[:,0])#
        ax1.set_ylabel(r"$x$")
        ax1.set_xlabel(r"$t$")
        

        ax4.plot(np.linspace(0, len(x_nextMat[:,1])*dt,len(x_nextMat[:,1])),x_nextMat[:,1])
        ax4.set_ylabel(r"$\theta$")
        ax4.set_xlabel(r"$t$")
        # dim_x = 0
        # dim_y = 1
        # ax1.plot(x_nextMat[0,dim_x],x_nextMat[0,dim_y],'ro')
        # ax1.plot(x_nextMat[:,dim_x],x_nextMat[:,dim_y])#
        # ax1.plot(x_nextMat[-1,dim_x],x_nextMat[-1,dim_y],'gx')
        # ax1.set_xlabel(r"$x$")
        # ax1.set_ylabel(r"$\theta$")
        # ax1.plot(x_des[dim_x],x_des[dim_y],'bx')
        
        # dim_x = 0
        # dim_y = 2
        # ax4.plot(x_nextMat[0,dim_x],x_nextMat[0,dim_y],'ro')
        # ax4.plot(x_nextMat[:,dim_x],x_nextMat[:,dim_y])#
        # ax4.plot(x_nextMat[-1,dim_x],x_nextMat[-1,dim_y],'gx')
        # ax4.set_xlabel(r"$x$")
        # ax4.set_ylabel(r"$\dot{x}$")
        # ax4.plot(x_des[dim_x],x_des[dim_y],'bx')

        
    # ax1.plot(x_des[0],x_des[1],'bx')
    # ax4.plot(x_des[0],x_des[1],'bx')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/cart_pole_linf_bound'+str(bound_level)+'_traj.png', dpi=600)
plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True)
# %%
