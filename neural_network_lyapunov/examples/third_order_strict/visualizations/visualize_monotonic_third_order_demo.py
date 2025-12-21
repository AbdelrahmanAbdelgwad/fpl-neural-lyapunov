#%%
import sys  
import sys  
sys.path.insert(0, '/home/zw2445/Documents/neural-network-lyapunov/')


from neural_network_lyapunov.examples.third_order_strict.visualizations.unicycle_visualization_utils import *
import neural_network_lyapunov.examples.third_order_strict.path_following as\
    path_following
import neural_network_lyapunov.utils as utils
dt = 0.01
bound_level = 50 # highest level 5 to give largest analysis region
bound_level_x = bound_level
bound_level_y = bound_level
bound_level_theta = bound_level
add_l1_state = True

dtype = torch.float64
bound_level_tmp = bound_level
x_lo = torch.tensor([-0.03*bound_level,-0.03*bound_level,-0.04*bound_level], dtype=torch.float64)
x_up = torch.tensor([0.03*bound_level,0.03*bound_level,0.04*bound_level], dtype=torch.float64)
# x_des = np.array([np.pi, 0])
# load_prefix = "/modified/data/modified_fixController_bound" +str(bound_level)
load_lossSeq_prefix = "data/monotonic/monotonic_bound"
load_prefix = load_lossSeq_prefix +str(bound_level)
dir_path = '/home/zw2445/Documents/neural-network-lyapunov/neural_network_lyapunov/examples/third_order_strict/'#os.path.dirname(os.path.realpath(__file__))#+"/.."

load_dynamics_relu = dir_path + "data/preprocess/third_order_forward_model.pt"
load_lyapunov_relu = dir_path+ load_prefix+"_lyapunov.pt"
load_controller_relu = dir_path+load_prefix+"_controller.pt"

load_loss_history = dir_path+load_prefix+"_loss_history.npy"
load_R = dir_path+load_prefix+"_R.pt"
load_controller_Ru = dir_path+load_prefix+"_Ru.pt"

dynamics_model = torch.load(load_dynamics_relu, map_location=torch.device('cpu'))

lyapunov_relu = torch.load(load_lyapunov_relu)
controller_relu = torch.load(load_controller_relu)

V_lambda = 0.6
R = torch.load(load_R)


forward_system,closed_loop_system,lyapunov_hybrid_system=\
    setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,lyapunov_relu,monotonic_flag=True)
lyapunov_hybrid_system.add_l1_state = add_l1_state
# lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_relu.forward(x_next).detach().numpy()
lyap_func = lambda x_next: lyapunov_hybrid_system.lyapunov_value(x_next,\
    forward_system.x_equilibrium,V_lambda=V_lambda,R=R).detach().numpy()
x_des = np.zeros((3, ))
v_eqlm = lyap_func(torch.tensor(x_des,dtype=dtype))
print('Lyapunov at x_eqlm: ',v_eqlm)

# plot_loss_history(load_loss_history)
#%%
def plot_loss_historySeq(dir_path,boundList):
    fig, ax = plt.subplots()
    total_time = 0.
    for bound_level in boundList:
        if os.path.exists(dir_path+str(bound_level)+"_loss_history.npy"):
            loss_history_dict = np.load(dir_path+str(bound_level)+"_loss_history.npy",allow_pickle='TRUE').item()

            loss_history_dict['loss_history'] = [x.item() for x in loss_history_dict['loss_history']]
        
            loss_history_dict['total_training_time'] = round(loss_history_dict['total_training_time']/60.,1)
            textstr = ''.join((
                r'$\mathrm{bd-level}~ %.0f ~\mathrm{takes}: %.1f \mathrm{ min}$' % (bound_level,loss_history_dict['total_training_time'] )))
            ax.plot(loss_history_dict['loss_history'],label=textstr)
            total_time+=loss_history_dict['total_training_time']
            print(loss_history_dict['loss_history'][-1])
    ax.legend(loc="upper right")
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Total Training Time: '+str(round(total_time,1)) + 'mins')

load_lossSeq_prefix = "data/monotonic/monotonic_bound"
# closed_loop_system.compute_u(forward_system.x_equilibrium)
plot_loss_historySeq(dir_path+load_lossSeq_prefix,[*range(1, 51, 1)])
# plt.title("Bound Level "+str(bound_level)+" Converged in "+str(loss_history_dict['total_training_time'])+" s")
#%%
def plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy() 
    x_dim = len(x_lo_np)
    xList = np.zeros((x_dim,N_x))
    np.random.seed(0)
    for i in range(x_dim):
        xList[i,:] = np.random.rand(N_x)*(x_up_np[i]-x_lo_np[i])+x_lo_np[i]
    # for i in range(N_x):
    #     x_tmp = xList[:,i]
    #     # print(torch.tensor(x_tmp,dtype=torch.float64).shape)
    #     if lyap_func(torch.tensor(x_tmp,dtype=torch.float64))[0]>0.24:
    #         xList[:,i] = 0.
            
    plt.figure(figsize=(10, 10), dpi=80)
    
    fig,((ax2,ax3),(ax1,ax4)) = plt.subplots(2, 2)
    
    
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    # ax2.set_yscale('log')
    ax3.set_xlabel('t')
    ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])')

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
        tol = 1*1E-4
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
            # if np.linalg.norm(x_next.detach().numpy())>2:
            #     break
            x_nextList.append(x_next)
            vList.append(lyap_func(x_next_for_v)[0])
            vDiffList.append(vList[-1]+(0.001-1)*vList[-2])
            
            if display_violation_states:
                if vDiffList[-1]>5*1E-4:
                    print('\n',x_nextList[-2],vDiffList[-1])
            i+=1
            if converged(x_next)<tol or i>5000:
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
        ax1.set_xlabel('$x_1$')
        ax1.set_ylabel('$x_2$')
        ax1.plot(x_des[dim_x],x_des[dim_y],'bx')
        
        dim_x = 0
        dim_y = 2
        ax4.plot(x_nextMat[0,dim_x],x_nextMat[0,dim_y],'ro')
        ax4.plot(x_nextMat[:,dim_x],x_nextMat[:,dim_y])#
        ax4.plot(x_nextMat[-1,dim_x],x_nextMat[-1,dim_y],'gx')
        ax4.set_xlabel('$x_1$')
        ax4.set_ylabel('$x_3$')
        ax4.plot(x_des[dim_x],x_des[dim_y],'bx')
    fig.tight_layout()
plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True)

#%%
print(states_tmp[0][:,-1])
#%%
lyapunov_relu_init = utils.setup_relu((3, 8, 8, 6, 1),
                                    params=None,
                                    negative_slope=0.1,
                                    bias=True,
                                    dtype=torch.float64)
num_samples = [80,80]
lyap_func_init = lambda x_next: lyapunov_relu_init(x_next).detach().numpy()
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80])
# plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80])
def plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,R=None,display_v=False,roa_level=None,mode=None,plot_bLast=False):
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
    ax2.clabel(CS4, fmt='%.2f', colors='w', fontsize=contour_font)
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
    if not add_l1_state and plot_bLast:
        v = monotonic_NN[0].v.detach().numpy()
        b = monotonic_NN[0].b.detach().numpy()
        for i in range(v.shape[0]): 
            b_last = v[i,:]*b[i,-1]
            transformed_vector = x_des + b_last
            plt.plot(transformed_vector[0],transformed_vector[1],'ro')
            
    plt.title('Contour Plot of '+mode+' Lyapunov Function')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2

plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu_init,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,mode='Initial')
plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,mode='Trained')
#%%

import gurobipy
x_lo_larger =  x_lo#torch.tensor([0., -10.], dtype=torch.float64)
x_up_larger =  x_up#torch.tensor([np.pi + np.pi,10.], dtype=torch.float64)
rho = lyapunov_hybrid_system.compute_region_of_attraction(V_lambda, R,forward_system.x_equilibrium,
                                        None, x_lo_larger, x_up_larger)

print(rho)
milp1, _, _, _, _ = lyapunov_hybrid_system._construct_milp_for_roa(V_lambda, R, 
    forward_system.x_equilibrium, x_lo_larger, x_up_larger, True)
milp1.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
milp1.gurobi_model.optimize()

print(milp1.gurobi_model.ObjVal)

milp, x = lyapunov_hybrid_system._construct_milp_for_roa_boundary(
    V_lambda, R, forward_system.x_equilibrium)
milp.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
milp.gurobi_model.optimize()

x_sol = torch.tensor([v.x for v in x]).detach().numpy()
print(milp.gurobi_model.ObjVal,x_sol)

#%%
import matplotlib.ticker as tck
from scipy.spatial import ConvexHull, convex_hull_plot_2d
def plot_convex_hull(x_points,z_points,zdir,zs,colorSet,alpha,markerSize,x_lo,x_up):
    points =  np.concatenate((x_points[...,np.newaxis], 
                            z_points[...,np.newaxis]),axis=1)
    # print(X.detach().numpy()[indices_roa].shape,points.shape)
    hull = ConvexHull(points)

    # print(points[hull.vertices])
    for simplex in hull.simplices:
        x_diff = np.abs(points[simplex[0],0] - points[simplex[1],0])
        y_diff = np.abs(points[simplex[0],1] - points[simplex[1],1])
        if  (points[simplex[0],0]!=x_lo[0]) and \
            (points[simplex[0],0]!=x_up[0]) and \
            (points[simplex[0],1]!=x_lo[1]) and \
            (points[simplex[0],1]!=x_up[1]) and \
            (points[simplex[1],0]!=x_lo[0]) and \
            (points[simplex[1],0]!=x_up[0]) and \
            (points[simplex[1],1]!=x_lo[1]) and \
            (points[simplex[1],1]!=x_up[1]):
            plt.plot(points[simplex, 0], points[simplex, 1], colorSet, lw=2, 
                zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
        else:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k--', lw=2, 
                zdir=zdir, zs=zs,alpha=alpha,markersize=markerSize)
def plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,
                         x_star=None,x_star_list=None,Linf_fit_box=False,add_projection=True,
                         scatter_color='r'):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    (pos_mesh, vel_mesh,acc_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]),
                            np.linspace(x_lo_np[2], x_up_np[2], num_samples[2]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), 
                                        np.reshape(vel_mesh, -1),
                                        np.reshape(acc_mesh, -1)))).t()
    v_pred = lyapunov_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,0]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Y =  x_samples[:,1]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    Z  =  x_samples[:,2]#.reshape(num_samples[0], num_samples[1], num_samples[2])
    indices = v_pred<=roa_level[0]
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(X.detach().numpy()[indices], 
            Y.detach().numpy()[indices], 
            Z.detach()[indices].numpy(), 
            s = 0.1,alpha=0.03,
            color='r',label='roa')
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
        plot_convex_hull(x_points,z_points,'y',x_up_np[1],scatter_color+'-',0.5,2,x_lo[[0,2]],x_up[[0,2]])
        plot_convex_hull(y_points, z_points,'x',x_lo_np[0],scatter_color+'-',0.5,2,x_lo[[1,2]],x_up[[0,1]])
        plot_convex_hull(x_points, y_points,'z',x_lo_np[2],scatter_color+'-',0.5,2,x_lo[[0,1]],x_up[[0,1]])
    axis_labels = [r'$x_1$',r'$x_2$',r'$x_3$']
    dim_check = [0,1,2]
    ax.xaxis.set_ticks(np.arange(x_lo[dim_check[0]],x_up[dim_check[0]]+0.1, 1.))
    # ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%.1f'))
    ax.yaxis.set_ticks(np.arange(x_lo[dim_check[1]],x_up[dim_check[1]]+0.1, 1.))
    ax.zaxis.set_ticks(np.arange(x_lo[dim_check[2]],x_up[dim_check[2]]+0.1, 1.))
    ax.set_xlim([x_lo[dim_check[0]],x_up[dim_check[0]]])
    ax.set_ylim([x_lo[dim_check[1]],x_up[dim_check[1]]])
    ax.set_zlim([x_lo[dim_check[2]],x_up[dim_check[2]]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.yaxis.labelpad=8
    ax.xaxis.labelpad=8
    ax.zaxis.labelpad=8
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    if display_v:
        v = monotonic_NN[0].v.detach().numpy()
        scale = min((x_up_np[0]-x_lo_np[0])/2.,
                    (x_up_np[1]-x_lo_np[1])/2.,
                    (x_up_np[2]-x_lo_np[2])/2.)-0.5
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        for v_i in range(v.shape[0]):
            ax.text(x_des[0]+v[v_i,0]*scale,x_des[1]+v[v_i,1]*scale,x_des[2]+v[v_i,2]*scale, str(v_i),
                bbox=props)
            ax.quiver(x_des[0],x_des[1],x_des[2],
                    v[v_i,0]*scale,v[v_i,1]*scale,v[v_i,2]*scale, 
                    length=1, normalize=True) 
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
    # plt.plot(x_des[0],x_des[1],x_des[2],'gD', label='$x^{eqlm}$')
    if Linf_fit_box:  
        linf_bottom_left_corner = (x_des[0]-l,
                                    x_des[1]-l)  
        # ax.add_patch(Rectangle(linf_bottom_left_corner, 
        #                         2.*l,2.*l,
        #                         alpha=0.3,
        #                         ec ='w',
        #                         lw = 2, label = 'L_inf box'))
        if x_star is not None:
            plt.plot(x_star[0],x_star[1],x_star[2],'rD', label='$x^*$')
        if x_star_list is not None:
            for i in range(len(x_star_list)):
                plt.plot(x_star_list[i][0],x_star_list[i][1],x_star_list[i][2],'rD',)   
    # ax.legend()
    plt.rcParams.update({'font.size': 16})
    ax.dist = 12
    # plt.tight_layout()
    # plt.show()
    plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/3rd_order_bound'+str(bound_level)+'_contour.png', dpi=300)
    
    return ax

ax2 = plot_lyapunov_3d(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[90,90,90],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[milp.gurobi_model.ObjVal],mode='Trained',
    l=1.,plot_setBox=False,x_star=None,Linf_fit_box=False)

#%%
monotonic_NN = lyapunov_relu
display_v = True
roa_level = [milp.gurobi_model.ObjVal]
num_samples=[50,50,50]

# %%
ax2 = plot_lyapunov_contour(x_lo,x_up,x_des,monotonic_NN=lyapunov_relu,num_samples=[80,80],\
    V_lambda=V_lambda,add_l1_state=add_l1_state,R=R,display_v=False,roa_level=[milp.gurobi_model.ObjVal],mode='Trained')
ax2.plot(x_sol[0],x_sol[1],'bo')
# %%
# %%
#%%
def compute_roa(all_points, lyapunov_hybrid_system, horizon=100, tol=1e-3, equilibrium=None, no_traj=True):
    """Compute the largest ROA as a set of states in a discretization."""

    # Forward-simulate all trajectories from initial points in the discretization
    if no_traj:
        end_states = all_points
        for t in range(1, horizon):
            print(end_states.shape)
            end_states = lyapunov_hybrid_system.system.step_forward(end_states)
    else:
        trajectories = np.empty((all_points.shape[0], 2, horizon))
        trajectories[:, :, 0] = all_points.detach().numpy()
        for t in range(1, horizon):
            # print(trajectories.shape,trajectories[:, :, t - 1].shape)
            trajectories[:, :, t] = lyapunov_hybrid_system.system.step_forward(torch.tensor(trajectories[:, :, t - 1])).detach().numpy()
        end_states =torch.tensor(trajectories[:, :, -1])

    if equilibrium is None:
        equilibrium = np.zeros((1, 2))

    # Compute an approximate ROA as all states that end up "close" to 0
    dists = np.linalg.norm(end_states.detach().numpy() - equilibrium, ord=2, axis=1, keepdims=True).ravel()
    roa = (dists <= tol)
    if no_traj:
        return roa
    else:
        return roa, trajectories

# Number of states along each dimension
num_samples = [251,251]
x_lo_np = x_lo.detach().numpy()
x_up_np = x_up.detach().numpy()
(pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                        np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
x_des = np.array([[np.pi, 0]])
roa,trajectories = compute_roa(x_samples, lyapunov_hybrid_system, horizon=1500, tol=1e-1, equilibrium=x_des, no_traj=False)

#%%
from matplotlib.colors import ListedColormap
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

fig = plt.figure(figsize=(5, 5), frameon=True)
plot_limits = np.column_stack((- np.rad2deg([theta_max, omega_max]), np.rad2deg([theta_max*2, omega_max])))


ax = plt.subplot(121)
alpha = 1
colors = [None] * 4
colors[0] = (0, 158/255, 115/255)       # ROA - bluish-green
colors[1] = (230/255, 159/255, 0)       # NN  - orange
colors[2] = (0, 114/255, 178/255)       # LQR - blue
colors[3] = (240/255, 228/255, 66/255)  # SOS - yellow

# True ROA
z = roa.reshape(num_samples)
print(z)
ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(colors[0],), linewidths=1)
ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(colors[0]), alpha=alpha)

# %%
