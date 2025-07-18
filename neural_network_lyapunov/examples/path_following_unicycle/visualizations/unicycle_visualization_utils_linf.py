import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.train_lyapunov_barrier as train_lyapunov_barrier
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.train_utils as train_utils
import neural_network_lyapunov.r_options as r_options
import torch
import scipy.integrate
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.cm as pltcm
import neural_network_lyapunov.monotonic_lyapunov.custom_lyapunov as custom_lyapunov
import neural_network_lyapunov.lyapunov as lyapunov
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as modified_utils
import gurobipy
import math
from matplotlib.patches import Rectangle
import random
import neural_network_lyapunov.examples.path_following_unicycle_paper.path_following as path_following


def plot_loss_history(load_loss_history):
    loss_history_dict = np.load(load_loss_history,allow_pickle='TRUE').item()

    # loss_history_dict['loss_history'] = [x.item() for x in loss_history_dict['loss_history']]
    fig, ax = plt.subplots()
    loss_history_dict['total_training_time'] = round(loss_history_dict['total_training_time'],1)
    textstr = '\n'.join((
        r'$\mathrm{time}: %.1f \mathrm{ min}$' % (loss_history_dict['total_training_time']/60., ),
        r'$\mathrm{iterations}: %.f$' % (len(loss_history_dict['loss_history']), )))
    ax.plot(loss_history_dict['loss_history'])

    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.65, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')

def plot_loss_historySeq(dir_path,boundList):
    fig, ax = plt.subplots()
    total_time = 0.
    for bound_level in boundList:
        loss_history_dict = np.load(dir_path+str(bound_level)+"_loss_history.npy",allow_pickle='TRUE').item()

    # loss_history_dict['loss_history'] = [x.item() for x in loss_history_dict['loss_history']]
    
        loss_history_dict['total_training_time'] = round(loss_history_dict['total_training_time']/60.,3)
        textstr = ''.join((
            r'$\mathrm{bd-level}~ %.0f ~\mathrm{takes}: %.1f \mathrm{ min}$' % (bound_level,loss_history_dict['total_training_time'] )))
        ax.plot(loss_history_dict['loss_history'],label=textstr)
        total_time+=loss_history_dict['total_training_time']
        print(loss_history_dict['loss_history'][-1])
    ax.legend(loc="upper right")
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title('Total Training Time: '+str(round(total_time,1)) + 'mins')

def setup_closed_loop_system(x_lo,x_up,dt,dynamics_model,controller_relu,controller_Ru,lyapunov_relu,monotonic_flag=True):
    
    dtype = torch.float64
    x_star = np.zeros((3, ))
    u_lo = torch.tensor([0, -0.15 * np.pi], dtype=torch.float64)
    u_up = torch.tensor([1, 0.15 * np.pi], dtype=torch.float64)

    thetadot_as_input = True
    forward_system = unicycle.UnicycleReLUZeroVelModel(torch.float64, x_lo,
                                                       x_up, u_lo, u_up,
                                                       dynamics_model, dt,
                                                       thetadot_as_input)
    # We only stabilize the horizontal position, not the orientation of the car
    Ru_options = r_options.SearchRfreeOptions(controller_Ru.shape)
    Ru_options.set_variable_value(controller_Ru.detach().numpy())
    
    controller_lambda_u = 4.
    # closed_loop_system = unicycle_feedback_system.UnicycleFeedbackSystem(
    #     forward_system, controller_relu,
    #     u_lo.detach().numpy(),
    #     u_up.detach().numpy(), controller_lambda_u, Ru_options)
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
    print('network value at eqlm: ',lyapunov_hybrid_system.lyapunov_relu.forward(torch.tensor(x_star,dtype=torch.float64)))
    return forward_system,closed_loop_system,lyapunov_hybrid_system

def plot_traj(lyapunov_hybrid_system,lyap_func,x_des,x_lo,x_up,N_x=6,dt=0.01,display_violation_states=True):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy() 
    x_dim = len(x_lo_np)
    xList = np.zeros((x_dim,N_x))
    np.random.seed(0)
    for i in range(x_dim):
        xList[i,:] =np.random.rand(N_x)*(x_up_np[i]-x_lo_np[i])+x_lo_np[i]

    plt.figure(figsize=(5, 11), dpi=80)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    # ax2.set_yscale('log')
    ax3.set_xlabel('t')
    ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])')

    for i in range(N_x):
        x=torch.tensor(xList[:,i],dtype=torch.float64)
        x_nextList = [x]
        vList = [lyap_func(x_nextList[-1])[0]]
        # print(lyap_func(x_nextList[-1]))
        vDiffList = [0]
        tol = 1E-3
        def converged(y):
            return np.linalg.norm(y.detach().numpy() - x_des) 
        while True:
            x_next = lyapunov_hybrid_system.system.step_forward(x_nextList[-1])
            x_nextList.append(x_next)
            vList.append(lyap_func(x_next)[0])
            vDiffList.append(vList[-1]+(0.001-1)*vList[-2])
            if display_violation_states:
                if vDiffList[-1]>1E-4:
                    print('\n',x_nextList[-2],vDiffList[-1])
            if converged(x_next)<tol:
                break
        x_nextMat = torch.cat(x_nextList, dim=0).detach().numpy().reshape(len(x_nextList),x_dim)
    
        ax1.plot(x_nextMat[0,0],x_nextMat[0,1],'ro')
        ax1.plot(x_nextMat[:,0],x_nextMat[:,1])#
        ax1.plot(x_nextMat[-1,0],x_nextMat[-1,1],'gx')
        ax2.plot(np.linspace(0, len(vList)*dt,len(vList)),vList)

        ax3.plot(np.linspace(0, len(vDiffList)*dt,len(vDiffList)),vDiffList)
    ax1.plot(x_des[0],x_des[1],'bx')

def l1_value(x,x_equilibrium,lyapunov_relu,V_lambda=0.8,R=None):
    if x.shape == (x_equilibrium.numel(), ):
        # A single state.
        l1_value = V_lambda * torch.norm(R @ (x - x_equilibrium), p=1) 
    else:
        # A batch of states.
        assert (x.shape[1] == x_equilibrium.numel())
        l1_value = V_lambda * torch.norm(R @ (x - x_equilibrium).T, p=1,
                                        dim=0)
    return l1_value
def lyapunov_value(x,x_equilibrium,lyapunov_relu,V_lambda=0.8,add_l1_state=False,R=None):
    relu_at_equilibrium = lyapunov_relu.forward(x_equilibrium)
    if x.shape == (x_equilibrium.numel(), ):
        # A single state.
        lyap_value = lyapunov_relu.forward(x) - relu_at_equilibrium
        lyap_value = lyap_value +\
            V_lambda * torch.norm(R @ (x - x_equilibrium), p=1) if add_l1_state else lyap_value
        return lyap_value
    else:
        # A batch of states.
        assert (x.shape[1] == x_equilibrium.numel())
        lyap_value = lyapunov_relu(x).squeeze() - \
            relu_at_equilibrium.squeeze()
        lyap_value = lyap_value+V_lambda * torch.norm(R @ (x - x_equilibrium).T, p=1,
                                        dim=0) if add_l1_state else lyap_value
        return lyap_value
    
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

def plot_l1_contour(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,R=None,display_v=False,roa_level=None):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
    v_pred = l1_value(x_samples,torch.tensor(x_des),monotonic_NN,V_lambda=V_lambda,R=R)
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
        
    plt.title('Contour Plot of L1 State')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2

def plot_monotonic_network_functions(monotonic_NN=None,plot_flag=True):
    # dim_in = 2
    b_list = monotonic_NN[0].b.clone().detach().numpy()
    a_list = monotonic_NN[2].a.clone().detach().numpy()
    # print(a.shape)
    if plot_flag:
        num_rows = int(np.ceil(b_list.shape[0]/4))
        fig, axs = plt.subplots(num_rows, 4,  figsize=(8,num_rows*2))
    for i in range(num_rows):
        for j in range(4):
            if i*4+j>=a_list.shape[0]:
                axs[i,j].axis('off')
            else:
                if plot_flag:
                    plot_ip = np.arange(-0.1,np.pi,0.2)[...,None]#np.arange(-0.1,b_list.max(),0.2)[...,None]
                    plot_op = np.maximum(0.,np.tile(plot_ip,(1,b_list.shape[1])) - b_list[i*4+j,:])@a_list[i*4+j,:][...,None]
                    # print(a[i,j,:],a_true,b[i,j,:],b_true)
                    # print(plot_ip.shape,plot_op.shape,len(b_true))
                    # plt.clf()
                    if num_rows==1:
                        axs[j].set_xlabel('y')
                        axs[j].set_ylabel('m%d%d(y)'%(i,j))
                        axs[j].set_ylim((-0.1,max(plot_op)))
                        # axs[i].set_title('monotonic unit '+str(i))
                        axs[j].plot(plot_ip,plot_op)
                        for k in range(b_list.shape[1]):
                            axs[j].plot(b_list[i*4+j,k],0,'o')
                            axs[j].axvline(x=b_list[i*4+j,k],linestyle=':')
                    else:
                        axs[i,j].set_xlabel('y')
                        axs[i,j].set_ylabel('m%d%d(y)'%(1,4*i+j))
                        axs[i,j].set_ylim((-0.1,max(plot_op)))
                        # axs[i].set_title('monotonic unit '+str(i))
                        axs[i,j].plot(plot_ip,plot_op)
                        for k in range(b_list.shape[1]):
                            axs[i,j].plot(b_list[i*4+j,k],0,'o')
                            axs[i,j].axvline(x=b_list[i*4+j,k],linestyle=':')
            plt.tight_layout()

def monotonic_layer_direction_visualization(monotonic_NN=None,plot_flag=True):
    v = monotonic_NN[0].v.clone().detach().numpy()
    # print(a.shape)
    if plot_flag:
        num_rows = int(np.ceil(v.shape[0]/4))
        fig, axs = plt.subplots(num_rows, 4,  figsize=(8,num_rows*2))
    for i in range(num_rows):
        for j in range(4):
            if i*4+j>=a.shape[1]:
                axs[i,j].axis('off')
            else:
                size_partition = 1
                t = torch.linspace(0,10,100).double()
                # print(t.shape)
                v_tmp = v[i*4+j,:][None,...]
                
                # print(layer.a_input.shape,layer.b_input.shape,layer.c_input.shape)
                if plot_flag:
                    
                    plot_ip = t[...,None]@v_tmp
                    plot_op = monotonic_NN.forward(plot_ip)
                    # print(plot_op.shape)
                    # print(a[i,j,:],a_true,b[i,j,:],b_true)
                    # print(plot_ip.shape,plot_op.shape,len(b_true))
                    # plt.clf()
                    if num_rows==1:
                        axs[j].set_xlabel('t')
                        axs[j].set_ylabel('M%d%d(t)'%(i,j))
                        axs[j].set_ylim((-0.1,max(plot_op)))
                        # axs[i].set_title('monotonic unit '+str(i))
                        axs[j].plot(t,plot_op)
                    else:
                        axs[i,j].set_xlabel('t')
                        axs[i,j].set_ylabel('M%d%d(t)'%(1,4*i+j))
                        axs[i,j].set_ylim((-0.1,max(plot_op)))
                        # axs[i].set_title('monotonic unit '+str(i))
                        axs[i,j].plot(t,plot_op)
                        
            plt.tight_layout()
            
def monotonic_layer_direction_visualization(monotonic_NN=None,plot_flag=True):
    v = monotonic_NN[0].v.detach()#.numpy()
    x_eqlm = monotonic_NN[0].x_eqlm.detach()[None,...]
    # print(a.shape)
    if plot_flag:
        num_rows = int(np.ceil(v.shape[0]/4))
        fig, axs = plt.subplots(num_rows, 4,  figsize=(8,num_rows*2))
    for i in range(num_rows):
        for j in range(4):
            if i*4+j>=v.shape[0]:
                axs[i,j].axis('off')
            else:
                size_partition = 1
                t = torch.linspace(0,10,100).double()
                # print(t.shape)
                v_tmp = v[i*4+j,:][None,...]
                if plot_flag:
                    plot_ip = t[...,None]@v_tmp + torch.tile(x_eqlm,(t.numel(),1))
                    plot_op = monotonic_NN.forward(plot_ip).detach().numpy()

                    if num_rows==1:
                        axs[j].set_xlabel('t')
                        axs[j].set_ylabel('M%d%d(t)'%(i,j))
                        axs[j].set_ylim((-0.1,max(plot_op)))
                        # axs[i].set_title('monotonic unit '+str(i))
                        axs[j].plot(t,plot_op)
                    else:
                        axs[i,j].set_xlabel('t')
                        axs[i,j].set_ylabel('M%d%d(t)'%(1,4*i+j))
                        axs[i,j].set_ylim((-0.1,max(plot_op)))
                        # axs[i].set_title('monotonic unit '+str(i))
                        axs[i,j].plot(t,plot_op)    
            plt.tight_layout()

def find_roa(lyapunov_hybrid_system,forward_system,x_lo_larger, x_up_larger,V_lambda=0.8, R=None):
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
    return x_sol,milp.gurobi_model.ObjVal

def plot_bLast_Contour(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,R=None,display_v=False,roa_level=None):
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
    if not add_l1_state:
        v = monotonic_NN[0].v.detach().numpy()
        b = monotonic_NN[0].b.detach().numpy()
        print(v.shape)
        for i in range(v.shape[0]): 
            b_last = v[i,:]*b[i,-1]
            transformed_vector = x_des + b_last
            plt.plot(transformed_vector[0],transformed_vector[1],'ro')
            
    plt.title('Contour Plot of Lyapunov Function')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2



# get_levelSet_region(lyapunov_relu_init,lyapunov_upper=1.)   
# result_list= get_levelSet_region(lyapunov_relu_init,lyapunov_upper=1.)
# def get_levelSet_region(lyapunov_relu_init,lyapunov_upper=1.):
#     size_partition = lyapunov_relu_init[0].size_partition
#     size_piecewise = lyapunov_relu_init[0].size_piecewise
#     a = lyapunov_relu_init[2].a.detach().numpy()
#     b = lyapunov_relu_init[0].b.detach().numpy()
#     v = lyapunov_relu_init[0].v.detach().numpy()
#     # print(a.shape,b.shape,v.shape,lyapunov_relu_init[0].size_partition,x_des.shape)
#     result_list = []
#     for i_direction in range(size_partition):
#             a_i = a[i_direction,:]
#             a_i_sum = np.cumsum(a_i)
#             b_i = b[i_direction,:]
#             a_b_prod = np.multiply(a_i,b_i)
#             a_b_prod_sum = np.cumsum(a_b_prod)
#             ncond = np.where(a_b_prod_sum<lyapunov_upper)[0][-1]
#             x = (lyapunov_upper+a_b_prod_sum[ncond])/a_i_sum[ncond]
#             result_list.append(x*v[i_direction,:])
#             # print(a_b_prod_sum,x)
#     l_inf_rect = np.linalg.norm(np.array(result_list), np.inf, axis=0)
#     return result_list, l_inf_rect

def get_levelSet_region(lyapunov_relu_init,lyapunov_upper=1.,add_l1_state=False,R=None,V_lambda=0.1):
    size_partition = lyapunov_relu_init[0].size_partition
    size_piecewise = lyapunov_relu_init[0].size_piecewise
    a = lyapunov_relu_init[2].a.detach().numpy()
    b = lyapunov_relu_init[0].b.detach().numpy()
    v = lyapunov_relu_init[0].v.detach().numpy()
    R = R.detach().numpy()
    # print(a.shape,b.shape,v.shape,lyapunov_relu_init[0].size_partition,x_des.shape)
    result_list = []
    for i_direction in range(size_partition):
        a_i = a[i_direction,:]
        a_i_sum = np.cumsum(a_i)
        b_i = b[i_direction,:]
        # [a0b0, a0b0+a1b1, a0b0+a1b1+a2b2, ....]
        a_b_prod = np.multiply(a_i,b_i)
        a_b_prod_sum = np.cumsum(a_b_prod)
        
        bk_points_val = np.multiply(a_i_sum[0:-1],b_i[1::])
        bk_points_val = np.concatenate((np.array([0.]),bk_points_val-a_b_prod_sum[0:-1]))
        ncond = np.where(bk_points_val<lyapunov_upper)[0][-1]
        x = (lyapunov_upper+a_b_prod_sum[ncond])/a_i_sum[ncond]
        result_list.append(x*v[i_direction,:])
        # print("direction: ", i_direction, ' condition: ', ncond)
        # print(ncond, a_b_prod_sum[ncond],x, v[i_direction,:])
        # print(a_b_prod_sum,x)
    if add_l1_state:
        for i_direction in range(R.shape[0]):
            result_list.append(lyapunov_upper*R[i_direction,:]/(V_lambda*np.linalg.norm(R[i_direction,:])))
            # result_list.append(-lyapunov_upper*R[i_direction,:]/(V_lambda*np.linalg.norm(R[i_direction,:])))
    l_inf_rect = np.linalg.norm(np.array(result_list), np.inf, axis=0)
    return result_list, l_inf_rect

def plot_levelSet_region(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,R=None,display_v=False,roa_level=None,mode='init'):
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
    # if not add_l1_state:
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
    plt.legend(loc="upper right")
    plt.title(mode + ' NN Optimized on Level Set $V^{-1}$(r)')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2

def plot_traj_in_levelSet(lyapunov_hybrid_system,lyapunov_relu,lyap_func,x_eqlm, x_lo_whole, x_up_whole,V_lambda=0.8,add_l1_state=False,R=None,roa_level=0.5,N_x=6,dt=0.01,display_violation_states=True):
    resultList, l_inf_rect = get_levelSet_region(lyapunov_relu,lyapunov_upper=roa_level,add_l1_state=add_l1_state,R=R,V_lambda=V_lambda)
    l_inf_bound_lo = torch.tensor([x_eqlm[0]-l_inf_rect[0], x_eqlm[1]-l_inf_rect[1]], dtype=torch.float64)
    l_inf_bound_up = torch.tensor([x_eqlm[0]+l_inf_rect[0], x_eqlm[1]+l_inf_rect[1]], dtype=torch.float64)
    
    x_lo_np = l_inf_bound_lo.detach().numpy()
    x_up_np = l_inf_bound_up.detach().numpy() 
    
    num_samples = [20,20]
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
    x_samples = torch.tensor(np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1)))).t()
    v_pred = lyapunov_value(x_samples,torch.tensor(x_eqlm),lyapunov_relu,V_lambda=V_lambda,add_l1_state=add_l1_state,R=R)
    v_pred = v_pred.detach().numpy()
    x_samples = x_samples.detach().numpy()
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())
    idx_in_levelSet = np.where(v_pred<=roa_level)[0]
    idx_to_select = random.choices(idx_in_levelSet, k=N_x)
    X = x_samples[idx_to_select,0]#.reshape(num_samples[0], num_samples[1])
    Y =  x_samples[idx_to_select,1]#.reshape(num_samples[0], num_samples[1])
    # Z = v_pred[idx_in_levelSet].reshape(num_samples[0], num_samples[1])
    # print(X,Y)
    xList = np.zeros((2,N_x))
    xList[0,:] = X
    xList[1,:] = Y
    x_des = np.array([np.pi, 0])
    plt.figure(figsize=(17, 5), dpi=80)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    ax1.set_xlabel('$\Theta$')
    ax1.set_ylabel('$\dot{\Theta}$')
    x_lo_whole_np = x_lo_whole.detach().numpy()
    x_up_whole_np = x_up_whole.detach().numpy()
    ax1.set_xlim(x_lo_whole_np[0],x_up_whole_np[0])
    ax1.set_ylim(x_lo_whole_np[1],x_up_whole_np[1])
    ax2.set_xlabel('t')
    ax2.set_ylabel('V')
    # ax2.set_yscale('log')
    ax3.set_xlabel('t')
    ax3.set_ylabel('V(x[n+1])-(1-$\epsilon$)V(x[n])')

    for i in range(N_x):
        x=torch.tensor(xList[:,i],dtype=torch.float64)
        x_nextList = [x]
        vList = [lyap_func(x_nextList[-1])]
        # print(lyap_func(x_nextList[-1]))
        vDiffList = [0]
        tol = 1E-3
        def converged(y):
            return np.linalg.norm(y.detach().numpy() - x_des) 
        while True:
            x_next = lyapunov_hybrid_system.system.step_forward(x_nextList[-1])
            x_nextList.append(x_next)
            vList.append(lyap_func(x_next))
            vDiffList.append(vList[-1]+(0.001-1)*vList[-2])
            if display_violation_states:
                if vDiffList[-1]>1E-4:
                    print('\n',x_nextList[-2],vDiffList[-1])
            if converged(x_next)<tol:
                break
        x_nextMat = torch.cat(x_nextList, dim=0).detach().numpy().reshape(len(x_nextList),2)
    
        ax1.plot(x_nextMat[0,0],x_nextMat[0,1],'ro')
        ax1.plot(x_nextMat[:,0],x_nextMat[:,1])#
        ax1.plot(x_nextMat[-1,0],x_nextMat[-1,1],'gx')
        
        ax2.plot(np.linspace(0, len(vList)*dt,len(vList)),vList)
        ax3.plot(np.linspace(0, len(vDiffList)*dt,len(vDiffList)),vDiffList)
    ax1.plot(np.pi,0,'bx')

def solve_optimization(lyapunov_hybrid_system,x_equilibrium,V_lambda,roa_level,l,mip_pool_solutions=1,R=None):
    milp_return = lyapunov_hybrid_system._construct_milp_for_roa_expand(x_equilibrium,
                                                                    V_lambda,lyapunov_lower=0., 
                                                                    lyapunov_upper=roa_level,
                                                                    L = l,R=R)

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

def Linf_box_bisection_search(lyapunov_hybrid_system,forward_system,V_lambda,roa_level,bisection_search=True,R=None):
    def solve_optimization(lyapunov_hybrid_system,x_equilibrium,V_lambda,roa_level,l,mip_pool_solutions=1):
        milp_return = lyapunov_hybrid_system._construct_milp_for_roa_expand(x_equilibrium,
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
        a = 1e-3
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
  
def l_inf_value(x,x_equilibrium,V_lambda):
    if x.shape == (x_equilibrium.numel(), ):
        # A single state.
        linf_value = V_lambda * torch.norm(x - x_equilibrium, p=torch.inf) 
    else:
        # A batch of states.
        assert (x.shape[1] == x_equilibrium.numel())
        linf_value =  torch.norm((x - x_equilibrium).T, p=torch.inf,dim=0)
    print(x.shape,linf_value.shape)
    return linf_value

def plot_levelSet_Linf_region(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,roa_level_at_bound=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         report_flag=False,x_star_list=None, fig_name=None):
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
    

    CS4 = ax2.contour(X, Y, Z, 
                    colors=('k',),
                    linewidths=(contour_linewidth,),
                    origin=origin)
    ax2.clabel(CS4, fmt='%2.1f', colors='w', fontsize=contour_font)
    if roa_level is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level,
                    colors=('r',),
                    linewidths=(2,), label='$V^{-1}(r)$')
        # ax2.clabel(CS5, fmt='%2.1f', colors='w', fontsize=14)
        contour_linewidth = 2
        # contour_font = 10
    if roa_level_at_bound is not None:
        CS5 = ax2.contour(X, Y, Z, roa_level_at_bound,
                    colors=('m',),
                    linewidths=(2,), label='$V^{-1}(r)$')
        ax2.clabel(CS5, fmt='%2.1f', colors='w', fontsize=14)
        contour_linewidth = 2
        # contour_font = 10
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
                            alpha=0.3,
                            ec ='w',
                            lw = 2, label = 'L_inf box'))
    if x_star is not None:
        plt.plot(x_star[0],x_star[1],'rD', label='$x^*$')
    if x_star_list is not None:
        for i in range(len(x_star_list)):
            plt.plot(x_star_list[i][0],x_star_list[i][1],'rD',)
    if not report_flag:
        plt.legend(loc="upper right")
        plt.title(mode + ' NN Optimized on Level Set $V^{-1}$(r)')
    plt.xlabel('${x_e}$')
    plt.ylabel('${\Theta_e}$')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig('/home/zw2445/Documents/neural-network-lyapunov/plots/report/'+fig_name, dpi=300)
    return ax2

def plot_levelSet_Linf_3D(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         L=None):
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

    fig2 = plt.figure()
    ax2= plt.axes(projection='3d')

    CS3 = ax2.plot3D(X, Y, Z, 
                    origin=origin)
    
    x_samples = np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1))).T
    v_pred = linf_value(x_samples,x_des,L=L)
    v_pred = v_pred
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,0].reshape(num_samples[0], num_samples[1])
    Y =  x_samples[:,1].reshape(num_samples[0], num_samples[1])
    Z = v_pred.reshape(num_samples[0], num_samples[1])

    CS4 = ax2.plot3D(X, Y, Z, 
                    origin=origin)
     
    # plt.legend(loc="upper right")
    plt.title(mode + ' NN Optimized on Level Set $V^{-1}$(r)')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2

def linf_value(x,x_equilibrium,L=0.8):
    print((x - x_equilibrium).T.shape)
    linf_value = np.linalg.norm(L * (x - x_equilibrium).T, ord=np.inf,
                                        axis=0)
    return linf_value

def plot_linf_contour(x_lo,x_up,x_des,num_samples=[80,80],L=None,roa_level=None):
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
    x_samples = np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1))).T
    v_pred = linf_value(x_samples,x_des,L=L)
    v_pred = v_pred
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

    plt.title('Contour Plot of Linf State')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2
def plot_levelSet_Linf_3D(x_lo,x_up,x_des,monotonic_NN=None,num_samples=[80,80],V_lambda=0.8,add_l1_state=False,
                         R=None,display_v=False,roa_level=None,mode='init',l=1.,plot_setBox=True,x_star=None,
                         L=None):
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

    fig2 = plt.figure()
    ax2= plt.axes(projection='3d')

    CS3 = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    
    # CS3 = ax2.contour3D(X, Y, Z, 50)
    
    x_lo_np = x_lo.detach().numpy()
    x_up_np = x_up.detach().numpy()
    (pos_mesh, vel_mesh) = np.meshgrid(np.linspace(x_lo_np[0], x_up_np[0], num_samples[0]),
                            np.linspace(x_lo_np[1], x_up_np[1], num_samples[1]))
    x_samples = np.vstack((np.reshape(pos_mesh, -1), np.reshape(vel_mesh, -1))).T
    v_pred = linf_value(x_samples,x_des,L=L)
    v_pred = v_pred
    # print(lyapunov_relu_init.forward(x_samples).detach().numpy()-lyapunov_relu_trained.forward(x_samples).detach().numpy())

    X = x_samples[:,0].reshape(num_samples[0], num_samples[1])
    Y =  x_samples[:,1].reshape(num_samples[0], num_samples[1])
    Z = v_pred.reshape(num_samples[0], num_samples[1])

    # CS4 = ax2.plot_surface(X, Y, Z, rstride=1, cstride=1, edgecolor='none',alpha=0.2)
    CS4 = ax2.contour3D(X, Y, Z, 50,alpha=0.5,cmap='gray')
    
    v = monotonic_NN.forward(torch.tensor(x_star).to(torch.float64)).detach().numpy()
    v =  v + V_lambda * np.linalg.norm(R.detach().numpy() @ (x_star- x_des), ord=1) if add_l1_state else v
    ax2.scatter3D(x_star[0], x_star[1], v, c=v, cmap='Reds',marker='D')
    # plt.legend(loc="upper right")
    plt.title(mode + ' NN Optimized on Level Set $V^{-1}$(r)')
    plt.xlabel('$\Theta$')
    plt.ylabel('$\dot{\Theta}$')
    return ax2


