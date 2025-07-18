import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
import gurobipy
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import scipy.integrate


def get_monotic_params(p_num,epsilon,dtype=torch.float64):
    F_1 = np.eye(p_num)
    # F_1[0,0] = 0
    F_1[1:,:-1] += -1*np.eye(p_num-1)
    # F_1[:2,0] = 0
    F_2 = np.zeros((1,p_num))
    F_2[0,0] = epsilon
    # F_2[0,1] = epsilon
    F_3 = np.zeros((1,p_num))
    # F_3[0,0] = 1E-3
    G = np.tril(np.ones((p_num,p_num)))
    G[0,0] = 0
    return torch.tensor(F_1.T,dtype=dtype),\
        torch.tensor(F_2,dtype=dtype),\
            torch.tensor(F_3,dtype=dtype),\
            torch.tensor(G.T,dtype=dtype)
            
def generate_partition_space(size_partition,dtype=torch.float64,size_in=2,seed=9):
    # if size_in==1:
    #     v_samples = np.linspace(-np.pi,np.pi,size_partition+1)[:-1][None,...].T
    # elif size_in==2:
    #     theta = np.linspace(0.,2.0*np.pi,size_partition+1)[:-1]
    #     v_samples = np.vstack((np.cos(theta), np.sin(theta))).T
    if size_in<=2:
        theta = np.linspace(0.,2.0*np.pi,size_partition+1)[:-1]
        v_samples = np.vstack((np.cos(theta), np.sin(theta))).T
    else:
        np.random.seed(seed=seed)
        v_samples = np.random.rand(size_partition,size_in)-0.5
        v_samples = v_samples/np.linalg.norm(v_samples,axis=1)[:, np.newaxis]
    # print(v_samples)
    return torch.tensor(v_samples,dtype=dtype).requires_grad_()

class LinearyLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, case=1,partition_space=None,grad_limit=0.1,size_in=1, size_out=1, \
        size_partition=1,size_piecewise=1,device=None,dtype=None,x_eqlm=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.size_in, self.size_out, self.size_partition, self.size_piecewise =\
    size_in, size_out,size_partition,size_piecewise
        self.v, self.epsilon = partition_space,grad_limit
        self.x_eqlm = x_eqlm
        self.dtype = dtype
        torch.manual_seed(0)
        if case==1:
            self.b_input = nn.Parameter(torch.empty((size_partition,size_piecewise), **factory_kwargs))
        else:
            self.a_input =nn.Parameter(torch.empty((size_partition, size_piecewise), **factory_kwargs))
        # self.reset_parameters()
        
        self.F_1,self.F_2,self.F_3,self.G = get_monotic_params(self.size_piecewise,self.epsilon,dtype=dtype)
        self.case = case
        self.init_parameters()
        self.reset_parameters()
        if self.case == 1:
            self.out_features = size_piecewise*size_partition
            self.in_features = size_in
        else:
            self.out_features = size_out
            self.in_features = size_piecewise*size_partition

    def init_parameters(self):
        if self.case==1:
            torch.nn.init.uniform_(self.b_input,a=0.1,b=0.5)
        else:
            torch.nn.init.uniform_(self.a_input,a=0.0,b=1.0)
    def reset_parameters(self) -> None:
        if self.case == 1:
            # self.b_input.data = torch.clamp(self.b_input.data,1e-1)
            
            self.b_input.data.clamp_(1e-1)
            self.b = self.b_input@self.G
            self.weight = torch.repeat_interleave(self.v,self.size_piecewise,
                                                        dim=0)
            if self.x_eqlm is None:
                self.bias = -self.b.reshape(-1)
            else:
                self.bias = -(self.b + \
                    (self.v@self.x_eqlm[...,None])@torch.ones(1,self.size_piecewise,dtype=self.dtype)).reshape(-1)
            self.b.retain_grad()
            
        else:
            # self.a_input.data = torch.clamp(self.a_input.data,0.)
            
            self.a_input.data.clamp_(0.)
            self.a = self.a_input@self.F_1 + \
                self.F_2.tile(self.a_input.shape[0],1)+ \
                self.F_3.tile(self.a_input.shape[0],1)
            self.bias = torch.zeros((1,), dtype=self.dtype).requires_grad_() 
            self.weight = self.a.reshape(-1)[...,None].t() 
            self.a.retain_grad()
            
        self.bias.retain_grad()
        self.weight.retain_grad()
            

    def forward(self, x):
        self.reset_parameters()
        return x@self.weight.t()+self.bias

 
def setup_monotonic_relu(size_out=1,
                            size_in=2,
                            epsilon=0.1,
                            size_partition=5,
                            size_piecewise=5,
                            params=None,
                            dtype=torch.float64,
                            x_eqlm=None):
    """
    Setup a relu network.
    @param negative_slope The negative slope of the leaky relu units.
    @param bias whether the linear layer has bias or not.
    """

    layers = [None] * 3
    v = generate_partition_space(size_partition,dtype=dtype,size_in=size_in)
    partition_space = v#.repeat(size_out,1,1)  
    
    layers[0] = LinearyLayer(case=1,partition_space=partition_space,\
        grad_limit=epsilon,size_in=size_in, size_out=size_out,\
            size_partition=size_partition,size_piecewise=size_piecewise,dtype=dtype,x_eqlm=x_eqlm)
    layers[1] = torch.nn.LeakyReLU(0.)
    layers[2] = LinearyLayer(case=2,partition_space=partition_space,\
        grad_limit=epsilon,size_in=size_in, size_out=size_out,\
            size_partition=size_partition,size_piecewise=size_piecewise,dtype=dtype,x_eqlm=x_eqlm)
    relu = torch.nn.Sequential(*layers)
    return relu