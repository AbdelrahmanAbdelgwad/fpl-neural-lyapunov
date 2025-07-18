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
    # F_3[0,0] = 1.
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
    theta = np.linspace(0.,2.0*np.pi,size_partition+1)[:-1]
    v_samples = np.vstack((np.cos(theta), np.sin(theta))).T
    # np.random.seed(seed=seed)
    # v_samples = np.random.rand(size_partition,size_in)-0.5
    # v_samples = v_samples/np.linalg.norm(v_samples,axis=1)[:, np.newaxis]
    # print(v_samples)
    return torch.tensor(v_samples,dtype=dtype).requires_grad_()

class LinearyLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, case=1,partition_space=None,grad_limit=0.1,size_in=1, size_out=1, \
        size_partition=1,size_piecewise=1,device=None,dtype=None,x_eqlm=None, symm_flag=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.size_in, self.size_out, self.size_partition, self.size_piecewise =\
    size_in, size_out,size_partition,size_piecewise
        self.v, self.epsilon = partition_space,grad_limit
        self.x_eqlm = x_eqlm
        self.dtype = dtype
        self.symm_flag = symm_flag
        torch.manual_seed(0)
        if case==1:
            self.b_input = nn.Parameter(torch.empty((size_partition,size_piecewise), **factory_kwargs))
        else:
            self.a_input =nn.Parameter(torch.empty((size_partition, size_piecewise), **factory_kwargs))
        # self.reset_parameters()
        
        feature_dim = size_piecewise*size_partition
        if self.symm_flag:
            self.v = torch.cat((self.v, -self.v),dim=0)
            feature_dim *= 2
        self.F_1,self.F_2,self.F_3,self.G = get_monotic_params(self.size_piecewise,self.epsilon,dtype=dtype)
        self.case = case
        self.init_parameters()
        self.reset_parameters()
        if self.case == 1:
            self.out_features = feature_dim
            self.in_features = size_in
        else:
            self.out_features = size_out
            self.in_features = feature_dim

    def init_parameters(self):
        if self.case==1:
            torch.nn.init.uniform_(self.b_input,a=0.1,b=0.5)
        else:
            torch.nn.init.uniform_(self.a_input,a=0.0,b=1.0)
    def reset_parameters(self) -> None:
        if self.case == 1:
            # self.b_input.data = torch.clamp(self.b_input.data,1e-1)
            
            self.b_input.data.clamp_(1e-1)
            if self.symm_flag:
                self.b = (self.b_input@self.G).tile(2,1)
            else:
                self.b = self.b_input@self.G
                

            self.weight = torch.repeat_interleave(self.v,self.size_piecewise,
                                                  dim=0)
            # self.bias = -self.b.reshape(-1)
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
            if self.symm_flag:
                self.weight = self.a.reshape(-1)[...,None].t().tile(1,2)
            else:
                self.weight = self.a.reshape(-1)[...,None].t() 
            self.a.retain_grad()
            
        self.bias.retain_grad()
        self.weight.retain_grad()
            

    def forward(self, x):
        self.reset_parameters()
        # print(self.weight.shape,self.bias.shape)
        return x@self.weight.t()+self.bias

 
def setup_monotonic_relu(size_out=1,
                            size_in=2,
                            epsilon=0.1,
                            size_partition=5,
                            size_piecewise=5,
                            params=None,
                            dtype=torch.float64,
                            x_eqlm=None,
                            symm_flag=False):
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
            size_partition=size_partition,size_piecewise=size_piecewise,\
                dtype=dtype,x_eqlm=x_eqlm,symm_flag=symm_flag)
    layers[1] = torch.nn.LeakyReLU(0.)
    layers[2] = LinearyLayer(case=2,partition_space=partition_space,\
        grad_limit=epsilon,size_in=size_in, size_out=size_out,\
            size_partition=size_partition,size_piecewise=size_piecewise,\
                dtype=dtype,x_eqlm=x_eqlm,symm_flag=symm_flag)
    relu = torch.nn.Sequential(*layers)
    return relu


def l_inf_as_mixed_integer_constraint(
        x_lo: torch.Tensor,
        x_up: torch.Tensor) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
    """
    Formulate y=max(x) as mixed-integer constraints on x.
    y >= xᵢ
    y <= xᵢ + (1-αᵢ)(max(x_up) - x_lo[i])
    ∑ᵢ αᵢ = 1
    The slack variable is y, which is also the output
    """
    
    """
    Formulate y=max(x,-x) as mixed-integer constraints on x.
    y >= xᵢ
    y <= xᵢ + (1-αᵢ^+)(max(x_up) - x_lo[i])
    
    y >= -xᵢ
    y <= -xᵢ + (1-αᵢ^-)(max(-x_lo) + x_up[i])
    ∑ᵢ αᵢ^+  + αᵢ^- = 1
    The slack variable is y, which is also the output
    
    xᵢ - y <= 0.
    -xᵢ + y + (αᵢ^+)(max(x_up) - x_lo[i]) <= max(x_up) - x_lo[i]
    -xᵢ - y <= 0.
    xᵢ + y  + (αᵢ^-)(-min(x_lo) + x_up[i])<= -min(x_lo) + x_up[i]
    ∑ᵢ αᵢ^+  + αᵢ^- = 1
    y = Aout_input * x + Aout_slack * slack + Aout_binary * binary + Cout
    Ain_input * x + Ain_slack * slack + Ain_binary * binary <= rhs_in
    Aeq_input * x + Aeq_slack * slack + Aeq_binary * binary = rhs_eq
    """
    
    assert (isinstance(x_lo, torch.Tensor))
    assert (isinstance(x_up, torch.Tensor))
    nx = x_lo.shape[0]
    assert (x_lo.shape == (nx, ))
    assert (x_up.shape == (nx, ))
    assert (torch.all(x_up >= x_lo))
    ret = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    dtype = x_lo.dtype
    ret.Aout_slack = torch.tensor([[1]], dtype=dtype)
    ret.Ain_input = torch.cat(
        (torch.eye(nx, dtype=dtype), -torch.eye(nx, dtype=dtype), 
         -torch.eye(nx, dtype=dtype), torch.eye(nx, dtype=dtype)), dim=0)
    ret.Ain_slack = torch.cat((-torch.ones((nx, 1), dtype=dtype), 
                               torch.ones((nx, 1), dtype=dtype),
                               -torch.ones((nx, 1), dtype=dtype), 
                               torch.ones((nx, 1), dtype=dtype)), dim=0)
    # max_x_up = torch.max(x_up)
    # min_x_lo = torch.min(x_lo)
    max_x_up_plus_minus = torch.max(torch.cat((x_up,-x_lo), dim=0))
    diag_plus = torch.diag(max_x_up_plus_minus - x_lo)
    diag_minus = torch.diag(max_x_up_plus_minus + x_up)
    ret.Ain_binary = torch.cat((torch.zeros((nx, nx*2), dtype=dtype), 
                                torch.cat((diag_plus,torch.zeros_like(diag_plus)), dim=1),
                                torch.zeros((nx, nx*2), dtype=dtype), 
                                torch.cat((torch.zeros_like(diag_minus),diag_minus), dim=1)), 
                               dim=0)
    ret.rhs_in = torch.cat((torch.zeros((nx, ), dtype=dtype),
                            max_x_up_plus_minus - x_lo,
                            torch.zeros((nx, ), dtype=dtype),
                            max_x_up_plus_minus + x_up))
    ret.Aeq_binary = torch.ones((1, 2*nx), dtype=dtype)
    ret.rhs_eq = torch.tensor([1], dtype=dtype)
    # If a the upper bound of x[i] is less than the lower bound of another
    # variable, then it can't be the maximal.
    # non_maximal_idx = torch.nonzero(
    #     torch.any(x_up.unsqueeze(1).repeat(1, nx) -
    #               x_lo.unsqueeze(1).T.repeat(nx, 1) < 0,
    #               dim=1)).squeeze().tolist()
    # non_maximal_idx_minus = torch.nonzero(
    #     torch.any(-x_lo.unsqueeze(1).repeat(1, nx) +
    #               x_up.unsqueeze(1).T.repeat(nx, 1) < 0,
    #               dim=1)).squeeze().tolist()
    # if (len(non_maximal_idx_minus) > 0) and (len(non_maximal_idx)>0):
    #     for (i, idx) in enumerate(non_maximal_idx):
    #         non_maximal_idx_minus[0] =  idx + len(non_maximal_idx)
    # # print(non_maximal_idx_minus)
    # non_maximal_idx.append(non_maximal_idx_minus)
    # if (len(non_maximal_idx) > 0):
    #     Aeq_binary_non_maximal = torch.zeros((len(non_maximal_idx), nx*2),
    #                                          dtype=dtype)
    #     for (i, idx) in enumerate(non_maximal_idx):
    #         Aeq_binary_non_maximal[i, idx] = 1
    #     ret.Aeq_binary = torch.cat((ret.Aeq_binary, Aeq_binary_non_maximal),
    #                                dim=0)
    #     ret.rhs_eq = torch.cat(
    #         (ret.rhs_eq, torch.zeros((len(non_maximal_idx), ), dtype=dtype)))
    #     ret.binary_lo = torch.zeros((nx*2, ), dtype=dtype)
    #     ret.binary_up = torch.ones((nx*2, ), dtype=dtype)
    #     ret.binary_up[non_maximal_idx] = 0
    # print(ret.__dict__)
    return ret

def l_inf_transformed_as_mixed_integer_constraint(
        x_lo: torch.Tensor,
        x_up: torch.Tensor,
        x_equilibrium) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
    """
    Formulate y=max(x) as mixed-integer constraints on x.
    y >= xᵢ
    y <= xᵢ + (1-αᵢ)(max(x_up) - x_lo[i])
    ∑ᵢ αᵢ = 1
    The slack variable is y, which is also the output
    """
    
    """
    Formulate y=max(x-x_eqlm,-x+x_eqlm) as mixed-integer constraints on x.
    y >= xᵢ
    y <= xᵢ + (1-αᵢ^+)(max(x_up) - x_lo[i])
    
    y >= -xᵢ
    y <= -xᵢ + (1-αᵢ^-)(max(-x_lo) + x_up[i])
    ∑ᵢ αᵢ^+  + αᵢ^- = 1
    The slack variable is y, which is also the output
    ###
    xᵢ - y <= 0.
    -xᵢ + y + (αᵢ^+)(max(x_up) - x_lo[i]) <= max(x_up) - x_lo[i]
    -xᵢ - y <= 0.
    xᵢ + y  + (αᵢ^-)(-min(x_lo) + x_up[i])<= -min(x_lo) + x_up[i]
    ∑ᵢ αᵢ^+  + αᵢ^- = 1
    y = Aout_input * x + Aout_slack * slack + Aout_binary * binary + Cout
    Ain_input * x + Ain_slack * slack + Ain_binary * binary <= rhs_in
    Aeq_input * x + Aeq_slack * slack + Aeq_binary * binary = rhs_eq
    """
    
    assert (isinstance(x_lo, torch.Tensor))
    assert (isinstance(x_up, torch.Tensor))
    nx = x_lo.shape[0]
    assert (x_lo.shape == (nx, ))
    assert (x_up.shape == (nx, ))
    assert (torch.all(x_up >= x_lo))
    x_lo = x_lo-x_equilibrium.reshape(x_lo.shape)
    x_up = x_up-x_equilibrium.reshape(x_up.shape)
    ret = gurobi_torch_mip.MixedIntegerConstraintsReturn()
    dtype = x_lo.dtype
    ret.Aout_slack = torch.tensor([[1]], dtype=dtype)
    ret.Ain_input = torch.cat(
        (torch.eye(nx, dtype=dtype), -torch.eye(nx, dtype=dtype), 
         -torch.eye(nx, dtype=dtype), torch.eye(nx, dtype=dtype)), dim=0)
    ret.Ain_slack = torch.cat((-torch.ones((nx, 1), dtype=dtype), 
                               torch.ones((nx, 1), dtype=dtype),
                               -torch.ones((nx, 1), dtype=dtype), 
                               torch.ones((nx, 1), dtype=dtype)), dim=0)
    # max_x_up = torch.max(x_up)
    # min_x_lo = torch.min(x_lo)
    max_x_up_plus_minus = torch.max(torch.cat((x_up,-x_lo), dim=0))
    diag_plus = torch.diag(max_x_up_plus_minus - x_lo)
    diag_minus = torch.diag(max_x_up_plus_minus + x_up)
    ret.Ain_binary = torch.cat((torch.zeros((nx, nx*2), dtype=dtype), 
                                torch.cat((diag_plus,torch.zeros_like(diag_plus)), dim=1),
                                torch.zeros((nx, nx*2), dtype=dtype), 
                                torch.cat((torch.zeros_like(diag_minus),diag_minus), dim=1)), 
                               dim=0)
    ret.rhs_in = torch.cat((torch.zeros((nx, ), dtype=dtype),
                            max_x_up_plus_minus - x_lo,
                            torch.zeros((nx, ), dtype=dtype),
                            max_x_up_plus_minus + x_up)) + torch.cat((x_equilibrium.reshape((nx, )),
                                                                     -x_equilibrium.reshape((nx, )),
                                                                      -x_equilibrium.reshape((nx, )),
                                                                      x_equilibrium.reshape((nx, ))))
    ret.Aeq_binary = torch.ones((1, 2*nx), dtype=dtype)
    ret.rhs_eq = torch.tensor([1], dtype=dtype)
    # If a the upper bound of x[i] is less than the lower bound of another
    # variable, then it can't be the maximal.
    # non_maximal_idx = torch.nonzero(
    #     torch.any(x_up.unsqueeze(1).repeat(1, nx) -
    #               x_lo.unsqueeze(1).T.repeat(nx, 1) < 0,
    #               dim=1)).squeeze().tolist()
    # non_maximal_idx_minus = torch.nonzero(
    #     torch.any(-x_lo.unsqueeze(1).repeat(1, nx) +
    #               x_up.unsqueeze(1).T.repeat(nx, 1) < 0,
    #               dim=1)).squeeze().tolist()
    # if (len(non_maximal_idx_minus) > 0) and (len(non_maximal_idx)>0):
    #     for (i, idx) in enumerate(non_maximal_idx):
    #         non_maximal_idx_minus[0] =  idx + len(non_maximal_idx)
    # # print(non_maximal_idx_minus)
    # non_maximal_idx.append(non_maximal_idx_minus)
    # if (len(non_maximal_idx) > 0):
    #     Aeq_binary_non_maximal = torch.zeros((len(non_maximal_idx), nx*2),
    #                                          dtype=dtype)
    #     for (i, idx) in enumerate(non_maximal_idx):
    #         Aeq_binary_non_maximal[i, idx] = 1
    #     ret.Aeq_binary = torch.cat((ret.Aeq_binary, Aeq_binary_non_maximal),
    #                                dim=0)
    #     ret.rhs_eq = torch.cat(
    #         (ret.rhs_eq, torch.zeros((len(non_maximal_idx), ), dtype=dtype)))
    #     ret.binary_lo = torch.zeros((nx*2, ), dtype=dtype)
    #     ret.binary_up = torch.ones((nx*2, ), dtype=dtype)
    #     ret.binary_up[non_maximal_idx] = 0
    # print(ret.__dict__)
    return ret