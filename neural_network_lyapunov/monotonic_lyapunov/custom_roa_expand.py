# -*- coding: utf-8 -*-
import gurobipy
import torch
import numpy as np

from enum import Enum
import collections

import neural_network_lyapunov.monotonic_lyapunov.custom_relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.hybrid_linear_system as hybrid_linear_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.monotonic_lyapunov.custom_mip_utils as mip_utils
import neural_network_lyapunov.dynamic_system as dynamic_system


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
    max_x_up = torch.max(x_up)
    min_x_lo = torch.min(x_lo)
    ret.Ain_binary = torch.cat((torch.zeros((nx, nx), dtype=dtype), 
                                torch.diag(max_x_up - x_lo),
                                torch.zeros((nx, nx), dtype=dtype), 
                                torch.diag(-min_x_lo + x_up)), dim=0)
    ret.rhs_in = torch.cat((torch.zeros((nx, ), dtype=dtype),
                            max_x_up - x_lo,
                            torch.zeros((nx, ), dtype=dtype),
                            -min_x_lo + x_up))
    ret.Aeq_binary = torch.ones((1, 2*nx), dtype=dtype)
    ret.rhs_eq = torch.tensor([1], dtype=dtype)
    # If a the upper bound of x[i] is less than the lower bound of another
    # variable, then it can't be the maximal.
    non_maximal_idx = torch.nonzero(
        torch.any(x_up.unsqueeze(1).repeat(1, nx) -
                  x_lo.unsqueeze(1).T.repeat(nx, 1) < 0,
                  dim=1)).squeeze().tolist()
    non_maximal_idx_minus = torch.nonzero(
        torch.any(-x_lo.unsqueeze(1).repeat(1, nx) +
                  x_up.unsqueeze(1).T.repeat(nx, 1) < 0,
                  dim=1)).squeeze().tolist()
    non_maximal_idx.append(non_maximal_idx_minus)
    if (len(non_maximal_idx) > 0):
        Aeq_binary_non_maximal = torch.zeros((len(non_maximal_idx), nx*2),
                                             dtype=dtype)
        for (i, idx) in enumerate(non_maximal_idx):
            Aeq_binary_non_maximal[i, idx] = 1
        ret.Aeq_binary = torch.cat((ret.Aeq_binary, Aeq_binary_non_maximal),
                                   dim=0)
        ret.rhs_eq = torch.cat(
            (ret.rhs_eq, torch.zeros((len(non_maximal_idx), ), dtype=dtype)))
        ret.binary_lo = torch.zeros((nx*2, ), dtype=dtype)
        ret.binary_up = torch.ones((nx*2, ), dtype=dtype)
        ret.binary_up[non_maximal_idx] = 0
    return ret

    
def add_roa_set_l_inf_constraint(self,
                            milp,
                            x_equilibrium,
                            x,
                            *,
                            slack_name="linf_gamma",
                            binary_var_name="linf_alpha",
                            binary_var_type=gurobipy.GRB.BINARY,
                            binary_for_zero_input=False):
    """
    ||x||_inf =max(x,-x) = linf_slack, linf_alpha are the binary variables
    """
    if not torch.all(torch.from_numpy(self.system.x_lo_all)
                        <= x_equilibrium) or\
        not torch.all(torch.from_numpy(self.system.x_up_all)
                        >= x_equilibrium):
        raise Exception("add_state_error_l1_constraint: we currently " +
                        "require that x_lo <= x_equilibrium <= x_up")
    mip_cnstr = l_inf_as_mixed_integer_constraint(
                    torch.from_numpy(self.system.x_lo_all),
                    torch.from_numpy(self.system.x_up_all))
    linf_slack, linf_alpha = milp.add_mixed_integer_linear_constraints(
            mip_cnstr, x, None, slack_name,
            binary_var_name, "linf_ineq", "linf_eq", "",
            binary_var_type)
    return (linf_slack, linf_alpha)

def _construct_milp_for_roa_expand(self, V_lambda, R, x_equilibrium):
        """
        Construct an MIP to solve the problem
        min l
        s.t V(x) - l||x||_\inf <= 0
            V(x) <= r
            x \in V^{-1}(r)
        """
        dtype = self.system.dtype
        # I could use the original gurobi interface directly, instead of the
        # gurobi_torch_mip interface, as we don't need to compute the gradient
        # of the results.
        milp = gurobi_torch_mip.GurobiTorchMILP(dtype)
        x_lo = torch.from_numpy(self.system.x_lo_all)
        x_up = torch.from_numpy(self.system.x_up_all)
        x = milp.addVars(self.system.x_dim,
                         lb=x_lo,
                         ub=x_up,
                         vtype=gurobipy.GRB.CONTINUOUS,
                         name="x[n]")
        
        l = milp.addVars(1, vtype=gurobipy.GRB.CONTINUOUS, name='l')

        z, beta, a_out, b_out, _ = self.add_lyap_relu_output_constraint(
            milp, x)
        (linf_gamma, linf_alpha) = self.add_roa_set_l_inf_constraint(milp,
                            x_equilibrium,
                            x,
                            None,
                            slack_name="linf_gamma",
                            binary_var_name="linf_alpha",
                            binary_var_type=gurobipy.GRB.BINARY)

        # V(x) - l||x||_\inf <= 0
        milp.addQConstr([a_out.squeeze(), 1., 1.], [z, l, linf_gamma],
            b=b_out,
            sense=gurobipy.GRB.LESS_EQUAL)

        relu_at_equilibrium = self.lyapunov_relu.forward(x_equilibrium)
        obj_coeff = [1.]
        obj_vars = [l]


        # Objective is
        # min V(x[n])
        # = ϕ(x) − ϕ(x*) + λ|R(x−x*)|₁
        # = a_out * z + b_out + λ * s - ϕ(x*)
        milp.setObjective(obj_coeff,obj_vars,
            constant=b_out - relu_at_equilibrium.squeeze(),
            sense=gurobipy.GRB.MINIMIZE)
        return milp, x