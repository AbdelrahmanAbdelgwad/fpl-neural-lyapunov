#%%
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip
import torch
import unittest
import numpy as np
import gurobipy
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils as monotonic_utils
import neural_network_lyapunov.monotonic_lyapunov.custom_mip_utils as mip_utils

# def constraint_tester(x_lo, x_up, x_eqlm):
    
#     dtype = x_lo.dtype
#     L = torch.tensor([2.],dtype=dtype)
#     M = torch.eye(x_lo.shape[0],dtype=dtype)
#     M = torch.vstack((M,torch.sum(M,dim=0)))
#     I = torch.eye(x_lo.shape[0],dtype=dtype)
#     print(M.shape)
#     print((M@x_eqlm).shape)
    
#     print(x_lo.shape,x_up.shape)
#     lx_lb, lx_ub = mip_utils.compute_range_by_IA(
#         M*L, -M@x_eqlm*L, x_lo,
#         x_up)
#     ret = monotonic_utils.l_inf_generalized_as_mixed_integer_constraint(lx_lb, lx_ub,M)
#     ret.transform_input(torch.eye(x_lo.shape[0],dtype=dtype), 
#                         -x_eqlm)
#     print(ret.Aout_slack.detach().numpy(),
#                                 np.array([[1.]]))
#     prog = gurobi_torch_mip.GurobiTorchMIP(dtype)
#     nx = x_lo.shape[0]
#     x_var = prog.addVars(nx, lb=-gurobipy.GRB.INFINITY)
#     y = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
#     slack, binary = prog.add_mixed_integer_linear_constraints(
#         ret, x_var, y, "", "", "", "", "")
#     prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
#     x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
#     x_eqlm_stack = x_eqlm.reshape((1,nx)).repeat(x_samples.shape[0],1)
#     tmp =(M.detach().clone()@(x_samples.detach().clone()-x_eqlm_stack.detach()).T).T#x_samples-x_eqlm_stack# (M@(x_samples-x_eqlm_stack).T).T
#     x_samples_plus_minus = torch.cat((tmp,-tmp),dim=1)
#     for i in range(x_samples.shape[0]):
#         for j in range(nx):
#             x_var[j].lb = x_samples[i, j].item()
#             x_var[j].ub = x_samples[i, j].item()
#         prog.gurobi_model.optimize()
#         print(y[0].x, torch.max(x_samples_plus_minus[i]).item())
#         binary_val = np.array([v.x for v in binary])
#         # print(np.sum(binary_val), 1)
#         # print(binary_val @ x_samples_plus_minus[i].detach().numpy(),
#         #                         y[0].x)
#         # print(y[0].x, slack[0].x)

# dtype = torch.float64
# constraint_tester(torch.tensor([-2, -3], dtype=dtype),
#                     torch.tensor([1, 4], dtype=dtype),
#                     torch.tensor([0.5, 1], dtype=dtype))

class TestMaxAsMixedIntegerConstraint(unittest.TestCase):
    def constraint_tester(self, x_lo, x_up,x_eqlm):
        dtype = torch.float64
        # x_eqlm = torch.tensor([0.5, 1], dtype=dtype)
        L = torch.tensor([3.],dtype=dtype)
        M = torch.eye(x_lo.shape[0],dtype=dtype)
        # M = torch.vstack((M,torch.sum(M,dim=0)))
        I = torch.eye(x_lo.shape[0],dtype=dtype)
        # print(M.shape)
        # print((M@x_eqlm).shape)
        
        # print(x_lo.shape,x_up.shape)
        # lx_lb, lx_ub = mip_utils.compute_range_by_IA(
        #     M*L, -M@x_eqlm*L, x_lo,
        #     x_up)
        # ret = monotonic_utils.l_inf_generalized_as_mixed_integer_constraint(lx_lb, lx_ub,M)
        # ret.transform_input(torch.eye(x_lo.shape[0],dtype=dtype), 
        #                     -x_eqlm)
        lx_lb, lx_ub = mip_utils.compute_range_by_IA(
            torch.eye(x_lo.shape[0])*L, -x_eqlm*L, x_lo,x_up)
        
        ret = monotonic_utils.l_inf_as_mixed_integer_constraint(
                        lx_lb,
                        lx_ub)
        ret.transform_input(torch.eye(x_lo.shape[0])*L, 
                                  -x_eqlm*L)
        
        # lx_lb, lx_ub = mip_utils.compute_range_by_IA(
        #     M*L, -M@x_eqlm*L, x_lo,x_up)
        
        # ret = monotonic_utils.l_inf_as_mixed_integer_constraint(
        #                 lx_lb,
        #                 lx_ub)
        # ret.transform_input(M*L, 
        #                     -M@x_eqlm*L)
        
        np.testing.assert_allclose(ret.Aout_slack.detach().numpy(),
                                    np.array([[1.]]))
        prog = gurobi_torch_mip.GurobiTorchMIP(dtype)
        nx = x_lo.shape[0]
        x_var = prog.addVars(nx, lb=-gurobipy.GRB.INFINITY)
        y = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
        slack, binary = prog.add_mixed_integer_linear_constraints(
            ret, x_var, y, "", "", "", "", "")
        self.assertEqual(len(slack), 1)
        prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
        x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
        x_eqlm_stack = x_eqlm.reshape((1,nx)).repeat(x_samples.shape[0],1)
        tmp =(M.detach().clone()@(x_samples.detach().clone()-x_eqlm_stack.detach()).T).T#x_samples-x_eqlm_stack# (M@(x_samples-x_eqlm_stack).T).T
        x_samples_plus_minus = torch.cat((tmp,-tmp),dim=1)
        for i in range(x_samples.shape[0]):
            for j in range(nx):
                x_var[j].lb = x_samples[i, j].item()
                x_var[j].ub = x_samples[i, j].item()
            prog.gurobi_model.optimize()
            # print(y[0].x, torch.max(x_samples_plus_minus[i]).item())
            self.assertEqual(prog.gurobi_model.status,
                             gurobipy.GRB.Status.OPTIMAL)
            self.assertAlmostEqual(y[0].x, torch.max(x_samples_plus_minus[i]).item())
            binary_val = np.array([v.x for v in binary])
            self.assertAlmostEqual(np.sum(binary_val), 1)
            tmp = M.detach().numpy()@(x_samples[i].detach().numpy()-x_eqlm.detach().numpy())
            self.assertAlmostEqual(binary_val @ np.concatenate((tmp,-tmp),axis=0),
                                   y[0].x)
            self.assertEqual(y[0].x, slack[0].x)
        

    def test1(self):
        dtype = torch.float64
        self.constraint_tester(torch.tensor([-2, -3], dtype=dtype),
                               torch.tensor([1, 4], dtype=dtype),
                                torch.tensor([0.5, 1], dtype=dtype))

    def test2(self):
        dtype = torch.float64
        self.constraint_tester(torch.tensor([-1, 3, -2], dtype=dtype),
                               torch.tensor([4, 4, 3], dtype=dtype),
                               torch.tensor([0, 3.5, 0], dtype=dtype))

    def test3(self):
        # Some variable's upper bound is smaller than other's lower bound
        dtype = torch.float64
        self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
                               torch.tensor([2, 5, 4, 2], dtype=dtype),
                               torch.tensor([0, 3.5, 0,1.5], dtype=dtype))
        self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
                               torch.tensor([2, 5, 2, 2], dtype=dtype),
                               torch.tensor([0, 3.5, 0,1.5], dtype=dtype))

# class test_absolute_value_as_mixed_integer_constraint(unittest.TestCase):
#     def satisfied(self, mip_cnstr_return, x_val: float, s_val: float,
#                   alpha_val: list):
#         dtype = torch.float64
#         if mip_cnstr_return.rhs_in is not None:
#             lhs_in = torch.zeros_like(mip_cnstr_return.rhs_in, dtype=dtype)
#             if mip_cnstr_return.Ain_input is not None:
#                 lhs_in += mip_cnstr_return.Ain_input @ torch.tensor(
#                     [x_val], dtype=dtype)
#             if mip_cnstr_return.Ain_slack is not None:
#                 lhs_in += mip_cnstr_return.Ain_slack @ torch.tensor(
#                     [s_val], dtype=dtype)
#             if mip_cnstr_return.Ain_binary is not None:
#                 lhs_in += mip_cnstr_return.Ain_binary @ torch.tensor(
#                     alpha_val, dtype=dtype)
#             if not torch.all(lhs_in <= mip_cnstr_return.rhs_in):
#                 return False
#         if mip_cnstr_return.rhs_eq is not None:
#             lhs_eq = torch.zeros_like(mip_cnstr_return.rhs_eq, dtype=dtype)
#             if mip_cnstr_return.Aeq_input is not None:
#                 lhs_eq += mip_cnstr_return.Aeq_input @ torch.tensor(
#                     [x_val], dtype=dtype)
#             if mip_cnstr_return.Aeq_slack is not None:
#                 lhs_eq += mip_cnstr_return.Aeq_slack @ torch.tensor(
#                     [s_val], dtype=dtype)
#             if mip_cnstr_return.Aeq_binary is not None:
#                 lhs_eq += mip_cnstr_return.Aeq_binary @ torch.tensor(
#                     alpha_val, dtype=dtype)
#             if torch.any(torch.abs(lhs_eq - mip_cnstr_return.rhs_eq) > 1E-12):
#                 return False
#         if mip_cnstr_return.binary_lo is not None and torch.any(
#                 torch.tensor(alpha_val, dtype=dtype) <
#                 mip_cnstr_return.binary_lo):
#             return False
#         if mip_cnstr_return.binary_up is not None and torch.any(
#                 torch.tensor(alpha_val, dtype=dtype) >
#                 mip_cnstr_return.binary_up):
#             return False
#         return True

#     def check_x(self, mip_cnstr_return, x_val, s_val, alpha_val, is_satisfied):
#         dtype = torch.float64
#         self.assertEqual(
#             self.satisfied(mip_cnstr_return, x_val, s_val, alpha_val),
#             is_satisfied)
#         mip = gurobi_torch_mip.GurobiTorchMIP(dtype)
#         x_var = mip.addVars(1, lb=-gurobipy.GRB.INFINITY)
#         abs_var = mip.addVars(1, lb=-gurobipy.GRB.INFINITY)
#         s_var, alpha_var = mip.add_mixed_integer_linear_constraints(
#             mip_cnstr_return, x_var, abs_var, "s", "alpha", "", "", "")
#         x_var[0].lb = x_val
#         x_var[0].ub = x_val
#         mip.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
#         mip.gurobi_model.optimize()
#         if mip.gurobi_model.status == gurobipy.GRB.Status.OPTIMAL:
#             self.assertAlmostEqual(abs_var[0].x, np.abs(x_val))

#     def test1(self):
#         # x_lo < 0 < x_up and binary_for_zero_input=False
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(-2, dtype=dtype),
#             torch.tensor(3, dtype=dtype),
#             binary_for_zero_input=False)
#         self.check_x(mip_cnstr_return, -1, 1, [0], True)
#         self.check_x(mip_cnstr_return, -2, 2, [0], True)
#         self.check_x(mip_cnstr_return, 1, 1, [1], True)
#         self.check_x(mip_cnstr_return, 3, 3, [1], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [1], True)
#         self.check_x(mip_cnstr_return, 4, 4, [1], False)
#         self.check_x(mip_cnstr_return, -3, 3, [0], False)
#         self.check_x(mip_cnstr_return, -1, -1, [0], False)
#         self.check_x(mip_cnstr_return, -1, 1, [1], False)
#         self.check_x(mip_cnstr_return, 1, 2, [1], False)
#         self.check_x(mip_cnstr_return, 1, 0, [1], False)
#         self.check_x(mip_cnstr_return, 1, 0, [0], False)

#     def test2(self):
#         # x_lo < 0 < x_up and binary_for_zero_input=True
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(-2, dtype=dtype),
#             torch.tensor(3, dtype=dtype),
#             binary_for_zero_input=True)
#         self.check_x(mip_cnstr_return, -2, 2, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, -1, 1, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 1, 0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 2, 2, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 3, 3, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 4, 4, [0, 0, 1], False)
#         self.check_x(mip_cnstr_return, -3, 3, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, -1, 1, [1, 1, 0], False)
#         self.check_x(mip_cnstr_return, -1, -1, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, 1, -1, [0, 0, 1], False)
#         self.check_x(mip_cnstr_return, 1, 1, [0, 1, 0], False)
#         self.check_x(mip_cnstr_return, 1, 1, [0, 1, 1], False)
#         self.check_x(mip_cnstr_return, -1, 1, [0, 1, 0], False)

#     def test3(self):
#         # x_lo >= 0 and binary_for_zero_input=False
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(0, dtype=dtype),
#             torch.tensor(3, dtype=dtype),
#             binary_for_zero_input=False)
#         self.check_x(mip_cnstr_return, 0, 0, [1], True)
#         self.check_x(mip_cnstr_return, 1, 1, [1], True)
#         self.check_x(mip_cnstr_return, 3, 3, [1], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0], False)
#         self.check_x(mip_cnstr_return, -1, 1, [0], False)
#         self.check_x(mip_cnstr_return, 4, 4, [1], False)
#         self.check_x(mip_cnstr_return, 1, 2, [1], False)
#         self.check_x(mip_cnstr_return, 1, 1, [0], False)

#     def test4(self):
#         # x_lo > 0 and binary_for_zero_input=True
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(1, dtype=dtype),
#             torch.tensor(3, dtype=dtype),
#             binary_for_zero_input=True)
#         self.check_x(mip_cnstr_return, 1, 1, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 2, 2, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 3, 3, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 2, 2, [0, 1, 0], False)
#         self.check_x(mip_cnstr_return, 2, 2, [0, 1, 1], False)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 0, 1], False)
#         self.check_x(mip_cnstr_return, 4, 4, [0, 0, 1], False)
#         self.check_x(mip_cnstr_return, 2, 3, [0, 0, 1], False)

#     def test5(self):
#         # x_lo == 0 and binary_for_zero_input=True
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(0, dtype=dtype),
#             torch.tensor(3, dtype=dtype),
#             binary_for_zero_input=True)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 1, 0], True)
#         self.check_x(mip_cnstr_return, 1, 1, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 3, 3, [0, 0, 1], True)
#         self.check_x(mip_cnstr_return, 0, 0, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 1, 1], False)
#         self.check_x(mip_cnstr_return, 1, 1, [0, 1, 0], False)
#         self.check_x(mip_cnstr_return, -1, 1, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, 4, 4, [0, 0, 1], False)
#         self.check_x(mip_cnstr_return, 2, 3, [0, 0, 1], False)

#     def test6(self):
#         # x_up <= 0 and binary_for_zero_input=False
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(-2, dtype=dtype),
#             torch.tensor(0, dtype=dtype),
#             binary_for_zero_input=False)
#         self.check_x(mip_cnstr_return, -2, 2, [0], True)
#         self.check_x(mip_cnstr_return, -1, 1, [0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [1], False)
#         self.check_x(mip_cnstr_return, -3, 3, [0], False)
#         self.check_x(mip_cnstr_return, 1, 1, [1], False)
#         self.check_x(mip_cnstr_return, -1, 1, [1], False)
#         self.check_x(mip_cnstr_return, -1, 2, [0], False)

#     def test7(self):
#         # x_up < 0 and binary_for_zero_input=True
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(-2, dtype=dtype),
#             torch.tensor(-1, dtype=dtype),
#             binary_for_zero_input=True)
#         self.check_x(mip_cnstr_return, -2, 2, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, -1.5, 1.5, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, -1, 1, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, -1, 1, [0, 1, 0], False)
#         self.check_x(mip_cnstr_return, -1, 1, [1, 1, 0], False)
#         self.check_x(mip_cnstr_return, -3, 3, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, -0.5, 0.5, [1, 0, 0], False)

#     def test8(self):
#         # x_up = 0 and binary_for_zero_input=True
#         dtype = torch.float64
#         mip_cnstr_return = utils.absolute_value_as_mixed_integer_constraint(
#             torch.tensor(-2, dtype=dtype),
#             torch.tensor(0, dtype=dtype),
#             binary_for_zero_input=True)
#         self.check_x(mip_cnstr_return, -2, 2, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, -1, 1, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [1, 0, 0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [0, 1, 0], True)
#         self.check_x(mip_cnstr_return, 0, 0, [1, 1, 0], False)
#         self.check_x(mip_cnstr_return, -3, 3, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, 1, 1, [0, 0, 1], False)
#         self.check_x(mip_cnstr_return, 0, 1, [1, 0, 0], False)
#         self.check_x(mip_cnstr_return, 0, 1, [0, 1, 0], False)


# class TestMaxAsMixedIntegerConstraint(unittest.TestCase):
#     def constraint_tester(self, x_lo, x_up):
#         dtype = x_lo.dtype
#         ret = utils.max_as_mixed_integer_constraint(x_lo, x_up)
#         np.testing.assert_allclose(ret.Aout_slack.detach().numpy(),
#                                    np.array([[1.]]))
#         prog = gurobi_torch_mip.GurobiTorchMIP(dtype)
#         nx = x_lo.shape[0]
#         x_var = prog.addVars(nx, lb=-gurobipy.GRB.INFINITY)
#         y = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
#         slack, binary = prog.add_mixed_integer_linear_constraints(
#             ret, x_var, y, "", "", "", "", "")
#         self.assertEqual(len(slack), 1)
#         prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
#         x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
#         for i in range(x_samples.shape[0]):
#             for j in range(nx):
#                 x_var[j].lb = x_samples[i, j].item()
#                 x_var[j].ub = x_samples[i, j].item()
#             prog.gurobi_model.optimize()
#             self.assertEqual(prog.gurobi_model.status,
#                              gurobipy.GRB.Status.OPTIMAL)
#             self.assertAlmostEqual(y[0].x, torch.max(x_samples[i]).item())
#             binary_val = np.array([v.x for v in binary])
#             self.assertAlmostEqual(np.sum(binary_val), 1)
#             self.assertAlmostEqual(binary_val @ x_samples[i].detach().numpy(),
#                                    y[0].x)
#             self.assertEqual(y[0].x, slack[0].x)

#     def test1(self):
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-2, -3], dtype=dtype),
#                                torch.tensor([1, 4], dtype=dtype))

#     def test2(self):
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-1, 3, -2], dtype=dtype),
#                                torch.tensor([4, 4, 3], dtype=dtype))

#     def test3(self):
#         # Some variable's upper bound is smaller than other's lower bound
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
#                                torch.tensor([2, 5, 4, 2], dtype=dtype))
#         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
#                                torch.tensor([2, 5, 2, 2], dtype=dtype))

# class TestLinfinityGradient(unittest.TestCase):
#     def test1(self):
#         dtype = torch.float64
#         grad = utils.l_infinity_gradient(torch.tensor([1, 2], dtype=dtype))
#         self.assertEqual(grad.dtype, dtype)
#         np.testing.assert_allclose(grad.detach().numpy(), np.array([[0, 1]]))
#         grad = utils.l_infinity_gradient(torch.tensor([-2, -1, 1],
#                                                       dtype=dtype))
#         np.testing.assert_allclose(grad.detach().numpy(), np.array([[-1, 0,
#                                                                      0]]))

#     def gradient_tester(self, x, max_tol, grad_expected):
#         grad = utils.l_infinity_gradient(x, max_tol=max_tol)
#         self.assertEqual(grad.dtype, x.dtype)
#         self.assertEqual(grad.shape, grad_expected.shape)
#         # Now check if grad is a shuffled version of grad_expected
#         for i in range(grad.shape[0]):
#             self.assertTrue(
#                 torch.any(
#                     torch.norm(grad[i].repeat(grad.shape[0], 1) -
#                                grad_expected,
#                                dim=1) < 1E-10))

#     def test2(self):
#         # Multiple entry in x has the maximal absolute value.
#         dtype = torch.float64
#         self.gradient_tester(torch.tensor([1, 1, 0], dtype=dtype),
#                              max_tol=0.,
#                              grad_expected=torch.tensor([[1, 0, 0], [0, 1, 0]],
#                                                         dtype=dtype))
#         self.gradient_tester(torch.tensor([-2, 0, 2], dtype=dtype),
#                              max_tol=0.,
#                              grad_expected=torch.tensor(
#                                  [[-1, 0, 0], [0, 0, 1]], dtype=dtype))

#         self.gradient_tester(torch.tensor([2.1, 0, 2, 1], dtype=dtype),
#                              max_tol=0.5,
#                              grad_expected=torch.tensor(
#                                  [[1, 0, 0, 0], [0, 0, 1, 0]], dtype=dtype))

#     def test3(self):
#         dtype = torch.float64
#         self.gradient_tester(torch.tensor([0, 0], dtype=dtype),
#                              max_tol=0.,
#                              grad_expected=torch.tensor(
#                                  [[1, 0], [0, 1], [-1, 0], [0, -1]],
#                                  dtype=dtype))
#         self.gradient_tester(torch.tensor([0.1, -0.1], dtype=dtype),
#                              max_tol=0.2,
#                              grad_expected=torch.tensor(
#                                  [[1, 0], [0, 1], [-1, 0], [0, -1]],
#                                  dtype=dtype))

# class TestMaxAsMixedIntegerConstraint(unittest.TestCase):
#     def constraint_tester(self, x_lo, x_up):
#         dtype = x_lo.dtype
#         ret = monotonic_utils.l_inf_as_mixed_integer_constraint(x_lo, x_up)
#         np.testing.assert_allclose(ret.Aout_slack.detach().numpy(),
#                                    np.array([[1.]]))
#         prog = gurobi_torch_mip.GurobiTorchMIP(dtype)
#         nx = x_lo.shape[0]
#         x_var = prog.addVars(nx, lb=-gurobipy.GRB.INFINITY)
#         y = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
#         slack, binary = prog.add_mixed_integer_linear_constraints(
#             ret, x_var, y, "", "", "", "", "")
#         self.assertEqual(len(slack), 1)
#         prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
#         x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
#         x_samples_plus_minus = torch.cat((x_samples,-x_samples),dim=1)
#         for i in range(x_samples.shape[0]):
#             for j in range(nx):
#                 x_var[j].lb = x_samples[i, j].item()
#                 x_var[j].ub = x_samples[i, j].item()
#             prog.gurobi_model.optimize()
#             self.assertEqual(prog.gurobi_model.status,
#                              gurobipy.GRB.Status.OPTIMAL)
#             self.assertAlmostEqual(y[0].x, 
#                                    torch.max(x_samples_plus_minus[i]).item())
#             binary_val = np.array([v.x for v in binary])
#             self.assertAlmostEqual(np.sum(binary_val), 1)
#             self.assertAlmostEqual(binary_val @ x_samples_plus_minus[i].detach().numpy(),
#                                    y[0].x)
#             self.assertEqual(y[0].x, slack[0].x)

#     def test1(self):
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-2, -3], dtype=dtype),
#                                torch.tensor([1, 4], dtype=dtype))

#     def test2(self):
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-1, 3, -2], dtype=dtype),
#                                torch.tensor([4, 4, 3], dtype=dtype))

#     def test3(self):
#         # Some variable's upper bound is smaller than other's lower bound
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
#                                torch.tensor([2, 5, 4, 2], dtype=dtype))
#         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
#                                torch.tensor([2, 5, 2, 2], dtype=dtype))


# class TestLinfAsMixedIntegerConstraint(unittest.TestCase):
#     def constraint_tester(self, x_lo, x_up, x_eqlm):
        
#         dtype = x_lo.dtype
#         L = 2
#         M = torch.eye(x_lo.shape[0],dtype=dtype)
#         M = torch.vstack((M,torch.sum(M,dim=0)))
#         lx_lb, lx_ub = mip_utils.compute_range_by_IA(
#             M*L, -M@x_eqlm*L, x_lo,
#             x_up)
        
#         ret = monotonic_utils.l_inf_generalized_as_mixed_integer_constraint(lx_lb, lx_ub,M)
#         ret.transform_input(M*L, 
#                             -M@x_eqlm*L)
#         np.testing.assert_allclose(ret.Aout_slack.detach().numpy(),
#                                    np.array([[1.]]))
#         prog = gurobi_torch_mip.GurobiTorchMIP(dtype)
#         nx = x_lo.shape[0]
#         x_var = prog.addVars(nx, lb=-gurobipy.GRB.INFINITY)
#         y = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
#         slack, binary = prog.add_mixed_integer_linear_constraints(
#             ret, x_var, y, "", "", "", "", "")
#         self.assertEqual(len(slack), 1)
#         prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
#         x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
#         x_eqlm_stack = x_eqlm.reshape((1,nx)).repeat(x_samples.shape[0],1)
#         x_samples_plus_minus = torch.cat((x_samples-x_eqlm_stack,-x_samples+x_eqlm_stack),dim=1)
#         for i in range(x_samples.shape[0]):
#             for j in range(nx):
#                 x_var[j].lb = x_samples[i, j].item()
#                 x_var[j].ub = x_samples[i, j].item()
#             prog.gurobi_model.optimize()
#             self.assertEqual(prog.gurobi_model.status,
#                              gurobipy.GRB.Status.OPTIMAL)
#             self.assertAlmostEqual(y[0].x, 
#                                    torch.max(x_samples_plus_minus[i]).item())
#             binary_val = np.array([v.x for v in binary])
#             self.assertAlmostEqual(np.sum(binary_val), 1)
#             self.assertAlmostEqual(binary_val @ x_samples_plus_minus[i].detach().numpy(),
#                                    y[0].x)
#             self.assertEqual(y[0].x, slack[0].x)

#     def test1(self):
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-2, -3], dtype=dtype),
#                                torch.tensor([1, 4], dtype=dtype),
#                                torch.tensor([0.5, 1], dtype=dtype))

#     def test2(self):
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-1, 3, -2], dtype=dtype),
#                                torch.tensor([4, 4, 3], dtype=dtype),
#                                torch.tensor([0, 0, 0], dtype=dtype))

#     def test3(self):
#         # Some variable's upper bound is smaller than other's lower bound
#         dtype = torch.float64
#         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
#                                torch.tensor([2, 5, 4, 2], dtype=dtype),
#                                torch.tensor([0, 0, 0, 0], dtype=dtype))
#         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
#                                torch.tensor([2, 5, 2, 2], dtype=dtype),
#                                torch.tensor([0, 0, 0, 0], dtype=dtype))

# # class TestMaxTransformedAsMixedIntegerConstraint(unittest.TestCase):
# #     def constraint_tester(self, x_lo, x_up, x_eqlm):
# #         dtype = x_lo.dtype
# #         ret = monotonic_utils.l_inf_transformed_as_mixed_integer_constraint(x_lo, x_up, x_eqlm)
# #         np.testing.assert_allclose(ret.Aout_slack.detach().numpy(),
# #                                    np.array([[1.]]))
# #         prog = gurobi_torch_mip.GurobiTorchMIP(dtype)
# #         nx = x_lo.shape[0]
# #         x_var = prog.addVars(nx, lb=-gurobipy.GRB.INFINITY)
# #         y = prog.addVars(1, lb=-gurobipy.GRB.INFINITY)
# #         slack, binary = prog.add_mixed_integer_linear_constraints(
# #             ret, x_var, y, "", "", "", "", "")
# #         self.assertEqual(len(slack), 1)
# #         prog.gurobi_model.setParam(gurobipy.GRB.Param.OutputFlag, False)
# #         x_samples = utils.uniform_sample_in_box(x_lo, x_up, 100)
# #         x_eqlm_stack = x_eqlm.reshape((1,nx)).repeat(x_samples.shape[0],1)
# #         x_samples_plus_minus = torch.cat((x_samples-x_eqlm_stack,-x_samples+x_eqlm_stack),dim=1)
# #         for i in range(x_samples.shape[0]):
# #             for j in range(nx):
# #                 x_var[j].lb = x_samples[i, j].item()
# #                 x_var[j].ub = x_samples[i, j].item()
# #             prog.gurobi_model.optimize()
# #             self.assertEqual(prog.gurobi_model.status,
# #                              gurobipy.GRB.Status.OPTIMAL)
# #             self.assertAlmostEqual(y[0].x, 
# #                                    torch.max(x_samples_plus_minus[i]).item())
# #             binary_val = np.array([v.x for v in binary])
# #             self.assertAlmostEqual(np.sum(binary_val), 1)
# #             self.assertAlmostEqual(binary_val @ x_samples_plus_minus[i].detach().numpy(),
# #                                    y[0].x)
# #             self.assertEqual(y[0].x, slack[0].x)

# #     def test1(self):
# #         dtype = torch.float64
# #         self.constraint_tester(torch.tensor([-2, -3], dtype=dtype),
# #                                torch.tensor([1, 4], dtype=dtype),
# #                                torch.tensor([0.5, 1], dtype=dtype))

# #     def test2(self):
# #         dtype = torch.float64
# #         self.constraint_tester(torch.tensor([-1, 3, -2], dtype=dtype),
# #                                torch.tensor([4, 4, 3], dtype=dtype),
# #                                torch.tensor([0, 0, 0], dtype=dtype))

# #     def test3(self):
# #         # Some variable's upper bound is smaller than other's lower bound
# #         dtype = torch.float64
# #         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
# #                                torch.tensor([2, 5, 4, 2], dtype=dtype),
# #                                torch.tensor([0, 0, 0, 0], dtype=dtype))
# #         self.constraint_tester(torch.tensor([-1, 3, -2, 1], dtype=dtype),
# #                                torch.tensor([2, 5, 2, 2], dtype=dtype),
# #                                torch.tensor([0, 0, 0, 0], dtype=dtype))


if __name__ == "__main__":
    unittest.main()

# %%
