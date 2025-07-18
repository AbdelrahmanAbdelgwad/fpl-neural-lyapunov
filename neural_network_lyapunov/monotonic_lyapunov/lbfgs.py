import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from copy import deepcopy
from torch.optim import Optimizer
import gurobipy


def is_legal(v):
    """
    Checks that tensor is not NaN or Inf.

    Inputs:
        v (tensor): tensor to be checked

    """
    legal = not torch.isnan(v).any() and not torch.isinf(v)

    return legal


def polyinterp(points, x_min_bound=None, x_max_bound=None, plot=False):
    """
    Gives the minimizer and minimum of the interpolating polynomial over given points
    based on function and derivative information. Defaults to bisection if no critical
    points are valid.

    Based on polyinterp.m Matlab function in minFunc by Mark Schmidt with some slight
    modifications.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 12/6/18.

    Inputs:
        points (nparray): two-dimensional array with each point of form [x f g]
        x_min_bound (float): minimum value that brackets minimum (default: minimum of points)
        x_max_bound (float): maximum value that brackets minimum (default: maximum of points)
        plot (bool): plot interpolating polynomial

    Outputs:
        x_sol (float): minimizer of interpolating polynomial
        F_min (float): minimum of interpolating polynomial

    Note:
      . Set f or g to np.nan if they are unknown

    """
    no_points = points.shape[0]
    order = np.sum(1 - np.isnan(points[:, 1:3]).astype('int')) - 1

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])

    # compute bounds of interpolation area
    if x_min_bound is None:
        x_min_bound = x_min
    if x_max_bound is None:
        x_max_bound = x_max

    # explicit formula for quadratic interpolation
    if no_points == 2 and order == 2 and plot is False:
        # Solution to quadratic interpolation is given by:
        # a = -(f1 - f2 - g1(x1 - x2))/(x1 - x2)^2
        # x_min = x1 - g1/(2a)
        # if x1 = 0, then is given by:
        # x_min = - (g1*x2^2)/(2(f2 - f1 - g1*x2))

        if points[0, 0] == 0:
            x_sol = -points[0, 2] * points[1, 0] ** 2 / (2 * (points[1, 1] - points[0, 1] - points[0, 2] * points[1, 0]))
        else:
            a = -(points[0, 1] - points[1, 1] - points[0, 2] * (points[0, 0] - points[1, 0])) / (points[0, 0] - points[1, 0]) ** 2
            x_sol = points[0, 0] - points[0, 2]/(2*a)

        x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)

    # explicit formula for cubic interpolation
    elif no_points == 2 and order == 3 and plot is False:
        # Solution to cubic interpolation is given by:
        # d1 = g1 + g2 - 3((f1 - f2)/(x1 - x2))
        # d2 = sqrt(d1^2 - g1*g2)
        # x_min = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2))
        d1 = points[0, 2] + points[1, 2] - 3 * ((points[0, 1] - points[1, 1]) / (points[0, 0] - points[1, 0]))
        d2 = np.sqrt(d1 ** 2 - points[0, 2] * points[1, 2])
        if np.isreal(d2):
            x_sol = points[1, 0] - (points[1, 0] - points[0, 0]) * ((points[1, 2] + d2 - d1) / (points[1, 2] - points[0, 2] + 2 * d2))
            x_sol = np.minimum(np.maximum(x_min_bound, x_sol), x_max_bound)
        else:
            x_sol = (x_max_bound + x_min_bound)/2

    # solve linear system
    else:
        # define linear constraints
        A = np.zeros((0, order + 1))
        b = np.zeros((0, 1))

        # add linear constraints on function values
        for i in range(no_points):
            if not np.isnan(points[i, 1]):
                constraint = np.zeros((1, order + 1))
                for j in range(order, -1, -1):
                    constraint[0, order - j] = points[i, 0] ** j
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 1])

        # add linear constraints on gradient values
        for i in range(no_points):
            if not np.isnan(points[i, 2]):
                constraint = np.zeros((1, order + 1))
                for j in range(order):
                    constraint[0, j] = (order - j) * points[i, 0] ** (order - j - 1)
                A = np.append(A, constraint, 0)
                b = np.append(b, points[i, 2])

        # check if system is solvable
        if A.shape[0] != A.shape[1] or np.linalg.matrix_rank(A) != A.shape[0]:
            x_sol = (x_min_bound + x_max_bound)/2
            f_min = np.Inf
        else:
            # solve linear system for interpolating polynomial
            coeff = np.linalg.solve(A, b)

            # compute critical points
            dcoeff = np.zeros(order)
            for i in range(len(coeff) - 1):
                dcoeff[i] = coeff[i] * (order - i)

            crit_pts = np.array([x_min_bound, x_max_bound])
            crit_pts = np.append(crit_pts, points[:, 0])

            if not np.isinf(dcoeff).any():
                roots = np.roots(dcoeff)
                crit_pts = np.append(crit_pts, roots)

            # test critical points
            f_min = np.Inf
            x_sol = (x_min_bound + x_max_bound) / 2 # defaults to bisection
            for crit_pt in crit_pts:
                if np.isreal(crit_pt) and crit_pt >= x_min_bound and crit_pt <= x_max_bound:
                    F_cp = np.polyval(coeff, crit_pt)
                    if np.isreal(F_cp) and F_cp < f_min:
                        x_sol = np.real(crit_pt)
                        f_min = np.real(F_cp)

            if(plot):
                plt.figure()
                x = np.arange(x_min_bound, x_max_bound, (x_max_bound - x_min_bound)/10000)
                f = np.polyval(coeff, x)
                plt.plot(x, f)
                plt.plot(x_sol, f_min, 'x')

    return x_sol


class LBFGS(Optimizer):
    """
    Implements the L-BFGS algorithm. Compatible with multi-batch and full-overlap
    L-BFGS implementations and (stochastic) Powell damping. Partly based on the 
    original L-BFGS implementation in PyTorch, Mark Schmidt's minFunc MATLAB code, 
    and Michael Overton's weak Wolfe line search MATLAB code.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 10/20/20.

    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    """

    def __init__(self, params, lr=0.01, history_size=10, line_search='Wolfe',
                 dtype=torch.float, debug=False):

        # ensure inputs are valid
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= history_size:
            raise ValueError("Invalid history size: {}".format(history_size))
        if line_search not in ['Armijo', 'Wolfe', 'None']:
            raise ValueError("Invalid line search: {}".format(line_search))

        defaults = dict(lr=lr, history_size=history_size, line_search=line_search, dtype=dtype, debug=debug)
        super(LBFGS, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("L-BFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        state = self.state['global_state']
        state.setdefault('n_iter', 0)
        state.setdefault('curv_skips', 0)
        state.setdefault('fail_skips', 0)
        state.setdefault('H_diag',1)
        state.setdefault('fail', True)

        state['old_dirs'] = []
        state['old_stps'] = []

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)
    
    def _gather_flat_grad_custom(self):
        views_V = []
        views_linf = []
        for p in self._params:
            if p.grad is None:
                view_V = p.data.new(p.data.numel()).zero_()
                view_linf = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view_V = p.grad_V.data.to_dense().view(-1)
                view_linf = p.grad_linf.data.to_dense().view(-1)
            else:
                view_V = p.grad_V.data.view(-1)
                view_linf = p.grad_linf.data.view(-1)
            views_V.append(view_V)
            views_linf.append(view_linf)
        return torch.cat(views_V, 0),torch.cat(views_linf, 0)
    
    def cal_search_dir_lp(self):
        views = []
        for p in self._params:
            grad_V = p.grad_V.data.clone().reshape(-1).detach().numpy()
            grad_linf = p.grad_linf.data.clone().reshape(-1).detach().numpy()
            model = gurobipy.Model()
            u = model.addMVar(len(grad_V),
                                lb=-gurobipy.GRB.INFINITY,
                                vtype=gurobipy.GRB.CONTINUOUS,
                                name="u")
            s = model.addMVar(1,
                                lb=0.,
                                vtype=gurobipy.GRB.CONTINUOUS,
                                name="s")
            model.addConstr(grad_V @ u == s, 
                            name="grad_V")
            model.addConstr(grad_linf @ u >= 0., 
                        name="grad_linf")
            model.addConstr(u @ u <= 1.0, 
                        name="norm_u")
            model.setObjective(grad_linf @ u + 0.001*s, gurobipy.GRB.MAXIMIZE)
            model.setParam(gurobipy.GRB.Param.OutputFlag, False)
            model.optimize()
            u_opt = u.X
            u_opt_torch = torch.tensor(u_opt,dtype=p.data.dtype).view(-1)
            # if len(p.data.shape)>=2:
            #     u_opt_torch = torch.tensor(u_opt.reshape(p.data.shape[0],p.data.shape[1]),dtype=p.data.dtype)
            # else:
            #     u_opt_torch = torch.tensor(u_opt.reshape(p.data.shape[0],),dtype=p.data.dtype)
            # print('PGD check - grad_V @ u_opt: ',grad_V @ u_opt, ", grad_linf @ u_opt: ",grad_linf @ u_opt)
            views.append(u_opt_torch)
        return -torch.cat(views, 0)

    def _add_update(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(update[offset:offset + numel].view_as(p.data), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _copy_params(self):
        current_params = []
        for param in self._params:
            current_params.append(deepcopy(param.data))
        return current_params

    def _load_params(self, current_params):
        i = 0
        for param in self._params:
            param.data[:] = current_params[i]
            i += 1

    def line_search(self, line_search):
        """
        Switches line search option.
        
        Inputs:
            line_search (str): designates line search to use
                Options:
                    'None': uses steplength designated in algorithm
                    'Armijo': uses Armijo backtracking line search
                    'Wolfe': uses Armijo-Wolfe bracketing line search
        
        """
        
        group = self.param_groups[0]
        group['line_search'] = line_search
        
        return

    def _step(self, p_k, g_Ok, g_Sk=None, options=None):
        """
        Performs a single optimization step.

        Inputs:
            p_k (tensor): 1-D tensor specifying search direction
            g_Ok (tensor): 1-D tensor of flattened gradient over overlap O_k used
                            for gradient differencing in curvature pair update
            g_Sk (tensor): 1-D tensor of flattened gradient over full sample S_k
                            used for curvature pair damping or rejection criterion,
                            if None, will use g_Ok (default: None)
            options (dict): contains options for performing line search (default: None)

        Options for Armijo backtracking line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'max_ls' (int): maximum number of line search steps permitted (default: 10)

        Outputs (depends on line search):
          . No line search:
                t (float): steplength
          . Armijo backtracking line search:
                F_new (tensor): loss function at new iterate
                t (tensor): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded

        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

        """

        if options is None:
            options = {}
        assert len(self.param_groups) == 1

        # load parameter options
        group = self.param_groups[0]
        lr = 0.01#group['lr']
        line_search = group['line_search']
        dtype = group['dtype']
        debug = group['debug']

        # variables cached in state (for tracing)
        state = self.state['global_state']
        d = state.get('d')
        t = state.get('t')
        prev_flat_grad = state.get('prev_flat_grad')
        Bs = state.get('Bs')

        # keep track of nb of iterations
        state['n_iter'] += 1

        # set search direction
        d = p_k

        # modify previous gradient
        if prev_flat_grad is None:
            prev_flat_grad = g_Ok.clone()
        else:
            prev_flat_grad.copy_(g_Ok)

        # set initial step size
        t = lr

        # closure evaluation counter
        closure_eval = 0

        if g_Sk is None:
            g_Sk = g_Ok.clone()

        # perform Armijo backtracking line search
        if line_search == 'Armijo':

            # load options
            if options:
                closure = options['closure']
                gtd = g_Sk.dot(d)   # 'gtd' (tensor): inner product g_Ok'd in line search (default: g_Ok'd)
                F_k, _,_ = closure()     # current loss
                closure_eval += 1   
                eta = 2.            # 'eta' (tensor): factor for decreasing steplength > 0 (default: 2)
                c1 = 1e-4           # 'c1' (float): sufficient decrease constant in (0, 1) (default: 1e-4)
                interpolate = False  # 'interpolate' (bool): flag for using interpolation (default: True)
                inplace = False      # 'inplace' (bool): flag for inplace operations (default: True)
                ls_debug = False    # 'ls_debug' (bool): debugging mode for line search
                if 'max_ls' not in options.keys():
                    max_ls = 10
                elif options['max_ls'] <= 0:
                    raise(ValueError('Invalid max_ls; must be positive.'))
                else:
                    max_ls = options['max_ls']
            else:
                raise(ValueError('Options are not specified; need closure evaluating function.'))

            # initialize values
            if interpolate:
                if torch.cuda.is_available():
                    F_prev = torch.tensor(np.nan, dtype=dtype).cuda()
                else:
                    F_prev = torch.tensor(np.nan, dtype=dtype)

            ls_step = 0
            t_prev = 0 # old steplength
            fail = False # failure flag

            # begin print for debug mode
            if ls_debug:
                print('==================================== Begin Armijo line search ===================================')
                print('F(x): %.8e  g*d: %.8e' % (F_k, gtd))

            # check if search direction is descent direction
            if gtd >= 0:
                desc_dir = False
                if debug:
                    print('Not a descent direction!')
            else:
                desc_dir = True

            # store values if not in-place
            if not inplace:
                current_params = self._copy_params()

            # update and evaluate at new point
            self._add_update(t, d)
            F_new,_,_ = closure()
            closure_eval += 1

            # print info if debugging
            if ls_debug:
                print('LS Step: %d  t: %.8e  F(x+td): %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                      % (ls_step, t, F_new, F_k + c1 * t * gtd, F_k))

            # check Armijo condition
            while (F_new > F_k + c1*t*gtd and F_new>=2e-6) or not is_legal(F_new):

                # check if maximum number of iterations reached
                if ls_step >= max_ls:
                    if inplace:
                        self._add_update(-t, d)
                    else:
                        self._load_params(current_params)

                    t = 0
                    F_new,_,_ = closure()
                    closure_eval += 1
                    fail = True
                    break

                else:
                    # store current steplength
                    t_new = t

                    # compute new steplength

                    # if first step or not interpolating, then multiply by factor
                    if ls_step == 0 or not interpolate or not is_legal(F_new):
                        t = t/eta

                    # if second step, use function value at new point along with 
                    # gradient and function at current iterate
                    elif ls_step == 1 or not is_legal(F_prev):
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan]]))

                    # otherwise, use function values at new point, previous point,
                    # and gradient and function at current iterate
                    else:
                        t = polyinterp(np.array([[0, F_k.item(), gtd.item()], [t_new, F_new.item(), np.nan], 
                                                [t_prev, F_prev.item(), np.nan]]))

                    # if values are too extreme, adjust t
                    if interpolate:
                        if t < 1e-3 * t_new:
                            t = 1e-3 * t_new
                        elif t > 0.6 * t_new:
                            t = 0.6 * t_new

                        # store old point
                        F_prev = F_new
                        t_prev = t_new

                    # update iterate and reevaluate
                    if inplace:
                        self._add_update(t - t_new, d)
                    else:
                        self._load_params(current_params)
                        self._add_update(t, d)

                    F_new, _,_ = closure()
                    closure_eval += 1
                    ls_step += 1 # iterate
                    
                    # print info if debugging
                    if ls_debug:
                        print('LS Step: %d  t: %.8e  F(x+td):   %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                              % (ls_step, t, F_new, F_k + c1 * t * gtd, F_k))

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))
                
            # print final steplength
            if ls_debug:
                print('Final Steplength:', t)
                print('===================================== End Armijo line search ====================================')

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = fail

            return F_new, t, ls_step, closure_eval, desc_dir, fail

        else:

            # perform update
            self._add_update(t, d)

            # store Bs
            if Bs is None:
                Bs = (g_Sk.mul(-t)).clone()
            else:
                Bs.copy_(g_Sk.mul(-t))

            state['d'] = d
            state['prev_flat_grad'] = prev_flat_grad
            state['t'] = t
            state['Bs'] = Bs
            state['fail'] = False

            return t
        
    def step(self, p_k, g_Ok, g_Sk=None, options={}):
        return self._step(p_k, g_Ok, g_Sk, options)


class FullBatchLBFGS(LBFGS):
    """
    Implements full-batch or deterministic L-BFGS algorithm. Compatible with
    Powell damping. Can be used when evaluating a deterministic function and
    gradient. Wraps the LBFGS optimizer. Performs the two-loop recursion,
    updating, and curvature updating in a single step.

    Implemented by: Hao-Jun Michael Shi and Dheevatsa Mudigere
    Last edited 11/15/18.

    Warnings:
      . Does not support per-parameter options and parameter groups.
      . All parameters have to be on a single device.

    Inputs:
        lr (float): steplength or learning rate (default: 1)
        history_size (int): update history size (default: 10)
        line_search (str): designates line search to use (default: 'Wolfe')
            Options:
                'None': uses steplength designated in algorithm
                'Armijo': uses Armijo backtracking line search
                'Wolfe': uses Armijo-Wolfe bracketing line search
        dtype: data type (default: torch.float)
        debug (bool): debugging mode

    """

    def __init__(self, params, lr=1, history_size=10, line_search='Wolfe', 
                 dtype=torch.float, debug=False):
        super(FullBatchLBFGS, self).__init__(params, lr, history_size, line_search, 
             dtype, debug)

    def step(self, options=None):
        """
        Performs a single optimization step.

        Inputs:
            options (dict): contains options for performing line search (default: None)
            
        Options for Armijo backtracking line search:
            'closure' (callable): reevaluates model and returns function value
            'current_loss' (tensor): objective value at current iterate (default: F(x_k))
            'max_ls' (int): maximum number of line search steps permitted (default: 10)

        Outputs (depends on line search):
          . No line search:
                t (float): steplength
          . Armijo backtracking line search:
                F_new (tensor): loss function at new iterate
                t (tensor): final steplength
                ls_step (int): number of backtracks
                closure_eval (int): number of closure evaluations
                desc_dir (bool): descent direction flag
                    True: p_k is descent direction with respect to the line search
                    function
                    False: p_k is not a descent direction with respect to the line
                    search function
                fail (bool): failure flag
                    True: line search reached maximum number of iterations, failed
                    False: line search succeeded

        Notes:
          . If encountering line search failure in the deterministic setting, one
            should try increasing the maximum number of line search steps max_ls.

        """
        
        
        # gather gradient
        grad_V,grad_linf = self._gather_flat_grad_custom()
        
        # update curvature if after 1st iteration
        # state = self.state['global_state']
        # if state['n_iter'] > 0:
        #     self.curvature_update(grad, eps, damping)

        # # compute search direction
        # p = self.two_loop_recursion(-grad)
        p = self.cal_search_dir_lp()

        # take step
        return self._step(p, grad_V, options=options)