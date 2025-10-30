import torch

import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import gurobipy

import neural_network_lyapunov.relu_to_optimization as relu_to_optimization
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.mip_utils as mip_utils
import neural_network_lyapunov.gurobi_torch_mip as gurobi_torch_mip


class Cart_Pole:
    def __init__(self, dtype):
        self.dtype = dtype
        self.M = 1
        self.m = 0.1    
        self.l = 1
        self.g = -9.81
        

    def dynamics(self, x, u):
        """
        Compute the continuous-time dynamics
        """
        q = x[:2]
        qdot = x[2:]
        if isinstance(x, np.ndarray):
            M_q = np.array([[self.M+self.m, -self.m*self.l*np.cos(q[1])],
                            [-self.m*self.l*np.cos(q[1]),self.m*(self.l**2)]])
            C_q = np.array([[0, self.m*self.l*qdot[1]*np.sin(q[1])],
                            [0, 0]])
            t_q =np.array([[0],[-self.m*self.g*self.l*np.sin(q[1])]])
            B = np.array([[1],[0]])
            # u_tmp = (t_q + B*u[0] - C_q@q)
            # print(t_q.shape,(B*u[0]).shape,(C_q@q[:,None]).shape)
            qddot = np.linalg.inv(M_q)@(t_q + B*u[0] - C_q@q[:,None])
            return np.concatenate((qdot, qddot[:,0]))
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            M_q = np.array([[self.M+self.m, -self.m*self.l*torch.cos(q[1])],
                            [-self.m*self.l*torch.cos(q[1]),self.m*(self.l**2)]])
            M_q_inv = torch.tensor(np.linalg.inv(M_q),dtype=dtype)
            C_q = torch.tensor([[0, self.m*self.l*qdot[1]*torch.sin(q[1])],
                            [0, 0]],dtype=dtype)
            t_q =torch.tensor([[0],[-self.m*self.g*self.l*torch.sin(q[1])]],dtype=dtype)
            B = torch.tensor([[1],[0]],dtype=dtype)
            qddot = M_q_inv@(t_q + B*u[0] - C_q@q[:,None])
            return torch.cat((qdot, qddot[:,0]))

    def linearized_dynamics(self, x, u):
        """
        Return ∂ẋ/∂x and ∂ẋ/∂ u
        """
        q = x[:2]
        qdot = x[2:]
        if isinstance(x, np.ndarray):
            M_q = np.array([[self.M+self.m, -self.m*self.l*np.cos(q[1])],
                            [-self.m*self.l*np.cos(q[1]),self.m*(self.l**2)]])
            M_q_inv = np.linalg.inv(M_q)
            t_q_der =np.array([[0,0],
                               [0,-self.m*self.g*self.l*np.cos(q[1])]])
            B_q = np.array([[1],[0]])
            
            A = np.zeros((4, 4))
            B = np.zeros((4, 1))
            A[:2, 2:4] = np.eye(2)
            A[2:4,:2] = M_q_inv@t_q_der
            B[2:4,:] = M_q_inv@B_q
            return A, B
        elif isinstance(x, torch.Tensor):
            dtype = x.dtype
            M_q = np.array([[self.M+self.m, -self.m*self.l*np.cos(q[1])],
                            [-self.m*self.l*np.cos(q[1]),self.m*(self.l**2)]])
            M_q_inv = torch.tensor(np.linalg.inv(M_q),dtype=dtype)
            t_q_der = torch.tensor([[0,0],
                               [0,-self.m*self.g*self.l*np.cos(q[1])]],dtype=dtype)
            B_q = torch.tensor([[1],[0]],dtype=dtype)
            A = torch.zeros((4,4), dtype=dtype)
            B = torch.zeros((4,1), dtype=dtype)
            A[:2, 2:4] = torch.eye(2, dtype=dtype)
            A[2:4,:2] = M_q_inv@t_q_der
            B[2:4,:] = M_q_inv@B_q
            return A, B

    

    # def lqr_control(self, Q, R, x, u):
    #     """
    #     The control action should be u = K * (x - x*) + u*
    #     """
    #     x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
    #     u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
    #     A, B = self.linearized_dynamics(x_np, u_np)
    #     S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    #     K = -np.linalg.solve(R, B.T @ S)
    #     # A, B = self.dynamics_gradient(
    #     #     torch.tensor([np.pi, 0], dtype=self.dtype))
    #     # S = scipy.linalg.solve_continuous_are(A.detach().numpy(),
    #     #                                       B.detach().numpy(), Q, R)
    #     # K = -np.linalg.solve(R, B.T @ S)
    #     # return K
    #     return K, S

    def next_pose(self, x, u, dt):
        """
        Computes the next pose of the car after dt.
        """
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        u_np = u.detach().numpy() if isinstance(u, torch.Tensor) else u
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u_np), [0, dt], x_np)
        return result.y[:, -1]
    
    # def dynamics(self, x, u):
    #     """
    #     Compute the continuous-time dynamics
    #     """
    #     q = x[:2]
    #     qdot = x[2:]
    #     if isinstance(x, np.ndarray):
    #         tmp1 = self.l* (qdot[1]**2) - self.g*np.cos(q[1])
    #         tmp2 = self.m*self.l*(qdot[1]**2)*np.cos(q[1])*np.sin(q[1])
    #         qddot = np.array([
    #             (u[0] + self.m*np.sin(q[1])*tmp1)/ \
    #                     (self.M+self.m* (np.sin(q[1])**2)),
    #             (-u[0]*np.cos(q[1]) - tmp2 + \
    #                 (self.M+self.m)*self.g*np.sin(q[1]))/ \
    #                     (self.l*(self.M+self.m*np.sin(q[1])**2))
    #             ])
    #         return np.concatenate((qdot, qddot))
    #     elif isinstance(x, torch.Tensor):
    #         tmp1 = self.l* (qdot[1]**2) - self.g*torch.cos(q[1])
    #         tmp2 = self.m*self.l*(qdot[1]**2)*torch.cos(q[1])*np.sin(q[1])
    #         qddot = torch.tensor([
    #             (u[0] + self.m*torch.sin(q[1])*tmp1)/ \
    #                     (self.M+self.m* (torch.sin(q[1])**2)),
    #             (-u[0]*torch.cos(q[1]) - tmp2 + \
    #                 (self.M+self.m)*self.g*torch.sin(q[1]))/ \
    #                     (self.l*(self.M+self.m*torch.sin(q[1])**2))
    #             ])
    #         return torch.cat((qdot, qddot))

    # def linearized_dynamics(self, x, u):
    #     """
    #     Return ∂ẋ/∂x and ∂ẋ/∂ u
    #     """
    #     q = x[:2]
    #     qdot = x[2:]
    #     if isinstance(x, np.ndarray):
            
    #         A = np.zeros((4, 4))
    #         B = np.zeros((4, 1))
    #         A[0,1] = 1
    #         A[1,3] = 1
    #         A[2,2] = -self.m*self.g/self.M
    #         A[3,2] = (self.m+ self.M)*self.g/(self.M*self.l)
    #         B[[2],:] = 1/(self.M)
    #         B[[3],:] = -1/(self.M)
    #         return A, B
    #     elif isinstance(x, torch.Tensor):
    #         dtype = x.dtype
            
    #         A = torch.zeros((4,4), dtype=dtype)
    #         B = torch.zeros((4,1), dtype=dtype)
    #         A[0,1] = 1
    #         A[1,3] = 1
    #         A[2,2] = -self.m*self.g/self.M
    #         A[3,2] = (self.m+ self.M)*self.g/(self.M*self.l)
    #         B[[2],:] = 1/(self.M)
    #         B[[3],:] = -1/(self.M)
    #         return A, B
        


    
    def lqr_control(self, Q, R, x, u):
        """
        The control action should be u = K * (x - x*) + u*
        """
        x_np = x if isinstance(x, np.ndarray) else x.detach().numpy()
        u_np = u if isinstance(u, np.ndarray) else u.detach().numpy()
        A, B = self.linearized_dynamics(x_np, u_np)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)
        # A, B = self.dynamics_gradient(
        #     torch.tensor([np.pi, 0], dtype=self.dtype))
        # S = scipy.linalg.solve_continuous_are(A.detach().numpy(),
        #                                       B.detach().numpy(), Q, R)
        # K = -np.linalg.solve(R, B.T @ S)
        # return K
        return K, S

    
class PendulumVisualizer:
    def __init__(self, x0, figsize=(10, 10), subplot=111):
        """
        @param figsize The size of the fig
        @param subplot The argument in add_subplot(subplot) when adding the
        axis for pendulum.
        """
        self._plant = Pendulum(torch.float64)
        self._fig = plt.figure(figsize=figsize)
        self._pendulum_ax = self._fig.add_subplot(subplot)
        theta0 = x0[0]
        l_ = self._plant.length
        self._pendulum_arm, = self._pendulum_ax.plot(
            np.array([0, l_ * np.sin(theta0)]),
            np.array([0, -l_ * np.cos(theta0)]),
            linewidth=5)
        self._pendulum_sphere, = self._pendulum_ax.plot(l_ * np.sin(theta0),
                                                        -l_ * np.cos(theta0),
                                                        marker='o',
                                                        markersize=15)
        self._pendulum_ax.set_xlim(-l_ * 1.1, l_ * 1.1)
        self._pendulum_ax.set_ylim(-1.1 * l_, 1.1 * l_)
        self._pendulum_ax.set_axis_off()
        self._pendulum_title = self._pendulum_ax.set_title("t=0s")
        self._fig.canvas.draw()

    def draw(self, t, x):
        l_ = self._plant.length
        sin_theta = np.sin(x[0])
        cos_theta = np.cos(x[0])
        self._pendulum_arm.set_xdata(np.array([0, l_ * sin_theta]))
        self._pendulum_arm.set_ydata(np.array([0, -l_ * cos_theta]))
        self._pendulum_sphere.set_xdata(l_ * sin_theta)
        self._pendulum_sphere.set_ydata(-l_ * cos_theta)
        self._pendulum_title.set_text(f"t={t:.2f}s")
        self._fig.canvas.draw()


class PendulumReluContinuousTime:
    """
    The dynamics is theta_ddot = phi(theta, theta_dot, u) - phi(0, 0, 0)
    """
    def __init__(self, dtype, x_lo, x_up, u_lo, u_up, dynamics_relu):
        self.x_dim = 2
        self.dtype = dtype
        assert (x_lo.shape == (self.x_dim, ))
        assert (x_up.shape == (self.x_dim, ))
        self.x_lo = x_lo
        self.x_up = x_up
        self.u_dim = 1
        assert (u_lo.shape == (self.u_dim, ))
        assert (u_up.shape == (self.u_dim, ))
        self.u_lo = u_lo
        self.u_up = u_up
        assert (dynamics_relu[0].in_features == 3)
        assert (dynamics_relu[-1].out_features == 1)
        self.dynamics_relu = dynamics_relu
        self.x_equilibrium = torch.tensor([np.pi, 0], dtype=self.dtype)
        self.u_equilibrium = torch.tensor([0], dtype=self.dtype)
        self.dynamics_relu_free_pattern = relu_to_optimization.ReLUFreePattern(
            dynamics_relu, dtype)
        self.network_bound_propagate_method = \
            mip_utils.PropagateBoundsMethod.IA

    @property
    def x_lo_all(self):
        return self.x_lo.detach().numpy()

    @property
    def x_up_all(self):
        return self.x_up.detach().numpy()

    def mixed_integer_constraints(
            self,
            u_lo=None,
            u_up=None) -> gurobi_torch_mip.MixedIntegerConstraintsReturn:
        if u_lo is None:
            u_lo = self.u_lo
        if u_up is None:
            u_up = self.u_up
        network_input_lo = torch.cat((self.x_lo, u_lo))
        network_input_up = torch.cat((self.x_up, u_up))
        result = self.dynamics_relu_free_pattern.output_constraint(
            network_input_lo, network_input_up,
            self.network_bound_propagate_method)
        # Add the constraint xdot[0] = x[1]
        # xdot[1] = phi(x, u) - phi(x*, u*)
        result.Cout = torch.cat(
            (torch.tensor([0], dtype=self.dtype),
             result.Cout[0] - self.dynamics_relu(
                 torch.cat((self.x_equilibrium, self.u_equilibrium)))))
        assert (result.Aout_input is None)
        result.Aout_input = torch.tensor([[0, 1, 0], [0, 0, 0]],
                                         dtype=self.dtype)
        result.Aout_slack = torch.cat((torch.zeros(
            (1, result.num_slack()), dtype=self.dtype), result.Aout_slack),
                                      dim=0)
        if (result.Aout_binary is None):
            result.Aout_binary = torch.zeros((2, result.num_binary()),
                                             dtype=self.dtype)
        else:
            result.Aout_binary = torch.cat(
                (torch.zeros((1, result.num_binary()),
                             dtype=self.dtype), result.Aout_binary),
                dim=0)
        relu_at_equilibrium = self.dynamics_relu(
            torch.cat((self.x_equilibrium, self.u_equilibrium)))
        result.x_next_lb = torch.stack(
            (self.x_lo[1], result.nn_output_lo[0] - relu_at_equilibrium[0]))
        result.x_next_ub = torch.stack(
            (self.x_up[1], result.nn_output_up[0] - relu_at_equilibrium[0]))
        return result

    def step_forward(self, x_start, u_start):
        if len(x_start.shape) == 1:
            theta_ddot = self.dynamics_relu(torch.cat(
                (x_start, u_start))) - self.dynamics_relu(
                    torch.cat((self.x_equilibrium, self.u_equilibrium)))
            return torch.stack((x_start[1], theta_ddot[0]))
        else:
            theta_ddot = self.dynamics_relu(
                torch.cat((x_start, u_start), dim=1)) - self.dynamics_relu(
                    torch.cat((self.x_equilibrium, self.u_equilibrium)))
            return torch.cat((x_start[:, 1:], theta_ddot), dim=1)

    def possible_dx(self, x, u):
        assert (isinstance(x, torch.Tensor))
        assert (isinstance(u, torch.Tensor))
        return [self.step_forward(x, u)]

    def add_dynamics_constraint(
        self,
        mip,
        x_var,
        x_next_var,
        u_var,
        slack_var_name,
        binary_var_name,
        additional_u_lo: torch.Tensor = None,
        additional_u_up: torch.Tensor = None,
        binary_var_type=gurobipy.GRB.BINARY,
        u_input_prog: relu_system.ControlBoundProg = None
    ) -> relu_system.ReLUDynamicsConstraintReturn:
        return relu_system._add_forward_dynamics_mip_constraints(
            self, mip, x_var, x_next_var, u_var, slack_var_name,
            binary_var_name, additional_u_lo, additional_u_up, binary_var_type,
            u_input_prog)
