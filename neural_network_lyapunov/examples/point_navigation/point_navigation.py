import torch
import numpy as np
import scipy
import scipy.linalg

class Unicycle:
    def __init__(self, dtype):
        self.dtype = dtype
        self.v0 = 1.0  # constant forward speed
        self.l = 0.3  # offset (look-ahead) point

    def dynamics(self, x, u):
        """
        State: x = [x_pos, y_pos, theta]
        Control: u = [v, omega]
        """
        theta = x[2]
        if isinstance(x, np.ndarray):
            x_dot = u[0] * np.cos(theta) - self.l * u[1] * np.sin(theta)
            y_dot = u[0] * np.sin(theta) + self.l * u[1] * np.cos(theta)
            theta_dot = u[1]
            return np.array([x_dot, y_dot, theta_dot])
        elif isinstance(x, torch.Tensor):
            x_dot = u[0] * torch.cos(theta) - self.l * u[1] * torch.sin(theta)
            y_dot = u[0] * torch.sin(theta) + self.l * u[1] * torch.cos(theta)
            theta_dot = u[1]
            return torch.cat((x_dot.view(1), y_dot.view(1), theta_dot.view(1)))

    def dynamics_gradient(self, x):
        """
        Returns the gradient of the dynamics for offset unicycle
        Linearized around equilibrium: u = [v0, 0]
        """
        theta = x[2]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # A matrix: ∂f/∂x evaluated at u=[v0, 0]
        A = torch.tensor(
            [[0, 0, -self.v0 * sin_theta], 
            [0, 0, self.v0 * cos_theta], 
            [0, 0, 0]],
            dtype=self.dtype,
        )
        
        # B matrix: ∂f/∂u
        B = torch.tensor(
            [[cos_theta, -self.l * sin_theta], 
            [sin_theta, self.l * cos_theta], 
            [0, 1]], 
            dtype=self.dtype
        )
        
        return A, B

    def lqr_control(self, Q, R):
        """
        LQR control around equilibrium point.
        Returns the controller gain K.
        The control action should be u = K * (x - x_des)
        """
        # Linearize around origin with zero heading
        A, B = self.dynamics_gradient(torch.tensor([0.0, 0.0, 0.0], dtype=self.dtype))
        S = scipy.linalg.solve_continuous_are(
            A.detach().numpy(), B.detach().numpy(), Q, R
        )
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    def next_pose(self, x, u, dt):
        """
        Computes the next pose after dt.
        """
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        u_np = u.detach().numpy() if isinstance(u, torch.Tensor) else u
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u_np), [0, dt], x_np
        )
        return result.y[:, -1]


if __name__ == "__main__":
    plant = Unicycle(torch.float64)

    import matplotlib.pyplot as plt

    # plant = Point_Stabilization(torch.float64, v_max=6.0, v_gain=2.0)
    # x = np.array([10.0, 0.5])              # [d, theta_e]
    # Q = np.diag([2.0, 1.0])                # (2x2)
    # R = np.array([[0.5]])                  # (1x1) <-- must be square!
    # K, _ = plant.lqr_control(Q, R)         # K is (1x2)

    x = np.array([0.5, 0.5, 0.1])  # 0.5 m along x, 0.5 m along y, heading 0.1 rad
    x_des = np.zeros(3)

    Q = np.diag([2.0, 2.0, 2.0])  # penalize x,y more than theta
    R = np.diag([0.5, 0.5])  # penalize v and omega

    # After you build A,B,Q,R
    from numpy.linalg import matrix_rank

    A, B = plant.dynamics_gradient(torch.tensor(x, dtype=plant.dtype))
    n = A.shape[0]
    Ctr = np.hstack([B, A @ B, A @ A @ B])  # up to n-1 = 2 powers for 3-state
    print("rank(Controllability) =", matrix_rank(Ctr))  # should be 3

    # R must be PD & Q at least PSD
    print("eig(R) >", np.linalg.eigvalsh(R))  # should all be > 0
    print("eig(Q) >=", np.linalg.eigvalsh(Q))  # should be >= 0

    K, _ = plant.lqr_control(Q, R)  # K is (2x3)

    states = []
    controls = []

    for _ in range(10000):
        u = (K @ (x - x_des)).ravel()  # u = [v, omega]
        x_next = plant.next_pose(x, u, dt=0.02)
        states.append(x.copy())
        controls.append(u.copy())
        x = x_next
        if np.linalg.norm(x[:2]) < 0.1 and abs(x[2]) < 0.05:
            print("Reached close to the origin.")
            break

    states = np.array(states)
    controls = np.array(controls)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(states[:, 0], label="x")
    plt.plot(states[:, 1], label="y")
    plt.plot(states[:, 2], label="theta")
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.legend()
    plt.title("State Trajectory")

    plt.subplot(1, 2, 2)
    plt.plot(controls[:, 0], label="v")
    plt.plot(controls[:, 1], label="omega")
    plt.xlabel("Step")
    plt.ylabel("Control")
    plt.legend()
    plt.title("Control Effort")
    plt.tight_layout()
    plt.show()
