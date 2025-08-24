import torch
import numpy as np
import scipy
import scipy.linalg


class Point_Stabilization:
    """
    2D unicycle stabilization to the origin in polar errors.

    State:  x = [d, theta_e]
            d       >= 0  (distance to the origin)
            theta_e in [-pi, pi] (bearing error = angle-to-goal - heading)

    Control: u = [omega] (yaw rate)

    Key idea for Lyapunov/compatibility:
      - Use v_eff(d) = min(v_max, v_gain * d) so v_eff -> 0 as d->0.
        That removes the 1/d singularity at the origin and makes (0,0) an equilibrium.
      - Dynamics:
            d_dot      = - v_eff(d) * cos(theta_e)
            thetae_dot =   u[0]    - (v_eff(d)/max(d, eps)) * sin(theta_e)
      - Same API as Path_Following: dynamics(), dynamics_gradient(), lqr_control(), next_pose().
    """

    def __init__(self, dtype, v_max=6.0, v_gain=2.0, eps=1e-8):
        self.dtype = dtype
        self.v = float(v_max)  # keep name 'v' for compatibility (interpreted as cap)
        self.v_gain = float(v_gain)  # slope for v_eff near the origin
        self.eps = float(eps)

    # ---------- helpers ----------
    def _v_eff_np(self, d):
        return min(self.v, self.v_gain * max(float(d), 0.0))

    def _v_eff_torch(self, d: torch.Tensor):
        # piecewise-smooth cap; derivative dv/dd = v_gain in unsaturated region, 0 when saturated
        v_lin = self.v_gain * torch.clamp(d, min=0.0)
        return torch.minimum(torch.as_tensor(self.v, dtype=d.dtype), v_lin)

    # ---------- continuous-time dynamics ----------
    def dynamics(self, x, u):
        """
        x: [d, theta_e]   (np.ndarray or torch.Tensor)
        u: [omega]
        returns xdot with same backend (np or torch)
        """
        if isinstance(x, np.ndarray):
            d = x[0]
            th = x[1]
            w = u[0]
            v_eff = self._v_eff_np(d)
            d_safe = max(d, self.eps)
            d_dot = -v_eff * np.cos(th)
            th_dot = w - (v_eff / d_safe) * np.sin(th)
            return np.array([d_dot, th_dot])

        elif isinstance(x, torch.Tensor):
            d = x[0]
            th = x[1]
            w = (
                u[0]
                if isinstance(u, torch.Tensor)
                else torch.tensor(u[0], dtype=x.dtype)
            )
            v_eff = self._v_eff_torch(d)
            d_safe = torch.clamp(d, min=self.eps)
            d_dot = -v_eff * torch.cos(th)
            th_dot = w - (v_eff / d_safe) * torch.sin(th)
            return torch.cat((d_dot.view(1), th_dot.view(1)))

        else:
            raise TypeError("x must be np.ndarray or torch.Tensor")

    # ---------- linearization (A,B) ----------
    def dynamics_gradient(self, x):
        """
        A = ∂f/∂x, B = ∂f/∂u around x = [d, theta_e].
        Implemented in Torch (like Path_Following) for downstream code.
        We account for the saturation kink via piecewise dv/dd.
        """
        assert isinstance(x, torch.Tensor), "Pass x as a torch.Tensor for gradients."
        d = x[0]
        th = x[1]
        v_lin = self.v_gain * torch.clamp(d, min=0.0)
        v_eff = torch.minimum(torch.as_tensor(self.v, dtype=x.dtype), v_lin)
        dvdd = torch.where(
            v_lin < self.v,
            torch.as_tensor(self.v_gain, dtype=x.dtype),
            torch.as_tensor(0.0, dtype=x.dtype),
        )
        d_safe = torch.clamp(d, min=self.eps)

        # f1 = -v_eff * cos(th)
        a11 = -dvdd * torch.cos(th)
        a12 = v_eff * torch.sin(th)

        # f2 = u - (v_eff/d_safe) * sin(th) = u - g * sin(th), g = v_eff/d_safe
        g = v_eff / d_safe
        # dg/dd = (dv/dd * d_safe - v_eff * 1_{d>eps}) / d_safe^2
        one_d = torch.where(
            d > self.eps,
            torch.as_tensor(1.0, dtype=x.dtype),
            torch.as_tensor(0.0, dtype=x.dtype),
        )
        dgdd = (dvdd * d_safe - v_eff * one_d) / (d_safe**2)
        a21 = -dgdd * torch.sin(th)
        a22 = -g * torch.cos(th)

        A = torch.tensor([[a11, a12], [a21, a22]], dtype=self.dtype)
        B = torch.tensor([[0.0], [1.0]], dtype=self.dtype)
        return A, B

    # ---------- LQR around the origin ----------
    def lqr_control(self, Q, R):
        """
        Continuous LQR gain around the equilibrium (0, 0).
        Returns K (u = K (x - x_des)) and S.
        """
        A, B = self.dynamics_gradient(torch.tensor([0.0, 0.0], dtype=self.dtype))
        S = scipy.linalg.solve_continuous_are(
            A.detach().numpy(), B.detach().numpy(), Q, R
        )
        K = -np.linalg.solve(R, B.T @ S)
        return K, S

    # ---------- one-step integrator ----------
    def next_pose(self, x, u, dt):
        """
        Integrate forward by dt using solve_ivp (same as Path_Following).
        """
        x_np = x.detach().numpy() if isinstance(x, torch.Tensor) else x
        u_np = u.detach().numpy() if isinstance(u, torch.Tensor) else u
        result = scipy.integrate.solve_ivp(
            lambda t, x_val: self.dynamics(x_val, u_np), [0, dt], x_np
        )
        return result.y[:, -1]


if __name__ == "__main__":
    # Example usage
    plant = Point_Stabilization(torch.float64, v_max=6.0, v_gain=2.0)

    import matplotlib.pyplot as plt

    x = np.array([10.0, 0.5])  # 10 m away, 0.5 rad bearing error
    K, _ = plant.lqr_control(np.diag([2.0, 1.0]), np.array([[0.5]]))

    states = []
    controls = []

    for _ in range(10000):
        u = (K @ (x - np.zeros(2))).ravel()
        x_next = plant.next_pose(x, u, dt=0.02)
        states.append(x.copy())
        controls.append(u.copy())
        x = x_next
        if np.linalg.norm(x) < 0.1:
            print("Reached close to the origin.")
            break

    states = np.array(states)
    controls = np.array(controls)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(states[:, 0], label="Distance to origin (d)")
    plt.plot(states[:, 1], label="Bearing error (theta_e)")
    plt.xlabel("Step")
    plt.ylabel("State")
    plt.legend()
    plt.title("State Trajectory")

    plt.subplot(1, 2, 2)
    plt.plot(controls[:, 0], label="Control effort (omega)")
    plt.xlabel("Step")
    plt.ylabel("Control")
    plt.legend()
    plt.title("Control Effort")

    plt.tight_layout()
    plt.show()
