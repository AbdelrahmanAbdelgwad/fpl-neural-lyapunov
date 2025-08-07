#!/usr/bin/env python3
"""
Test and Visualization Script for Path Following Unicycle with Lyapunov Control
This script loads trained models and creates comprehensive visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
import os
import argparse

# Import the necessary modules from your project
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.examples.path_following_unicycle.path_following as path_following
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.monotonic_lyapunov_init.custom_lyapunov as lyapunov
import neural_network_lyapunov.r_options as r_options


class PathFollowingVisualizer:
    """Visualizer for path following unicycle system with Lyapunov control"""

    def __init__(self, model_dir, bound_level=40):
        """
        Initialize the visualizer

        Args:
            model_dir: Directory containing the trained models
            bound_level: The bound level used during training
        """
        self.model_dir = model_dir + f"/monotonic/monotonic_bound{bound_level}/"
        self.bound_level = bound_level
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.dtype = torch.float64

        # System parameters
        self.plant = path_following.Path_Following(torch.float64)

        # State bounds - MOVE TO DEVICE
        self.x_lo = (
            torch.tensor([-0.8, -0.8], dtype=self.dtype, device=self.device)
            * bound_level
            / 40.0
        )
        self.x_up = (
            torch.tensor([0.8, 0.8], dtype=self.dtype, device=self.device)
            * bound_level
            / 40.0
        )

        # Control bounds - MOVE TO DEVICE
        self.u_lo = torch.tensor([-10], dtype=self.dtype, device=self.device)
        self.u_up = torch.tensor([10], dtype=self.dtype, device=self.device)

        dynamics_model_path = (
            model_dir + "/preprocess/path_following_unicycle_forward_model.pt"
        )

        dynamics_relu = torch.load(dynamics_model_path, map_location=self.device)
        dt = 0.01
        # MOVE TO DEVICE
        self.x_equilibrium = torch.tensor(
            [0.0, 0.0], dtype=self.dtype, device=self.device
        )
        self.u_equilibrium = torch.tensor(
            [self.plant.v], dtype=self.dtype, device=self.device
        )

        self.forward_system = relu_system.ReLUSystemGivenEquilibrium(
            torch.float64,
            self.x_lo,
            self.x_up,
            self.u_lo,
            self.u_up,
            dynamics_relu,
            self.x_equilibrium,
            self.u_equilibrium,
            dt,
        )

        # Load models
        self._load_models()

    def _load_models(self):
        """Load the trained controller and Lyapunov networks"""
        # Load controller
        controller_path = os.path.join(
            self.model_dir, f"monotonic_bound{self.bound_level}_controller.pt"
        )
        self.controller_relu = torch.load(controller_path, map_location=self.device)

        # Load Lyapunov network
        lyapunov_path = os.path.join(
            self.model_dir, f"monotonic_bound{self.bound_level}_lyapunov.pt"
        )
        self.lyapunov_relu = torch.load(lyapunov_path, map_location=self.device)

        # Load R matrix
        R_path = os.path.join(self.model_dir, f"monotonic_bound{self.bound_level}_R.pt")
        self.R = torch.load(R_path, map_location=self.device)

        # Create feedback system
        self.closed_loop_system = feedback_system.FeedbackSystem(
            self.forward_system,
            self.controller_relu,
            self.x_equilibrium,
            self.u_equilibrium,
            self.u_lo.detach().cpu().numpy(),
            self.u_up.detach().cpu().numpy(),
        )

        # Create Lyapunov system
        self.lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            self.closed_loop_system, self.lyapunov_relu
        )

        print(f"✓ Loaded models from {self.model_dir}")
        print(f"  - Controller: {controller_path}")
        print(f"  - Lyapunov: {lyapunov_path}")
        print(f"  - R matrix shape: {self.R.shape}")

    def compute_lyapunov_value(self, x, V_lambda=0.1):
        """Compute Lyapunov function value at state x"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype)
        return self.lyapunov_hybrid_system.lyapunov_value(
            x, self.x_equilibrium, V_lambda, R=self.R
        )

    def compute_control(self, x):
        """Compute control input for state x"""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype)
        return self.closed_loop_system.compute_u(x)

    def simulate_trajectory(self, x0, t_final=10.0, dt=0.01):
        """
        Simulate system trajectory from initial condition x0

        Returns:
            t: Time vector
            x_traj: State trajectory
            u_traj: Control trajectory
            V_traj: Lyapunov values along trajectory
        """

        def dynamics(t, x):
            x_torch = torch.tensor(x, dtype=self.dtype)
            x_next = self.closed_loop_system.step_forward(x_torch)
            # Convert discrete-time to continuous-time derivative
            return (x_next.detach().numpy() - x) / dt

        # Simulate
        t_eval = np.arange(0, t_final, dt)
        sol = scipy.integrate.solve_ivp(
            dynamics, [0, t_final], x0, t_eval=t_eval, method="RK45"
        )

        # Extract trajectories
        t = sol.t
        x_traj = sol.y.T

        # Compute control and Lyapunov values
        u_traj = np.zeros((len(t), 1))
        V_traj = np.zeros(len(t))

        for i in range(len(t)):
            u_traj[i] = self.compute_control(x_traj[i]).detach().numpy()
            V_traj[i] = self.compute_lyapunov_value(x_traj[i]).item()

        return t, x_traj, u_traj, V_traj

    def plot_lyapunov_function(self, n_points=100, V_lambda=0.1):
        """Plot the Lyapunov function as a 3D surface and contour plot"""
        # Create grid
        de_range = np.linspace(self.x_lo[0].item(), self.x_up[0].item(), n_points)
        theta_e_range = np.linspace(self.x_lo[1].item(), self.x_up[1].item(), n_points)
        DE, THETA_E = np.meshgrid(de_range, theta_e_range)

        # Compute Lyapunov values
        V = np.zeros_like(DE)
        for i in range(n_points):
            for j in range(n_points):
                x = torch.tensor([DE[i, j], THETA_E[i, j]], dtype=self.dtype)
                V[i, j] = self.compute_lyapunov_value(x, V_lambda).item()

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 6))

        # 3D surface plot
        ax1 = fig.add_subplot(121, projection="3d")
        surf = ax1.plot_surface(DE, THETA_E, V, cmap="viridis", alpha=0.8)
        ax1.set_xlabel("$d_e$ (distance error)")
        ax1.set_ylabel("$\\theta_e$ (angle error)")
        ax1.set_zlabel("V(x)")
        ax1.set_title("Lyapunov Function Surface")
        fig.colorbar(surf, ax=ax1, shrink=0.5)

        # Contour plot
        ax2 = fig.add_subplot(122)
        levels = np.percentile(V.flatten(), np.linspace(5, 95, 20))
        contour = ax2.contour(DE, THETA_E, V, levels=levels, colors="blue", alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        contourf = ax2.contourf(
            DE, THETA_E, V, levels=levels, cmap="viridis", alpha=0.3
        )

        # Mark equilibrium
        ax2.plot(0, 0, "r*", markersize=15, label="Equilibrium")

        # Add state space bounds
        rect = patches.Rectangle(
            (self.x_lo[0].item(), self.x_lo[1].item()),
            self.x_up[0].item() - self.x_lo[0].item(),
            self.x_up[1].item() - self.x_lo[1].item(),
            linewidth=2,
            edgecolor="k",
            facecolor="none",
            linestyle="--",
        )
        ax2.add_patch(rect)

        ax2.set_xlabel("$d_e$ (distance error)")
        ax2.set_ylabel("$\\theta_e$ (angle error)")
        ax2.set_title("Lyapunov Function Level Sets")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")

        plt.tight_layout()
        return fig

    def plot_region_of_attraction(self, V_max=1.0, n_points=100):
        """Plot the region of attraction"""
        # Create grid
        de_range = np.linspace(self.x_lo[0].item(), self.x_up[0].item(), n_points)
        theta_e_range = np.linspace(self.x_lo[1].item(), self.x_up[1].item(), n_points)
        DE, THETA_E = np.meshgrid(de_range, theta_e_range)

        # Compute Lyapunov values
        V = np.zeros_like(DE)
        for i in range(n_points):
            for j in range(n_points):
                x = torch.tensor([DE[i, j], THETA_E[i, j]], dtype=self.dtype)
                V[i, j] = self.compute_lyapunov_value(x).item()

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))

        # Find ROA boundary (largest level set within state bounds)
        V_flat = V.flatten()
        V_sorted = np.sort(V_flat)

        # Plot multiple level sets
        percentiles = [10, 25, 50, 75, 90, 95]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(percentiles)))

        for i, p in enumerate(percentiles):
            level = np.percentile(V_sorted, p)
            contour = ax.contour(
                DE, THETA_E, V, levels=[level], colors=[colors[i]], linewidths=2
            )
            # ax.clabel(contour, inline=True, fontsize=8, fmt=f"{p}%")
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f%%")

        # Highlight the ROA boundary
        roa_level = np.percentile(V_sorted, 95)  # Adjust this based on your needs
        ax.contour(
            DE, THETA_E, V, levels=[roa_level], colors="red", linewidths=3, label="ROA"
        )

        # State space bounds
        rect = patches.Rectangle(
            (self.x_lo[0].item(), self.x_lo[1].item()),
            self.x_up[0].item() - self.x_lo[0].item(),
            self.x_up[1].item() - self.x_lo[1].item(),
            linewidth=2,
            edgecolor="k",
            facecolor="none",
            linestyle="--",
            label="State bounds",
        )
        ax.add_patch(rect)

        # Equilibrium
        ax.plot(0, 0, "r*", markersize=15, label="Equilibrium")

        # Unit circle (path to follow)
        circle = Circle(
            (0, 0),
            1.0,
            fill=False,
            edgecolor="green",
            linewidth=2,
            linestyle=":",
            label="Unit circle path",
        )
        ax.add_patch(circle)

        ax.set_xlabel("$d_e$ (distance error)")
        ax.set_ylabel("$\\theta_e$ (angle error)")
        ax.set_title("Region of Attraction")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        return fig

    def plot_trajectories(self, n_trajectories=10, t_final=5.0):
        """Plot multiple trajectories with random initial conditions"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Generate random initial conditions
        initial_conditions = []
        for _ in range(n_trajectories):
            # Sample within state bounds
            x0 = np.random.uniform(
                [self.x_lo[0].item(), self.x_lo[1].item()],
                [self.x_up[0].item(), self.x_up[1].item()],
            )
            initial_conditions.append(x0)

        # Simulate trajectories
        trajectories = []
        for x0 in initial_conditions:
            t, x_traj, u_traj, V_traj = self.simulate_trajectory(x0, t_final)
            trajectories.append((t, x_traj, u_traj, V_traj))

        # Plot state space trajectories
        ax = axes[0, 0]
        for i, (t, x_traj, _, _) in enumerate(trajectories):
            ax.plot(
                x_traj[:, 0],
                x_traj[:, 1],
                alpha=0.7,
                label=f"Traj {i+1}" if i < 3 else "",
            )
            ax.plot(x_traj[0, 0], x_traj[0, 1], "o", markersize=8)
            ax.plot(x_traj[-1, 0], x_traj[-1, 1], "s", markersize=8)

        ax.plot(0, 0, "r*", markersize=15, label="Equilibrium")
        ax.set_xlabel("$d_e$")
        ax.set_ylabel("$\\theta_e$")
        ax.set_title("State Space Trajectories")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        # Plot time evolution of states
        ax = axes[0, 1]
        for t, x_traj, _, _ in trajectories:
            ax.plot(t, x_traj[:, 0], alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$d_e$")
        ax.set_title("Distance Error vs Time")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for t, x_traj, _, _ in trajectories:
            ax.plot(t, x_traj[:, 1], alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$\\theta_e$")
        ax.set_title("Angle Error vs Time")
        ax.grid(True, alpha=0.3)

        # Plot Lyapunov function evolution
        ax = axes[1, 1]
        for t, _, _, V_traj in trajectories:
            ax.semilogy(t, V_traj, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("V(x)")
        ax.set_title("Lyapunov Function vs Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_control_effort(self, n_trajectories=5, t_final=5.0):
        """Plot control effort for multiple trajectories"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate trajectories
        for i in range(n_trajectories):
            x0 = np.random.uniform(
                [self.x_lo[0].item(), self.x_lo[1].item()],
                [self.x_up[0].item(), self.x_up[1].item()],
            )
            t, x_traj, u_traj, _ = self.simulate_trajectory(x0, t_final)

            ax.plot(t, u_traj, alpha=0.7, label=f"x0=[{x0[0]:.2f}, {x0[1]:.2f}]")

        # Plot control limits
        ax.axhline(
            y=self.u_lo.item(), color="r", linestyle="--", label="Control limits"
        )
        ax.axhline(y=self.u_up.item(), color="r", linestyle="--")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control Input")
        ax.set_title("Control Effort vs Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def create_animation(self, x0, t_final=10.0, dt=0.01):
        """Create an animation of the system evolution (requires additional setup)"""
        # This is a placeholder for animation functionality
        # You would need matplotlib.animation for full implementation
        print("Animation functionality requires matplotlib.animation setup")
        print("Trajectory simulation from x0 =", x0)

        t, x_traj, u_traj, V_traj = self.simulate_trajectory(x0, t_final, dt)
        print(f"  - Simulation completed: {len(t)} time steps")
        print(f"  - Final state: {x_traj[-1]}")
        print(f"  - Final Lyapunov value: {V_traj[-1]:.6f}")

    def save_all_plots(self, output_dir="./visualization_results"):
        """Save all visualization plots"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating visualizations...")

        # Lyapunov function
        fig = self.plot_lyapunov_function()
        fig.savefig(
            os.path.join(output_dir, "lyapunov_function.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print("  ✓ Lyapunov function plot saved")

        # Region of attraction
        fig = self.plot_region_of_attraction()
        fig.savefig(
            os.path.join(output_dir, "region_of_attraction.png"),
            dpi=150,
            bbox_inches="tight",
        )
        print("  ✓ Region of attraction plot saved")

        # Trajectories
        fig = self.plot_trajectories()
        fig.savefig(
            os.path.join(output_dir, "trajectories.png"), dpi=150, bbox_inches="tight"
        )
        print("  ✓ Trajectories plot saved")

        # Control effort
        fig = self.plot_control_effort()
        fig.savefig(
            os.path.join(output_dir, "control_effort.png"), dpi=150, bbox_inches="tight"
        )
        print("  ✓ Control effort plot saved")

        plt.close("all")
        print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Test and visualize Lyapunov controller"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./neural_network_lyapunov/examples/path_following_unicycle/data",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--bound_level", type=int, default=40, help="Bound level used during training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./visualization_results",
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Show plots interactively"
    )

    args = parser.parse_args()

    # Create visualizer
    visualizer = PathFollowingVisualizer(args.model_dir, args.bound_level)

    # Generate and save all plots
    visualizer.save_all_plots(args.output_dir)

    # Test specific functionality
    print("\n" + "=" * 50)
    print("Testing controller and Lyapunov function:")
    print("=" * 50)

    # Test at equilibrium
    x_eq = np.array([0.0, 0.0])
    u_eq = visualizer.compute_control(x_eq).detach().numpy()
    V_eq = visualizer.compute_lyapunov_value(x_eq).item()
    print(f"\nAt equilibrium x = {x_eq}:")
    print(f"  - Control u = {u_eq[0]:.6f} (expected: {visualizer.plant.v})")
    print(f"  - Lyapunov V = {V_eq:.6f} (expected: 0)")

    # Test at random point
    x_test = np.array([0.1, -0.05])
    u_test = visualizer.compute_control(x_test).detach().numpy()
    V_test = visualizer.compute_lyapunov_value(x_test).item()
    print(f"\nAt test point x = {x_test}:")
    print(f"  - Control u = {u_test[0]:.6f}")
    print(f"  - Lyapunov V = {V_test:.6f}")

    # Simulate a trajectory
    print("\nSimulating trajectory from x0 = [0.5, -0.3]:")
    t, x_traj, u_traj, V_traj = visualizer.simulate_trajectory([0.5, -0.3], t_final=5.0)
    print(f"  - Initial Lyapunov value: {V_traj[0]:.6f}")
    print(f"  - Final Lyapunov value: {V_traj[-1]:.6f}")
    print(f"  - Lyapunov decrease: {(V_traj[-1] - V_traj[0])/V_traj[0]*100:.2f}%")

    if args.show_plots:
        plt.show()

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
