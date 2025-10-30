#!/usr/bin/env python3
"""
Unified visualization framework for comparing controllers with and without FPL.
Optimized for multi-core processing and GPU acceleration.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.patches import Rectangle
import scipy.integrate
import argparse
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device for GPU acceleration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')  # Force CPU for debugging
print(f"Using device: {DEVICE}")

# Import your modules (adjust paths as needed)
import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.examples.path_following_unicycle.path_following as path_following
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils_0615 as monotonic_utils


class BaseVisualizationFramework:
    """Base class for environment-agnostic visualizations."""
    
    def __init__(self, bound_level=3, V_lambda=0.6, n_workers=32):
        self.bound_level = bound_level
        self.V_lambda = V_lambda
        self.n_workers = min(n_workers, os.cpu_count() or 1)
        self.device = DEVICE
        self.dtype = torch.float64
        
        # To be defined by subclasses
        self.plant = None
        self.x_equilibrium = None
        self.u_equilibrium = None
        self.dynamics_model = None
        self.forward_system = None
        
    def load_models(self, seed=0, use_fpl=False, env=None):
        """Load trained models for a specific seed."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        suffix = "_fpl" if use_fpl else ""
        seed_suffix = f"_seed{seed}" if seed > 0 else ""

        if env == "path_following":
            env_specific_path = "/neural_network_lyapunov/examples/path_following_unicycle"
        elif env == "pendulum":
            env_specific_path = "/neural_network_lyapunov/examples/pendulum"

        # Load controller
        if env == "path_following":
            base_path = dir_path + env_specific_path + f"/data/monotonic/monotonic_bound{self.bound_level}{suffix}{seed_suffix}"
            controller_path = base_path + f"/monotonic_bound{self.bound_level}{suffix}_controller.pt"
        elif env == "pendulum":
            base_path = dir_path + env_specific_path + f"/data/monotonic_bound{self.bound_level}{suffix}{seed_suffix}"
            controller_path = base_path + f"/monotonic_bound{self.bound_level}{suffix}_controller.pt"
        else:
            raise ValueError("Unsupported environment specified.")
        controller_relu = torch.load(controller_path, map_location=self.device)
        controller_relu.eval()
        
        # Load Lyapunov
        lyapunov_path = base_path + f"/monotonic_bound{self.bound_level}{suffix}_lyapunov.pt"
        lyapunov_relu = torch.load(lyapunov_path, map_location=self.device)
        lyapunov_relu.eval()
        
        # Load R matrix
        R_path = base_path + f"/monotonic_bound{self.bound_level}{suffix}_R.pt"
        R = torch.load(R_path, map_location=self.device)
        
        return controller_relu.to(self.device), lyapunov_relu.to(self.device), R.to(self.device)
    
    def compute_lyapunov_value_batch(self, x_batch, lyapunov_relu, R):
        """GPU-accelerated batch Lyapunov computation."""
        with torch.no_grad():
            if len(x_batch.shape) == 1:
                x_batch = x_batch.unsqueeze(0)
            
            x_batch = x_batch.to(self.device)
            x_eq = self.x_equilibrium.to(self.device)
            R = R.to(self.device)
            
            relu_at_equilibrium = lyapunov_relu(x_eq)
            V = lyapunov_relu(x_batch) - relu_at_equilibrium + \
                self.V_lambda * torch.norm(R @ (x_batch - x_eq).T, p=1, dim=0).unsqueeze(1)
            
            return V.squeeze().cpu().numpy()

    def simulate_trajectory_single(self, x0, controller, T=10.0, dt=0.01, track_time=False, device=None):
        """Simulate a single trajectory."""
        controller_cpu = controller.cpu()
        x_eq_cpu = self.x_equilibrium.cpu()
        u_eq_cpu = self.u_equilibrium.cpu()
        u_lo_cpu = self.u_lo.cpu()
        u_up_cpu = self.u_up.cpu()

        convergence_time = None
        converged = False
            
        def dynamics(t, x):
            with torch.no_grad():
                # SciPy runs on CPU; keep tensors on CPU to avoid device mismatch.
                x_torch = torch.tensor(x, dtype=self.dtype)  # CPU by default
                u_pre_sat = controller_cpu(x_torch) - controller_cpu(x_eq_cpu) + u_eq_cpu
                u = torch.max(torch.min(u_pre_sat, u_up_cpu), u_lo_cpu).detach().cpu().numpy()
                return self.plant.dynamics(x, u)
        
        if track_time:
            # Track convergence time
            t_eval = np.arange(0, T, dt)
            result = scipy.integrate.solve_ivp(
                dynamics, [0, T], x0, t_eval=t_eval, method='RK45', rtol=1e-6
            )
            
            # Check convergence at each timestep
            for i, t in enumerate(result.t):
                error = np.linalg.norm(result.y[:, i] - self.x_equilibrium.numpy())
                if error < 1e-3:
                    convergence_time = t
                    converged = True
                    break
            
            return converged, convergence_time, result.t, result.y
        else:
            # Simple convergence check
            def convergence_event(t, x):
                return np.linalg.norm(x - self.x_equilibrium.numpy()) - 1e-3
            
            convergence_event.terminal = True
            convergence_event.direction = -1
            
            result = scipy.integrate.solve_ivp(
                dynamics, [0, T], x0, method='RK45', rtol=1e-6,
                events=convergence_event, max_step=0.5
            )
            
            converged = len(result.t_events[0]) > 0 if result.t_events else False
            return converged, result.t[-1] if converged else T, result.t, result.y
    
    def parallel_simulate_batch(self, x0_batch, controller, T=10.0, track_time=True):
        """Parallel simulation of multiple trajectories."""
        simulate_fn = partial(self.simulate_trajectory_single, 
                             controller=controller, T=T, track_time=track_time, device=None)
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(
                executor.map(simulate_fn, x0_batch),
                total=len(x0_batch),
                desc="Simulating trajectories"
            ))
        
        return results
    
    def compute_3d_lyapunov(self, lyapunov_relu, R, grid_resolution=50):
        """Compute 3D Lyapunov function over state space."""
        raise NotImplementedError("Subclasses must implement this")
    
    def plot_training_curves(self, seeds=None, save_path=None):
        """Plot training loss vs wall clock time."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, use_fpl in enumerate([False, True]):
            ax = axes[idx]
            suffix = "FPL" if use_fpl else "Standard"
            
            if seeds is None:
                seeds = [0]
            
            all_times = []
            all_losses = []
            
            for seed in seeds:
                log_file = self.get_log_file_path(seed, use_fpl)
                if not os.path.exists(log_file):
                    print(f"Warning: Log file not found: {log_file}")
                    continue
                
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                
                times = [entry['wall_time'] for entry in logs]
                losses = [entry['loss'] for entry in logs]
                
                all_times.append(times)
                all_losses.append(losses)
                
                ax.plot(times, losses, alpha=0.3, linewidth=0.5)
            
            # Plot mean and std if multiple seeds
            if len(all_times) > 1:
                # Interpolate to common time points
                max_time = min(max(t) for t in all_times)
                common_times = np.linspace(0, max_time, 1000)
                interpolated = np.array([
                    np.interp(common_times, times, losses)
                    for times, losses in zip(all_times, all_losses)
                ])
                
                mean_loss = np.mean(interpolated, axis=0)
                std_loss = np.std(interpolated, axis=0)
                
                ax.plot(common_times, mean_loss, 'r-', linewidth=2, label='Mean')
                ax.fill_between(common_times, mean_loss - std_loss, mean_loss + std_loss,
                               alpha=0.2, color='red')
            
            ax.set_xlabel('Wall Clock Time (s)')
            ax.set_ylabel('Loss')
            ax.set_title(f'{suffix} Controller Training')
            ax.grid(True, alpha=0.3)
            if len(all_times) > 1:
                ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.svg', bbox_inches='tight')
        return fig
    
    def plot_distance_from_equilibrium(self, test_x0s, T=25.0, save_path=None, env=None):
        """Plot distance from equilibrium over time for multiple ICs."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        controllers = {}
        for use_fpl in [False, True]:
            controller, _, _ = self.load_models(use_fpl=use_fpl, env=env)
            controllers[use_fpl] = controller
        
        for idx, (use_fpl, ax) in enumerate(zip([False, True], axes)):
            controller = controllers[use_fpl]
            suffix = "FPL" if use_fpl else "Standard"
            
            for ic_idx, x0 in enumerate(test_x0s):
                _, _, t, y = self.simulate_trajectory_single(
                    x0, controller, T=T, track_time=False,
                )
                
                # Compute distance from equilibrium
                distances = np.linalg.norm(
                    y - self.x_equilibrium.numpy().reshape(-1, 1), 
                    axis=0
                )
                
                ax.semilogy(t, distances, label=f'IC{ic_idx+1}', linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('||x - x_eq||')
            ax.set_title(f'{suffix} Controller - Distance from Equilibrium')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.svg', bbox_inches='tight')
        return fig
    
    def plot_roa_with_convergence_time(self, n_samples=1000, T=10.0, 
                                      time_bins=None, save_path=None, env=None):
        """Plot ROA with color-coded convergence times."""
        if time_bins is None:
            time_bins = [0.5, 1.0, 2.0, 5.0, 10.0]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sample initial conditions
        np.random.seed(42)
        x0_samples = self.sample_initial_conditions(n_samples)
        
        for idx, use_fpl in enumerate([False, True]):
            ax = axes[idx]
            controller, _, _ = self.load_models(use_fpl=use_fpl, env=env)
            suffix = "FPL" if use_fpl else "Standard"
            
            # Simulate all trajectories in parallel
            results = self.parallel_simulate_batch(x0_samples, controller, T=T, track_time=True)
            
            # Extract convergence times and states
            conv_times = np.array([r[1] if r[0] else T for r in results])
            converged = np.array([r[0] for r in results])
            
            # Create color map based on time bins
            colors = np.zeros(n_samples)
            for i, t in enumerate(conv_times):
                if not converged[i]:
                    colors[i] = -1  # Failed (red)
                else:
                    # Find appropriate bin
                    bin_idx = np.searchsorted(time_bins, t)
                    colors[i] = bin_idx
            
            # Create colormap
            n_bins = len(time_bins) + 1
            cmap = cm.get_cmap('RdYlGn_r', n_bins + 1)
            
            # Plot with color coding
            self.plot_roa_scatter(ax, x0_samples, colors, cmap, time_bins, converged)
            
            ax.set_title(f'{suffix} Controller - ROA with Convergence Time')
            
            # Add statistics
            conv_rate = np.mean(converged) * 100
            mean_time = np.mean(conv_times[converged]) if np.any(converged) else 0
            ax.text(0.02, 0.98, f'Conv Rate: {conv_rate:.1f}%\nMean Time: {mean_time:.2f}s',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.svg', bbox_inches='tight')
        return fig

    def plot_3d_lyapunov_comparison(self, grid_resolution=50, save_path=None, env=None):
        """Plot 3D Lyapunov functions for both controllers."""
        fig = plt.figure(figsize=(14, 6))
        
        for idx, use_fpl in enumerate([False, True]):
            ax = fig.add_subplot(1, 2, idx+1, projection='3d')
            _, lyapunov_relu, R = self.load_models(use_fpl=use_fpl, env=env)
            suffix = "FPL" if use_fpl else "Standard"
            
            X, Y, V = self.compute_3d_lyapunov(lyapunov_relu, R, grid_resolution)
            
            # Surface plot
            surf = ax.plot_surface(X, Y, V, cmap='viridis', alpha=0.8,
                                  edgecolor='none', antialiased=True)
            
            # Add contour lines at the bottom
            ax.contour(X, Y, V, levels=10, zdir='z', offset=np.min(V),
                      colors='black', alpha=0.3, linewidths=0.5)
            
            ax.set_xlabel(self.state_labels[0])
            ax.set_ylabel(self.state_labels[1])
            ax.set_zlabel('V(x)')
            ax.set_title(f'{suffix} Controller - Lyapunov Function')
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.svg', bbox_inches='tight')
        return fig
    
    def create_comprehensive_report(self, test_x0s=None, n_samples=1000, 
                                   time_bins=None, grid_resolution=50,
                                   seeds=None, save_dir=None, env=None):
        """Generate all visualizations and save them."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Training curves
        print("Generating training curves...")
        self.plot_training_curves(
            seeds=seeds,
            save_path=os.path.join(save_dir, 'training_curves') if save_dir else None
        )
        
        # 2. Distance from equilibrium
        print("Generating distance from equilibrium plots...")
        if test_x0s is None:
            test_x0s = self.get_default_test_points()
        self.plot_distance_from_equilibrium(
            test_x0s=test_x0s,
            save_path=os.path.join(save_dir, 'distance_from_eq') if save_dir else None,
            env=env
        )
        
        # 3. ROA with convergence times
        print("Generating ROA with convergence times...")
        self.plot_roa_with_convergence_time(
            n_samples=n_samples,
            time_bins=time_bins,
            save_path=os.path.join(save_dir, 'roa_convergence') if save_dir else None,
            env=env
        )
        
        # 4. 3D Lyapunov functions
        print("Generating 3D Lyapunov functions...")
        self.plot_3d_lyapunov_comparison(
            grid_resolution=grid_resolution,
            save_path=os.path.join(save_dir, 'lyapunov_3d') if save_dir else None,
            env=env
        )
        
        if not save_dir:
            plt.show()

    def simulate_trajectory_ml(self, x0, controller, T=10.0, dt=0.01, track_time=False, device=None):
        """Simulate a single trajectory using ML dynamics model.
        
        This uses the learned neural network dynamics model instead of analytical ODE.
        """
        if not hasattr(self, 'forward_system') or self.forward_system is None:
            raise NotImplementedError("ML forward system not implemented for this environment")
        
        # Keep EVERYTHING on the same device (GPU if available).
        device = self.device if device is None else device
        controller_dev = controller.to(device)
        x_eq_dev = self.x_equilibrium.to(device)
        u_eq_dev = self.u_equilibrium.to(device)
        u_lo_dev = self.u_lo.to(device)
        u_up_dev = self.u_up.to(device)

        convergence_time = None
        converged = False
        
        # ML-based simulation using the forward dynamics model
        num_steps = int(T / dt)
        t = np.arange(0, T + dt, dt)[:num_steps]
        x = np.zeros((len(self.x_equilibrium), num_steps))
        x[:, 0] = x0

        x_curr = torch.tensor(x0, dtype=self.dtype, device=device)

        with torch.no_grad():
            for i in range(num_steps - 1):
                # Compute control using neural network controller (same device as state)
                u_pre_sat = controller_dev(x_curr) - controller_dev(x_eq_dev) + u_eq_dev
                u_sat = torch.max(torch.min(u_pre_sat, u_up_dev), u_lo_dev)
                u_sat = u_sat.to(device)


                # Step forward using ML dynamics model
                x_next = self.forward_system.step_forward(x_curr, u_sat)
                x[:, i+1] = x_next.detach().cpu().numpy()
                x_curr = x_next
                
                if track_time:
                    error = torch.norm(x_curr - x_eq_dev).item()
                    if error < 1e-3 and not converged:
                        convergence_time = t[i+1]
                        converged = True
        
        return converged, convergence_time, t, x

    def plot_distance_from_equilibrium(self, test_x0s, T=25.0, save_path=None, env=None):
        """Plot distance from equilibrium for both analytical and ML models.
        
        Shows comparison between Standard and FPL controllers:
        - Same IC has same color
        - Solid line for Standard controller
        - Dashed line for FPL controller
        - Left subplot: Analytical ODE model
        - Right subplot: ML dynamics model
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Load both controllers
        controller_std, _, _ = self.load_models(use_fpl=False, env=env)
        controller_fpl, _, _ = self.load_models(use_fpl=True, env=env)
        
        # Define colors for different ICs
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(test_x0s)))
        
        # Plot for both model types
        for model_idx, (ax, model_type) in enumerate(zip(axes, ['Analytical ODE', 'ML Neural Network'])):
            
            # Create legend entries tracking
            legend_elements = []
            
            for ic_idx, x0 in enumerate(test_x0s):
                color = colors[ic_idx]
                
                # === STANDARD CONTROLLER (SOLID LINE) ===
                if model_type == 'Analytical ODE':
                    # Uses self.plant.dynamics() - the analytical differential equations
                    _, _, t_std, y_std = self.simulate_trajectory_single(
                        x0, controller_std, T=T, track_time=False, device=self.device
                    )
                else:  # ML Neural Network
                    # Uses self.forward_system.step_forward() - the learned dynamics
                    _, _, t_std, y_std = self.simulate_trajectory_ml(
                        x0, controller_std, T=T, track_time=False, device=self.device
                    )
                
                distances_std = np.linalg.norm(
                    y_std - self.x_equilibrium.numpy().reshape(-1, 1), 
                    axis=0
                )
                
                # === FPL CONTROLLER (DASHED LINE) ===
                if model_type == 'Analytical ODE':
                    # Uses self.plant.dynamics() - the analytical differential equations
                    _, _, t_fpl, y_fpl = self.simulate_trajectory_single(
                        x0, controller_fpl, T=T, track_time=False, device=self.device
                    )
                else:  # ML Neural Network
                    # Uses self.forward_system.step_forward() - the learned dynamics
                    _, _, t_fpl, y_fpl = self.simulate_trajectory_ml(
                        x0, controller_fpl, T=T, track_time=False, device=self.device
                    )
                
                distances_fpl = np.linalg.norm(
                    y_fpl - self.x_equilibrium.numpy().reshape(-1, 1), 
                    axis=0
                )
                
                # Plot both controllers for this IC
                line_std = ax.semilogy(t_std, distances_std, '-', color=color, 
                                    linewidth=1.5, alpha=0.9)[0]
                line_fpl = ax.semilogy(t_fpl, distances_fpl, '--', color=color,
                                    linewidth=1.5, alpha=0.9)[0]
                
                # Add to legend (only for first subplot to avoid duplication)
                if model_idx == 0:
                    legend_elements.append((line_std, f'IC{ic_idx+1}'))
            
            # Customize subplot
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('||x - x_eq|| (log scale)', fontsize=10)
            ax.set_title(f'{model_type} Model', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(bottom=1e-4)  # Set lower limit for better visualization
            
            # Add custom legend for IC labels
            if model_idx == 0:
                ic_legend = ax.legend([elem[0] for elem in legend_elements], 
                                    [elem[1] for elem in legend_elements],
                                    loc='upper right', fontsize=8, title='Initial Conditions')
            
            # Add annotation box explaining line styles
            style_text = 'Line Styles:\n━━ Standard\n┅┅ FPL'
            ax.text(0.02, 0.15, style_text, 
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7),
                verticalalignment='top')
        
        # Add overall title
        fig.suptitle('Distance from Equilibrium: Analytical vs ML Dynamics', 
                    fontsize=13, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.svg', bbox_inches='tight')
        return fig
    
    # Abstract methods to be implemented by subclasses
    def sample_initial_conditions(self, n_samples):
        raise NotImplementedError
    
    def plot_roa_scatter(self, ax, x0_samples, colors, cmap, time_bins, converged):
        raise NotImplementedError
    
    def get_default_test_points(self):
        raise NotImplementedError
    
    def get_log_file_path(self, seed, use_fpl):
        raise NotImplementedError


class PendulumVisualization(BaseVisualizationFramework):
    """Pendulum-specific visualization implementation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.plant = pendulum.Pendulum(self.dtype)
        self.x_equilibrium = torch.tensor([np.pi, 0], dtype=self.dtype)
        self.u_equilibrium = torch.tensor([0], dtype=self.dtype)
        self.u_lo = torch.tensor([-20], dtype=self.dtype)
        self.u_up = torch.tensor([20], dtype=self.dtype)
        self.state_labels = ['θ (rad)', 'θ̇ (rad/s)']
        
        # Load dynamics model
        self.load_dynamics_model()
        self.create_forward_system()
    
    def load_dynamics_model(self):
        """Load the ML dynamics model for pendulum."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dynamics_model_path = dir_path + "/neural_network_lyapunov/examples/pendulum/data/pendulum_second_order_forward_relu2.pt"
        dynamics_model_data = torch.load(dynamics_model_path, map_location=self.device)
        
        self.dynamics_model = utils.setup_relu(
            dynamics_model_data["linear_layer_width"],
            params=None,
            negative_slope=dynamics_model_data["negative_slope"],
            bias=True,
            dtype=self.dtype,
        )
        self.dynamics_model.load_state_dict(dynamics_model_data["state_dict"])
        self.dynamics_model.to(self.device)
        self.dynamics_model.eval()
    
    def sample_initial_conditions(self, n_samples):
        """Sample initial conditions for pendulum."""
        theta_samples = np.random.uniform(
            np.pi - 0.1 * self.bound_level * np.pi,
            np.pi + 0.1 * self.bound_level * np.pi,
            n_samples
        )
        theta_dot_samples = np.random.uniform(
            -0.5 * self.bound_level,
            0.5 * self.bound_level,
            n_samples
        )
        return np.column_stack((theta_samples, theta_dot_samples))
    
    def plot_roa_scatter(self, ax, x0_samples, colors, cmap, time_bins, converged):
        """Plot ROA scatter for pendulum."""
        from matplotlib.patches import Rectangle
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        
        # Separate converged and failed
        failed_mask = ~converged
        
        # Plot failed trajectories
        ax.scatter(x0_samples[failed_mask, 0], x0_samples[failed_mask, 1],
                  c='red', s=10, alpha=0.7, label='Failed')
        
        # Plot converged with time-based colors
        for i in range(len(time_bins) + 1):
            mask = (colors == i) & converged
            if np.any(mask):
                if i == 0:
                    label = f't < {time_bins[0]}s'
                elif i == len(time_bins):
                    label = f't > {time_bins[-1]}s'
                else:
                    label = f'{time_bins[i-1]}s < t < {time_bins[i]}s'
                
                ax.scatter(x0_samples[mask, 0], x0_samples[mask, 1],
                          c=[cmap(i)], s=10, alpha=0.7, label=label)
        
        # Add equilibrium point
        ax.plot(np.pi, 0, 'k*', markersize=10, label='Equilibrium')
        
        # Add verification region
        rect_width = 0.2 * self.bound_level * np.pi
        rect_height = self.bound_level
        rect = Rectangle((np.pi - rect_width/2, -rect_height/2),
                        rect_width, rect_height,
                        linewidth=2, edgecolor='black', facecolor='none',
                        linestyle='--', label='Verification region')
        ax.add_patch(rect)
        
        ax.set_xlabel('θ (rad)')
        ax.set_ylabel('θ̇ (rad/s)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def compute_3d_lyapunov(self, lyapunov_relu, R, grid_resolution=50):
        """Compute 3D Lyapunov for pendulum."""
        theta_range = np.linspace(np.pi - 0.5, np.pi + 0.5, grid_resolution)
        theta_dot_range = np.linspace(-2, 2, grid_resolution)
        
        Theta, ThetaDot = np.meshgrid(theta_range, theta_dot_range)
        V = np.zeros_like(Theta)
        
        # Batch compute for efficiency
        for i in range(grid_resolution):
            batch = torch.tensor(
                np.column_stack((Theta[i, :], ThetaDot[i, :])),
                dtype=self.dtype
            )
            V[i, :] = self.compute_lyapunov_value_batch(batch, lyapunov_relu, R)
        
        return Theta, ThetaDot, V
    
    def get_default_test_points(self):
        """Get default test points for pendulum."""
        return [
            np.array([np.pi + 0.3, 0.5]),
            np.array([np.pi - 0.3, -0.5]),
            np.array([np.pi + 0.1, -1.0]),
        ]
    
    def get_log_file_path(self, seed, use_fpl):
        """Get log file path for pendulum."""
        suffix = "_fpl" if use_fpl else ""
        seed_suffix = f"_seed{seed}" if seed > 0 else ""
        return f"data/monotonic_bound{self.bound_level}{suffix}{seed_suffix}/training_log_seed{seed}.json"

    def create_forward_system(self):
        """Create forward system for ML simulation in pendulum."""
        # Define bounds for the system
        x_lo = torch.tensor(
            [np.pi - 0.1 * self.bound_level * np.pi, -0.5 * self.bound_level], 
            dtype=self.dtype
        )
        x_up = torch.tensor(
            [np.pi + 0.1 * self.bound_level * np.pi, 0.5 * self.bound_level], 
            dtype=self.dtype
        )

        # Move all tensors used inside the forward system to the same device
        x_lo_dev = x_lo.to(self.device)
        x_up_dev = x_up.to(self.device)
        u_lo_dev = self.u_lo.to(self.device)
        u_up_dev = self.u_up.to(self.device)
        x_eq_pos_dev = self.x_equilibrium[:1].to(self.device)  # Only position for second-order system
        u_eq_dev = self.u_equilibrium.to(self.device)
        
        self.forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
            self.dtype,
            x_lo_dev,
            x_up_dev,
            u_lo_dev,
            u_up_dev,
            self.dynamics_model,
            x_eq_pos_dev,
            u_eq_dev,
            dt=0.01
        )


class PathFollowingVisualization(BaseVisualizationFramework):
    """Path following unicycle-specific visualization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.plant = path_following.Path_Following(self.dtype)
        self.x_equilibrium = torch.tensor([0.0, 0.0], dtype=self.dtype)
        self.u_equilibrium = torch.tensor([self.plant.v], dtype=self.dtype)
        self.u_lo = torch.tensor([-10], dtype=self.dtype)
        self.u_up = torch.tensor([10], dtype=self.dtype)
        self.state_labels = ['Distance Error (m)', 'Angle Error (rad)']
        
        # Load dynamics model
        self.load_dynamics_model()
    
    def load_dynamics_model(self):
        """Load the ML dynamics model for path following."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dynamics_model_path = dir_path + "/neural_network_lyapunov/examples/path_following_unicycle/data/preprocess/path_following_unicycle_forward_model.pt"
        self.dynamics_model = torch.load(dynamics_model_path, map_location=self.device)
        self.dynamics_model.to(self.device)
        self.dynamics_model.eval()
        
        # Create forward system
        x_lo = torch.tensor(
            [-0.8 * self.bound_level / 40.0, -0.8 * self.bound_level / 40.0], 
            dtype=self.dtype
        )
        x_up = torch.tensor(
            [0.8 * self.bound_level / 40.0, 0.8 * self.bound_level / 40.0], 
            dtype=self.dtype
        )

        # Ensure all tensors passed into the forward system are on the same device
        self.forward_system = relu_system.ReLUSystemGivenEquilibrium(
            self.dtype,
            x_lo.to(self.device),
            x_up.to(self.device),
            self.u_lo.to(self.device),
            self.u_up.to(self.device),
            self.dynamics_model,
            self.x_equilibrium.to(self.device),
            self.u_equilibrium.to(self.device),
            0.01
        )

    def sample_initial_conditions(self, n_samples):
        """Sample initial conditions for path following."""
        dist_e_samples = np.random.uniform(
            -0.8/40 * self.bound_level,
            0.8/40 * self.bound_level,
            n_samples
        )
        theta_e_samples = np.random.uniform(
            -0.8/40 * self.bound_level,
            0.8/40 * self.bound_level,
            n_samples
        )
        return np.column_stack((dist_e_samples, theta_e_samples))
    
    def plot_roa_scatter(self, ax, x0_samples, colors, cmap, time_bins, converged):
        """Plot ROA scatter for path following."""
        from matplotlib.patches import Rectangle
        
        # Separate converged and failed
        failed_mask = ~converged
        
        # Plot failed trajectories
        ax.scatter(x0_samples[failed_mask, 0], x0_samples[failed_mask, 1],
                  c='red', s=10, alpha=0.7, label='Failed')
        
        # Plot converged with time-based colors
        for i in range(len(time_bins) + 1):
            mask = (colors == i) & converged
            if np.any(mask):
                if i == 0:
                    label = f't < {time_bins[0]}s'
                elif i == len(time_bins):
                    label = f't > {time_bins[-1]}s'
                else:
                    label = f'{time_bins[i-1]}s < t < {time_bins[i]}s'
                
                ax.scatter(x0_samples[mask, 0], x0_samples[mask, 1],
                          c=[cmap(i)], s=10, alpha=0.7, label=label)
        
        # Add equilibrium point
        ax.plot(0, 0, 'k*', markersize=10, label='Equilibrium')
        
        # Add verification region
        rect_width = 0.016 * self.bound_level
        rect_height = 0.016 * self.bound_level
        rect = Rectangle((-rect_width/2, -rect_height/2),
                        rect_width, rect_height,
                        linewidth=2, edgecolor='black', facecolor='none',
                        linestyle='--', label='Verification region')
        ax.add_patch(rect)
        
        ax.set_xlabel('Distance Error (m)')
        ax.set_ylabel('Angle Error (rad)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def compute_3d_lyapunov(self, lyapunov_relu, R, grid_resolution=50):
        """Compute 3D Lyapunov for path following."""
        dist_range = np.linspace(-0.05, 0.05, grid_resolution)
        angle_range = np.linspace(-0.05, 0.05, grid_resolution)
        
        Dist, Angle = np.meshgrid(dist_range, angle_range)
        V = np.zeros_like(Dist)
        
        # Batch compute for efficiency
        for i in range(grid_resolution):
            batch = torch.tensor(
                np.column_stack((Dist[i, :], Angle[i, :])),
                dtype=self.dtype
            )
            V[i, :] = self.compute_lyapunov_value_batch(batch, lyapunov_relu, R)
        
        return Dist, Angle, V
    
    def get_default_test_points(self):
        """Get default test points for path following."""
        return [
            np.array([0.02, 0.02]),
            np.array([-0.02, -0.02]),
            np.array([0.01, -0.015]),
        ]
    
    def get_log_file_path(self, seed, use_fpl):
        """Get log file path for path following."""
        suffix = "_fpl" if use_fpl else ""
        seed_suffix = f"_seed{seed}" if seed > 0 else ""
        return f"data/monotonic_bound{self.bound_level}{suffix}{seed_suffix}/training_log_seed{seed}.json"


def main():
    parser = argparse.ArgumentParser(description='Unified visualization for FPL comparison')
    parser.add_argument('--env', type=str, choices=['pendulum', 'path_following'],
                       required=True, help='Environment to visualize')
    parser.add_argument('--bound_level', type=int, default=3,
                       help='Bound level of trained controllers')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples for ROA')
    parser.add_argument('--time_bins', type=float, nargs='+',
                       default=[0.5, 1.0, 2.0, 5.0],
                       help='Time bins for convergence coloring')
    parser.add_argument('--grid_resolution', type=int, default=50,
                       help='Grid resolution for 3D Lyapunov')
    parser.add_argument('--n_workers', type=int, default=32,
                       help='Number of parallel workers')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                       help='Seeds to aggregate over')
    parser.add_argument('--save_dir', type=str, default='figures',
                       help='Directory to save figures')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save figures')
    
    args = parser.parse_args()
    
    # Create appropriate visualization object
    if args.env == 'pendulum':
        viz = PendulumVisualization(
            bound_level=args.bound_level,
            n_workers=args.n_workers
        )
    else:
        viz = PathFollowingVisualization(
            bound_level=args.bound_level,
            n_workers=args.n_workers
        )
    
    # Generate comprehensive report
    save_dir = None if args.no_save else args.save_dir
    viz.create_comprehensive_report(
        n_samples=args.n_samples,
        time_bins=args.time_bins,
        grid_resolution=args.grid_resolution,
        seeds=args.seeds,
        save_dir=save_dir,
        env=args.env
    )
    
    print(f"Visualization complete! Figures saved to {save_dir}")


if __name__ == "__main__":
    main()