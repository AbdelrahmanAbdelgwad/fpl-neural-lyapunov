import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.integrate
import argparse
import os
from matplotlib.patches import Rectangle

# Import your modules
import neural_network_lyapunov.examples.path_following_unicycle.path_following as path_following
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils_0615 as monotonic_utils

def load_dynamics_model(dir_path, dt=0.01):
    """Load the ML dynamics model used for training."""
    dynamics_model_path = dir_path + "/data/preprocess/path_following_unicycle_forward_model.pt"
    device = torch.device("cpu")
    dynamics_model = torch.load(dynamics_model_path, map_location=device)
    dynamics_model = dynamics_model.to(device)
    
    # Create forward system wrapper
    q_equilibrium = torch.tensor([0.0, 0.0], dtype=torch.float64)
    u_equilibrium = torch.tensor([6.0], dtype=torch.float64)  # v = 6 from path_following.py
    bound_level_x = 10
    bound_level_y = 10
    x_lo = torch.tensor(
        [-0.8 * bound_level_x / 40.0, -0.8 * bound_level_y / 40.0], dtype=torch.float64
    )
    x_up = torch.tensor(
        [0.8 * bound_level_x / 40.0, 0.8 * bound_level_y / 40.0], dtype=torch.float64
    )
    u_lo = torch.tensor([-10], dtype=torch.float64)
    u_up = torch.tensor([10], dtype=torch.float64)
    
    forward_system = relu_system.ReLUSystemGivenEquilibrium(
        torch.float64,
        x_lo,
        x_up,
        u_lo,
        u_up,
        dynamics_model,
        q_equilibrium,
        u_equilibrium,
        dt,
    )
    
    return dynamics_model, forward_system

def load_models(bound_level, use_fpl=False):
    """Load saved models for a specific bound level and training type."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    suffix = "_fpl" if use_fpl else ""
    device = torch.device("cpu")
    
    base_path = dir_path + f"/data/monotonic/monotonic_bound{bound_level}{suffix}"
    
    # Load controller
    controller_path = base_path + f"/monotonic_bound{bound_level}{suffix}_controller.pt"
    controller_relu = torch.load(controller_path, map_location=device)
    controller_relu = controller_relu.to(device)
    
    # Load Lyapunov
    lyapunov_path = base_path + f"/monotonic_bound{bound_level}{suffix}_lyapunov.pt"
    lyapunov_relu = torch.load(lyapunov_path, map_location=device)
    lyapunov_relu = lyapunov_relu.to(device)
    
    # Load R matrix
    R_path = base_path + f"/monotonic_bound{bound_level}{suffix}_R.pt"
    R = torch.load(R_path, map_location=device)
    R = R.to(device) if isinstance(R, torch.Tensor) else R
    
    return controller_relu, lyapunov_relu, R

def compute_lyapunov_value_batch(x_batch, lyapunov_relu, x_equilibrium, V_lambda, R):
    """Compute Lyapunov function value for a batch of states."""
    device = torch.device("cpu")
    if len(x_batch.shape) == 1:
        x_batch = x_batch.unsqueeze(0)
    x_batch = x_batch.to(device)
    x_equilibrium = x_equilibrium.to(device)
    R = R.to(device) if isinstance(R, torch.Tensor) else torch.tensor(R, device=device)
    
    relu_at_equilibrium = lyapunov_relu(x_equilibrium)
    V = lyapunov_relu(x_batch) - relu_at_equilibrium + \
        V_lambda * torch.norm(R @ (x_batch - x_equilibrium).T, p=1, dim=0).unsqueeze(1)
    return V.squeeze()

def simulate_trajectory_analytical_fast(plant, controller_relu, x_equilibrium, u_equilibrium, 
                                        u_lo, u_up, x0, T=10.0, dt=0.01, check_convergence=False):
    """Fast simulation with optional early termination."""
    def dynamics(t, x):
        with torch.no_grad():
            x_torch = torch.tensor(x, dtype=torch.float64)
            u_pre_sat = controller_relu(x_torch) - controller_relu(x_equilibrium) + u_equilibrium
            u = torch.max(torch.min(u_pre_sat, u_up), u_lo).detach().numpy()
            return plant.dynamics(x, u)
    
    if check_convergence:
        # Use event detection for early termination
        def convergence_event(t, x):
            return np.sqrt(x[0]**2 + x[1]**2) - 1e-3

        convergence_event.terminal = True
        convergence_event.direction = -1
        
        result = scipy.integrate.solve_ivp(
            dynamics, [0, T], x0, method='DOP853', rtol=1e-6, atol=1e-8,
            events=convergence_event, dense_output=False
        )
        converged = len(result.t_events[0]) > 0 if result.t_events else False
        return converged, result.t[-1]
    else:
        t_eval = np.arange(0, T, dt)
        result = scipy.integrate.solve_ivp(
            dynamics, [0, T], x0, t_eval=t_eval, method='RK45', rtol=1e-8
        )
        
        # Compute control inputs
        with torch.no_grad():
            x_torch = torch.tensor(result.y.T, dtype=torch.float64)
            u_pre_sat = controller_relu(x_torch) - controller_relu(x_equilibrium) + u_equilibrium
            controls = torch.max(torch.min(u_pre_sat, u_up), u_lo).numpy().flatten()
        
        return result.t, result.y, controls

def simulate_trajectory_ml_fast(forward_system, controller_relu, x_equilibrium, u_equilibrium,
                                u_lo, u_up, x0, T=10.0, dt=0.01, check_convergence=False):
    """Fast ML simulation with optional convergence checking."""
    num_steps = int(T / dt)
    device = torch.device("cpu")
    
    if check_convergence:
        x_curr = torch.tensor(x0, dtype=torch.float64, device=device)
        with torch.no_grad():
            for _ in range(num_steps):
                u_pre_sat = controller_relu(x_curr) - controller_relu(x_equilibrium) + u_equilibrium
                u_sat = torch.max(torch.min(u_pre_sat, u_up), u_lo)
                x_curr = forward_system.step_forward(x_curr, u_sat)
                
                error = torch.sqrt(x_curr[0]**2 + x_curr[1]**2)
                if error < 1e-3:
                    return True, _ * dt
        return False, T
    else:
        t = np.arange(0, T, dt)
        x = np.zeros((2, num_steps))
        u = np.zeros(num_steps)
        x[:, 0] = x0
        
        with torch.no_grad():
            for i in range(num_steps - 1):
                x_curr = torch.tensor(x[:, i], dtype=torch.float64, device=device)
                u_pre_sat = controller_relu(x_curr) - controller_relu(x_equilibrium) + u_equilibrium
                u_sat = torch.max(torch.min(u_pre_sat, u_up), u_lo)
                u[i] = u_sat.item()
                x_next = forward_system.step_forward(x_curr, u_sat)
                x[:, i+1] = x_next.cpu().numpy()
        
        return t, x, u

def compute_convergence_regions_vectorized(configurations, n_samples, x_lo, x_up,
                                          plant, forward_system, x_equilibrium, 
                                          u_equilibrium, u_lo, u_up, use_ml_dynamics):
    """Compute convergence regions using vectorized operations where possible."""
    results = {}
    
    # Sample initial conditions
    np.random.seed(42)  # For reproducibility
    dist_e_samples = np.random.uniform(x_lo[0].item(), x_up[0].item(), n_samples)
    theta_e_samples = np.random.uniform(x_lo[1].item(), x_up[1].item(), n_samples)
    
    for controller, dynamics_type, title in configurations:
        print(f"  Computing {title}...")
        converged = np.zeros(n_samples, dtype=bool)
        
        # Process in batches to show progress
        batch_size = 100
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            
            if batch_idx % 10 == 0:
                print(f"    Progress: {batch_idx}/{n_batches} batches")
            
            for i in range(start_idx, end_idx):
                x0 = np.array([dist_e_samples[i], theta_e_samples[i]])
                
                try:
                    if dynamics_type == 'Analytical' or (dynamics_type == 'Selected' and not use_ml_dynamics):
                        conv, _ = simulate_trajectory_analytical_fast(
                            plant, controller, x_equilibrium, u_equilibrium,
                            u_lo, u_up, x0, T=4.0, check_convergence=True
                        )
                        converged[i] = conv
                    else:  # ML or (Selected and use_ml_dynamics)
                        conv, _ = simulate_trajectory_ml_fast(
                            forward_system, controller, x_equilibrium, u_equilibrium,
                            u_lo, u_up, x0, T=4.0, dt=0.01, check_convergence=True
                        )
                        converged[i] = conv
                except:
                    converged[i] = False
        
        results[title] = (dist_e_samples, theta_e_samples, converged)
    
    return results

def create_full_comparison_plots_optimized(bound_level=3, V_lambda=0.6, use_ml_dynamics=False):
    """Optimized version with vectorized operations and scatter plot visualization."""
    
    # Set up the systems
    dtype = torch.float64
    plant = path_following.Path_Following(dtype)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dynamics_model, forward_system = load_dynamics_model(dir_path, dt=0.01)
    
    x_equilibrium = torch.tensor([0.0, 0.0], dtype=dtype)
    u_equilibrium = torch.tensor([plant.v], dtype=dtype)
    u_lo = torch.tensor([-10], dtype=dtype)
    u_up = torch.tensor([10], dtype=dtype)
    
    # Load models
    controller_fpl, lyapunov_fpl, R_fpl = load_models(bound_level, use_fpl=True)
    controller_std, lyapunov_std, R_std = load_models(bound_level, use_fpl=False)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Test initial conditions within the verification region using the optimized bounds
    x_lo = torch.tensor(
        [-0.8 * bound_level / 40.0, -0.8 * bound_level / 40.0], dtype=torch.float64
    )
    x_up = torch.tensor(
        [0.8 * bound_level / 40.0, 0.8 * bound_level / 40.0], dtype=torch.float64
    )
    
    test_x0s = [
        np.array([x_up[0].item()/3, x_up[1].item()/3]),  # Top-right corner
        np.array([x_lo[0].item()/3, x_lo[1].item()/3])   # Bottom-left corner
    ]

    # Helper function to select simulation method
    def simulate_trajectory(plant, forward_system, controller, x_equilibrium, u_equilibrium,
                           u_lo, u_up, x0, T=25.0, use_ml=False):
        if use_ml:
            return simulate_trajectory_ml_fast(
                forward_system, controller, x_equilibrium, u_equilibrium,
                u_lo, u_up, x0, T=T
            )
        else:
            return simulate_trajectory_analytical_fast(
                plant, controller, x_equilibrium, u_equilibrium,
                u_lo, u_up, x0, T=T
            )

    # 1. Phase portraits
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    colors = ['blue', 'green']
    
    for idx, x0 in enumerate(test_x0s):
        # Standard controller
        t_std, y_std, u_std = simulate_trajectory(
            plant, forward_system, controller_std, x_equilibrium, u_equilibrium, 
            u_lo, u_up, x0, T=25.0, use_ml=use_ml_dynamics
        )
        ax1.plot(y_std[0], y_std[1], '-', color=colors[idx], 
                label=f'IC{idx+1}', alpha=0.8, linewidth=1.5)
        
        # FPL controller
        t_fpl, y_fpl, u_fpl = simulate_trajectory(
            plant, forward_system, controller_fpl, x_equilibrium, u_equilibrium,
            u_lo, u_up, x0, T=25.0, use_ml=use_ml_dynamics
        )
        ax2.plot(y_fpl[0], y_fpl[1], '-', color=colors[idx],
                label=f'IC{idx+1}', alpha=0.8, linewidth=1.5)
    
    ax1.plot(0, 0, 'r*', markersize=12, label='Equilibrium')
    ax1.set_xlabel('Distance Error (m)')
    ax1.set_ylabel('Angle Error (rad)')
    ax1.set_title('Standard Controller Trajectories')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(0, 0, 'r*', markersize=12, label='Equilibrium')
    ax2.set_xlabel('Distance Error (m)')
    ax2.set_ylabel('Angle Error (rad)')
    ax2.set_title('FPL Controller Trajectories')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 2. Lyapunov function evolution
    ax3 = fig.add_subplot(gs[1, 0])
    x0 = test_x0s[0]
    
    t_std, y_std, _ = simulate_trajectory(
        plant, forward_system, controller_std, x_equilibrium, u_equilibrium,
        u_lo, u_up, x0, T=25.0, use_ml=use_ml_dynamics
    )
    t_fpl, y_fpl, _ = simulate_trajectory(
        plant, forward_system, controller_fpl, x_equilibrium, u_equilibrium,
        u_lo, u_up, x0, T=25.0, use_ml=use_ml_dynamics
    )
    
    # Vectorized Lyapunov computation
    with torch.no_grad():
        V_std = compute_lyapunov_value_batch(
            torch.tensor(y_std.T, dtype=dtype), 
            lyapunov_std, x_equilibrium, V_lambda, R_std
        ).numpy()
        V_fpl = compute_lyapunov_value_batch(
            torch.tensor(y_fpl.T, dtype=dtype),
            lyapunov_fpl, x_equilibrium, V_lambda, R_fpl
        ).numpy()
    
    ax3.semilogy(t_std, V_std, 'b--', label='Standard', linewidth=1.5)
    ax3.semilogy(t_fpl, V_fpl, 'r-', label='FPL', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('V(x)')
    ax3.set_title('Lyapunov Evolution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 3. Performance metrics
    ax4 = fig.add_subplot(gs[1, 1:3])
    
    metrics_labels = ['Standard', 'FPL']
    settling_times = []
    control_efforts = []
    
    test_dist_e = np.linspace(-0.02*bound_level, 0.02*bound_level, 2)
    test_theta_e = np.linspace(-0.02*bound_level, 0.02*bound_level, 2)
    
    for controller, name in [(controller_std, 'Standard'), (controller_fpl, 'FPL')]:
        settling_list = []
        effort_list = []
        
        for d_e in test_dist_e:
            for th_e in test_theta_e:
                x0 = np.array([d_e, th_e])
                if use_ml_dynamics:
                    t, y, u = simulate_trajectory_ml_fast(
                        forward_system, controller, x_equilibrium, u_equilibrium,
                        u_lo, u_up, x0, T=25.0
                    )
                else:
                    t, y, u = simulate_trajectory_analytical_fast(
                        plant, controller, x_equilibrium, u_equilibrium,
                        u_lo, u_up, x0, T=25.0
                    )
                
                errors = np.sqrt(y[0]**2 + y[1]**2)
                settling_idx = np.where(errors < 0.001)[0]
                settling_time = t[settling_idx[0]] if len(settling_idx) > 0 else 10.0
                settling_list.append(settling_time)
                effort_list.append(np.trapz(np.abs(u), t))
        
        settling_times.append(np.mean(settling_list))
        control_efforts.append(np.mean(effort_list))
    
    x_pos = np.arange(len(metrics_labels))
    width = 0.35
    
    ax4.bar(x_pos - width/2, settling_times, width, label='Settling Time (s)', alpha=0.7)
    ax4.bar(x_pos + width/2, control_efforts, width, label='Control Effort', alpha=0.7)
    ax4.set_xlabel('Controller')
    ax4.set_ylabel('Metric Value')
    ax4.set_title('Performance Metrics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 4. Control effort over time
    ax5 = fig.add_subplot(gs[1, 3])
    x0 = test_x0s[0]
    
    t_std, y_std, u_std = simulate_trajectory(
        plant, forward_system, controller_std, x_equilibrium, u_equilibrium,
        u_lo, u_up, x0, T=25.0, use_ml=use_ml_dynamics
    )
    t_fpl, y_fpl, u_fpl = simulate_trajectory(
        plant, forward_system, controller_fpl, x_equilibrium, u_equilibrium,
        u_lo, u_up, x0, T=25.0, use_ml=use_ml_dynamics
    )
    
    ax5.plot(t_std, u_std, 'b--', label='Standard', linewidth=1.5)
    ax5.plot(t_fpl, u_fpl, 'r-', label='FPL', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Control Input')
    ax5.set_title('Control Effort')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 5. Convergence regions with scatter plot
    print("Computing convergence regions...")
    axes_conv = [fig.add_subplot(gs[2, i]) for i in range(4)]
    
    # Define test region
    x_lo = torch.tensor(
        [-1.2 * bound_level / 40.0, -1.2 * bound_level / 40.0], dtype=torch.float64
    )
    x_up = torch.tensor(
        [1.2 * bound_level / 40.0, 1.2 * bound_level / 40.0], dtype=torch.float64
    )
    
    # Number of samples
    n_samples = 1000  # Fixed for consistency

    configurations = [
        (controller_std, 'Analytical', 'Std-Analytical'),
        (controller_std, 'ML', 'Std-ML'), 
        (controller_fpl, 'Analytical', 'FPL-Analytical'),
        (controller_fpl, 'ML', 'FPL-ML')
    ]
    
    # If use_ml_dynamics is True, change configurations to use selected dynamics
    if use_ml_dynamics:
        dynamics_label = 'ML' if use_ml_dynamics else 'Analytical'
        configurations = [
            (controller_std, 'Selected', f'Std-{dynamics_label}'),
            (controller_std, 'Selected', f'Std-{dynamics_label}'), 
            (controller_fpl, 'Selected', f'FPL-{dynamics_label}'),
            (controller_fpl, 'Selected', f'FPL-{dynamics_label}')
        ]
    
    # Compute convergence regions
    results = compute_convergence_regions_vectorized(
        configurations, n_samples, x_lo, x_up,
        plant, forward_system, x_equilibrium, 
        u_equilibrium, u_lo, u_up, use_ml_dynamics
    )
    
    # Plot scatter points
    for ax_idx, (_, _, title) in enumerate(configurations):
        dist_e_vals, theta_e_vals, converged = results[title]
        
        # Create scatter plot
        mask_conv = converged
        mask_not_conv = ~converged
        
        # Plot points
        ax = axes_conv[ax_idx]
        ax.scatter(dist_e_vals[mask_not_conv], theta_e_vals[mask_not_conv], 
                  c='red', s=10, alpha=0.7, edgecolors='none', label='Does not converge')
        ax.scatter(dist_e_vals[mask_conv], theta_e_vals[mask_conv], 
                  c='green', s=10, alpha=0.7, edgecolors='none', label='Converges')
        
        # Add equilibrium point
        ax.plot(0, 0, 'b*', markersize=10, zorder=5)
        
        # Add verification region box
        rect_width = 0.8 * bound_level / 40.0 * 2
        rect_height = 0.8 * bound_level / 40.0 * 2
        rect = Rectangle((-rect_width/2, -rect_height/2), 
                        rect_width, rect_height, 
                        linewidth=2, edgecolor='black', facecolor='none', 
                        zorder=4, linestyle='--', label='Verification region')
        ax.add_patch(rect)
        
        ax.set_xlabel('Distance Error (m)')
        ax.set_ylabel('Angle Error (rad)')
        ax.set_title(f'{title}')
        ax.set_xlim(x_lo[0].item(), x_up[0].item())
        ax.set_ylim(x_lo[1].item(), x_up[1].item())
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
        
        # Add legend for first plot only
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Add convergence statistics
    print("\nConvergence Statistics:")
    for title in results.keys():
        _, _, converged = results[title]
        conv_rate = np.mean(converged) * 100
        print(f"  {title}: {conv_rate:.1f}% convergence rate")
    
    dynamics_type = "ML Dynamics" if use_ml_dynamics else "Analytical Dynamics"
    fig.suptitle(f'Path Following Controller Comparison (Bound Level {bound_level}, {dynamics_type})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and compare path following controllers')
    parser.add_argument('--bound_level', type=int, default=3, 
                       help='Bound level of the trained controllers')
    parser.add_argument('--use_ml_dynamics', action='store_true',
                       help='Use ML dynamics model instead of analytical')
    parser.add_argument('--save_figs', action='store_true',
                       help='Save figures to files')
    args = parser.parse_args()
    
    # Use optimized version
    fig = create_full_comparison_plots_optimized(
        bound_level=args.bound_level, 
        use_ml_dynamics=args.use_ml_dynamics
    )
    
    if args.save_figs:
        dynamics_suffix = "_ml" if args.use_ml_dynamics else "_analytical"
        filename = f'path_following_comparison_bound{args.bound_level}{dynamics_suffix}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nSaved to {filename}")
    
    plt.show()