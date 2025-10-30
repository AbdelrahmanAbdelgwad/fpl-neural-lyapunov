import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.integrate
import argparse
import os
from matplotlib.patches import Rectangle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import your modules
import neural_network_lyapunov.examples.pendulum.pendulum as pendulum
import neural_network_lyapunov.utils as utils
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.monotonic_lyapunov.monotonic_utils_0615 as monotonic_utils

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_dynamics_model(dir_path, dt=0.01):
    """Load the ML dynamics model used for training."""
    dynamics_model_path = dir_path + "/data/pendulum_second_order_forward_relu2.pt"
    dynamics_model_data = torch.load(dynamics_model_path, weights_only=False)
    
    dynamics_model = utils.setup_relu(
        dynamics_model_data["linear_layer_width"],
        params=None,
        negative_slope=dynamics_model_data["negative_slope"],
        bias=True,
        dtype=torch.float64,
    )
    dynamics_model.load_state_dict(dynamics_model_data["state_dict"])
    
    # Keep on CPU for initialization
    q_equilibrium = torch.tensor([np.pi], dtype=torch.float64)
    u_equilibrium = torch.tensor([0], dtype=torch.float64)
    bound_level_x = 10
    bound_level_y = 10
    x_lo = torch.tensor(
        [np.pi - 0.1 * bound_level_x * np.pi, -0.5 * bound_level_y], 
        dtype=torch.float64
    )
    x_up = torch.tensor(
        [np.pi + 0.1 * bound_level_x * np.pi, 0.5 * bound_level_y], 
        dtype=torch.float64
    )
    u_lo = torch.tensor([-20], dtype=torch.float64)
    u_up = torch.tensor([20], dtype=torch.float64)
    
    forward_system = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
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
    
    # Move dynamics model to device after initialization
    dynamics_model = dynamics_model.to(device)
    dynamics_model.eval()
    
    return dynamics_model, forward_system

def load_models(bound_level, use_fpl=False):
    """Load saved models for a specific bound level and training type."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    suffix = "_fpl" if use_fpl else ""
    
    base_path = dir_path + f"/data/monotonic_bound{bound_level}{suffix}"
    
    # Load controller
    controller_path = base_path + f"/monotonic_bound{bound_level}{suffix}_controller.pt"
    controller_relu = torch.load(controller_path, weights_only=False)
    controller_relu.eval()
    
    # Load Lyapunov
    lyapunov_path = base_path + f"/monotonic_bound{bound_level}{suffix}_lyapunov.pt"
    lyapunov_relu = torch.load(lyapunov_path, weights_only=False)
    lyapunov_relu.eval()
    
    # Load R matrix
    R_path = base_path + f"/monotonic_bound{bound_level}{suffix}_R.pt"
    R = torch.load(R_path, weights_only=False)
    
    return controller_relu, lyapunov_relu, R

def compute_lyapunov_value_batch(x_batch, lyapunov_relu, x_equilibrium, V_lambda, R):
    """Compute Lyapunov function value for a batch of states."""
    # Move to device if needed
    x_batch = x_batch.to(device)
    lyapunov_relu = lyapunov_relu.to(device)
    x_equilibrium = x_equilibrium.to(device)
    R = R.to(device)

    with torch.no_grad():
        relu_at_equilibrium = lyapunov_relu(x_equilibrium)
        if len(x_batch.shape) == 1:
            x_batch = x_batch.unsqueeze(0)
        V = lyapunov_relu(x_batch) - relu_at_equilibrium + \
            V_lambda * torch.norm(R @ (x_batch - x_equilibrium).T, p=1, dim=0).unsqueeze(1)
    return V.squeeze()

def simulate_trajectory_analytical_fast(plant, controller_relu, x_equilibrium, u_equilibrium, 
                                        u_lo, u_up, x0, T=10.0, dt=0.01, check_convergence=False):
    """Fast simulation with optional early termination."""
    # Ensure everything is on CPU for scipy integration
    if hasattr(x_equilibrium, 'cpu'):
        x_eq_cpu = x_equilibrium.cpu()
        u_eq_cpu = u_equilibrium.cpu()
        u_lo_cpu = u_lo.cpu()
        u_up_cpu = u_up.cpu()
    else:
        x_eq_cpu = x_equilibrium
        u_eq_cpu = u_equilibrium
        u_lo_cpu = u_lo
        u_up_cpu = u_up
    
    # Clone controller to CPU if needed
    if next(controller_relu.parameters()).is_cuda:
        controller_cpu = controller_relu.cpu()
    else:
        controller_cpu = controller_relu
    controller_cpu.eval()
    
    def dynamics(t, x):
        with torch.no_grad():
            x_torch = torch.tensor(x, dtype=torch.float64)
            u_pre_sat = controller_cpu(x_torch) - controller_cpu(x_eq_cpu) + u_eq_cpu
            u = torch.max(torch.min(u_pre_sat, u_up_cpu), u_lo_cpu).detach().numpy()
            return plant.dynamics(x, u)
    
    if check_convergence:
        def convergence_event(t, x):
            dist = np.sqrt((x[0] - np.pi)**2 + x[1]**2)
            if dist > 20:  # Early termination for divergence
                return -1
            return dist - 1e-3

        convergence_event.terminal = True
        convergence_event.direction = -1
        
        result = scipy.integrate.solve_ivp(
            dynamics, [0, T], x0, method='RK45', rtol=1e-4, atol=1e-6,
            events=convergence_event, max_step=0.5
        )
        converged = len(result.t_events[0]) > 0 if result.t_events else False
        return converged, result.t[-1] if converged else T
    else:
        t_eval = np.arange(0, T, dt)
        result = scipy.integrate.solve_ivp(
            dynamics, [0, T], x0, t_eval=t_eval, method='RK45', rtol=1e-6
        )
        
        # Compute control inputs
        with torch.no_grad():
            x_torch = torch.tensor(result.y.T, dtype=torch.float64)
            u_pre_sat = controller_cpu(x_torch) - controller_cpu(x_eq_cpu) + u_eq_cpu
            controls = torch.max(torch.min(u_pre_sat, u_up_cpu), u_lo_cpu).numpy().flatten()
        
        return result.t, result.y, controls

def simulate_trajectories_ml_vectorized(forward_system, controller_relu, x_equilibrium, 
                                       u_equilibrium, u_lo, u_up, x0_batch, T=10.0, 
                                       dt=0.01, check_convergence=False):
    """Vectorized ML simulation for multiple initial conditions at once."""
    num_steps = int(T / dt)
    batch_size = x0_batch.shape[0] if len(x0_batch.shape) > 1 else 1
    
    # Move controller to device if needed
    if device.type == 'cuda' and not next(controller_relu.parameters()).is_cuda:
        controller_relu = controller_relu.to(device)
    
    # Convert to tensors on device
    if len(x0_batch.shape) == 1:
        x_batch = torch.tensor(x0_batch.reshape(1, -1), dtype=torch.float64, device=device)
    else:
        x_batch = torch.tensor(x0_batch, dtype=torch.float64, device=device)
    
    x_eq_device = x_equilibrium.to(device) if hasattr(x_equilibrium, 'to') else torch.tensor(x_equilibrium, device=device)
    u_eq_device = u_equilibrium.to(device) if hasattr(u_equilibrium, 'to') else torch.tensor(u_equilibrium, device=device)
    u_lo_device = u_lo.to(device) if hasattr(u_lo, 'to') else torch.tensor(u_lo, device=device)
    u_up_device = u_up.to(device) if hasattr(u_up, 'to') else torch.tensor(u_up, device=device)
    
    if check_convergence:
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        convergence_times = torch.full((batch_size,), T, dtype=torch.float64, device=device)
        
        with torch.no_grad():
            for step in range(num_steps):
                # Compute control for all trajectories at once
                u_pre_sat = controller_relu(x_batch) - controller_relu(x_eq_device) + u_eq_device
                u_sat = torch.max(torch.min(u_pre_sat, u_up_device), u_lo_device)
                
                # Step forward for all trajectories
                x_batch_new = torch.zeros_like(x_batch)
                for i in range(batch_size):
                    xi = x_batch[i].to(device)
                    ui = u_sat[i].to(device)
                    x_batch_new[i] = forward_system.step_forward(xi, ui)
                x_batch = x_batch_new
                
                # Check convergence
                errors = torch.sqrt((x_batch[:, 0] - np.pi)**2 + x_batch[:, 1]**2)
                newly_converged = (errors < 1e-3) & (~converged)
                convergence_times[newly_converged] = step * dt
                converged |= newly_converged
                
                # # Early termination for divergence
                # diverged = errors > 20
                # x_batch[diverged] = torch.tensor([np.pi, 0], device=device, dtype=torch.float64)
                
                if converged.all():
                    break
        
        return converged.cpu().numpy(), convergence_times.cpu().numpy()
    
    else:
        # For single trajectory visualization
        x0 = x0_batch[0] if len(x0_batch.shape) > 1 else x0_batch
        t = np.arange(0, T, dt)
        x = np.zeros((2, num_steps))
        u = np.zeros(num_steps)
        x[:, 0] = x0
        
        x_curr = torch.tensor(x0, dtype=torch.float64, device=device)
        
        with torch.no_grad():
            for i in range(num_steps - 1):
                u_pre_sat = controller_relu(x_curr) - controller_relu(x_eq_device) + u_eq_device
                u_sat = torch.max(torch.min(u_pre_sat, u_up_device), u_lo_device)
                u[i] = u_sat.cpu().item()
                x_curr_cpu = x_curr.cpu()
                x_next = forward_system.step_forward(x_curr_cpu, u_sat.cpu())
                x_curr = x_next.to(device)
                x[:, i+1] = x_next.cpu().numpy()
        
        return t, x, u

def parallel_analytical_convergence(
    x0_samples,
    controller,
    plant,
    x_equilibrium,
    u_equilibrium,
    u_lo,
    u_up,
    T=50.0,
    max_workers=32,
    stall_timeout=60,     # seconds with no completions -> print a heartbeat
    fail_fast=False       # stop early if any task errors
):
    """
    Runs simulate_trajectory_analytical_fast on many initial states in parallel,
    with a tqdm progress bar + stall detection.

    Returns:
        converged: np.ndarray[bool] of shape (N,)
        conv_times: np.ndarray[float] of shape (N,)
    """
    import time
    import numpy as np
    import concurrent.futures as cf
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    N = len(x0_samples)
    print(f"    Total samples to process: {N} (max_workers={max_workers})")

    def check_single(x0):
        # Wrap the call so exceptions bubble up (we'll catch them at the future)
        return simulate_trajectory_analytical_fast(
            plant, controller, x_equilibrium, u_equilibrium, u_lo, u_up,
            x0, T=T, check_convergence=True
        )

    results = [None] * N
    errors = []
    # Pre-assign indices so we can place results predictably
    idx_map = {idv: i for i, idv in enumerate(range(N))}  # dummy map we’ll override below

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Submit futures and remember which index each corresponds to
        fut_to_idx = {}
        for i, x0 in enumerate(x0_samples):
            fut = ex.submit(check_single, x0)
            fut_to_idx[fut] = i

        # Progress bar
        pbar = tqdm(total=N, desc="    Evaluating", unit="traj", dynamic_ncols=True, mininterval=0.5)

        # Stall detection
        remaining = set(fut_to_idx.keys())
        last_progress = time.time()

        while remaining:
            # Wait for at least one to complete or time out to print a heartbeat
            done, not_done = cf.wait(remaining, timeout=stall_timeout, return_when=cf.FIRST_COMPLETED)

            if not done:
                # Nothing finished in stall_timeout seconds -> heartbeat
                elapsed = int(time.time() - last_progress)
                print(f"    ⏳ No completions in the last {elapsed}s "
                      f"(still running: {len(not_done)}/{N})")
                continue

            # Consume all completed futures
            for fut in done:
                remaining.remove(fut)
                i = fut_to_idx[fut]
                try:
                    conv, t_conv = fut.result()
                except Exception as e:
                    errors.append((i, e))
                    # Put a fallback so array construction still works
                    conv, t_conv = False, float(T)
                    if fail_fast:
                        pbar.close()
                        # Optionally cancel remaining
                        for f in not_done:
                            f.cancel()
                        raise RuntimeError(
                            f"Task {i} failed during parallel_analytical_convergence"
                        ) from e

                results[i] = (bool(conv), float(t_conv))
                pbar.update(1)
                last_progress = time.time()

        pbar.close()

    if errors:
        # Summarize errors at the end instead of silently swallowing them
        print(f"    ⚠️ {len(errors)} task(s) raised exceptions. "
              f"Example: index {errors[0][0]} -> {repr(errors[0][1])}")

    # Unzip results into arrays (all slots are filled; failed ones use (False, T))
    converged = np.array([r[0] for r in results], dtype=bool)
    conv_times = np.array([r[1] for r in results], dtype=float)
    print("    Parallel processing completed.")
    return converged, conv_times


def compute_convergence_regions_fast(configurations, n_samples, x_lo, x_up,
                                    plant, forward_system, x_equilibrium, 
                                    u_equilibrium, u_lo, u_up):
    """Compute convergence regions using vectorized and parallel operations."""
    results = {}
    
    # Sample initial conditions
    np.random.seed(42)
    x_lo_np = x_lo.cpu().numpy() if hasattr(x_lo, 'cpu') else x_lo.numpy()
    x_up_np = x_up.cpu().numpy() if hasattr(x_up, 'cpu') else x_up.numpy()
    
    theta_samples = np.random.uniform(x_lo_np[0], x_up_np[0], n_samples)
    theta_dot_samples = np.random.uniform(x_lo_np[1], x_up_np[1], n_samples)
    x0_samples = np.column_stack((theta_samples, theta_dot_samples))
    
    for controller, dynamics_type, title in configurations:
        print(f"  Computing {title}...")
        
        if dynamics_type == 'ML':
            # Process in smaller batches to avoid memory issues
            batch_size = min(100, n_samples)
            converged = np.zeros(n_samples, dtype=bool)
            
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                batch = x0_samples[i:end_idx]
                batch_conv, _ = simulate_trajectories_ml_vectorized(
                    forward_system, controller, x_equilibrium, u_equilibrium,
                    u_lo, u_up, batch, T=40.0, dt=0.01, check_convergence=True
                )
                converged[i:end_idx] = batch_conv

                if i % 100 == 0:
                    print(f"    Progress: {i}/{n_samples} samples")
        else:
            # Parallel analytical simulation
            converged, _ = parallel_analytical_convergence(
                x0_samples, controller, plant, x_equilibrium,
                u_equilibrium, u_lo, u_up, T=40.0
            )
        
        results[title] = (theta_samples, theta_dot_samples, converged)
        print(f"    Convergence rate: {np.mean(converged)*100:.1f}%")
    
    return results

def create_full_comparison_plots_fast(bound_level=3, V_lambda=0.6):
    """Optimized version with GPU acceleration and parallel processing."""
    
    # Set up the systems
    dtype = torch.float64
    plant = pendulum.Pendulum(dtype)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dynamics_model, forward_system = load_dynamics_model(dir_path, dt=0.01)
    
    x_equilibrium = torch.tensor([np.pi, 0], dtype=dtype)
    u_equilibrium = torch.tensor([0], dtype=dtype)
    u_lo = torch.tensor([-20], dtype=dtype)
    u_up = torch.tensor([20], dtype=dtype)
    
    # Load models
    controller_fpl, lyapunov_fpl, R_fpl = load_models(bound_level, use_fpl=True)
    controller_std, lyapunov_std, R_std = load_models(bound_level, use_fpl=False)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Test initial conditions
    x_lo = torch.tensor(
        [np.pi - 0.1 * bound_level * np.pi, -0.5 * bound_level * np.pi], 
        dtype=torch.float64
    )
    x_up = torch.tensor(
        [np.pi + 0.1 * bound_level * np.pi, 0.5 * bound_level * np.pi], 
        dtype=torch.float64
    )
    
    test_x0s = [
    np.array([
        (x_up[0].detach().cpu().item() if hasattr(x_up, 'detach') else float(x_up[0])) / 3,
        (x_up[1].detach().cpu().item() if hasattr(x_up, 'detach') else float(x_up[1])) / 3
    ]),
    np.array([
        (x_lo[0].detach().cpu().item() if hasattr(x_lo, 'detach') else float(x_lo[0])) / 3,
        (x_lo[1].detach().cpu().item() if hasattr(x_lo, 'detach') else float(x_lo[1])) / 3
    ])
    
]
    print("Test initial conditions:", test_x0s)

    def _move_obj_to_device(obj, device):
        import torch, torch.nn as nn
        # Move modules
        if isinstance(obj, nn.Module):
            obj.to(device)
        # Move any tensors attached as attributes
        for name, val in vars(obj).items():
            if isinstance(val, torch.Tensor):
                setattr(obj, name, val.to(device))
            elif isinstance(val, nn.Module):
                val.to(device)

    # After you construct/load forward_system:
    _move_obj_to_device(forward_system, device)


    # 1. Phase portraits
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    colors = ['blue', 'green']
    
    for idx, x0 in enumerate(test_x0s):
        # Standard controller
        t_std, y_std, u_std = simulate_trajectory_analytical_fast(
            plant, controller_std, x_equilibrium, u_equilibrium, 
            u_lo, u_up, x0, T=25.0
        )
        ax1.plot(y_std[0], y_std[1], '-', color=colors[idx], 
                label=f'IC{idx+1}', alpha=0.8, linewidth=1.5)
        
        # FPL controller
        t_fpl, y_fpl, u_fpl = simulate_trajectory_analytical_fast(
            plant, controller_fpl, x_equilibrium, u_equilibrium,
            u_lo, u_up, x0, T=25.0
        )
        ax2.plot(y_fpl[0], y_fpl[1], '-', color=colors[idx],
                label=f'IC{idx+1}', alpha=0.8, linewidth=1.5)
    
    ax1.plot(np.pi, 0, 'r*', markersize=12, label='Equilibrium')
    ax1.set_xlabel('θ (rad)')
    ax1.set_ylabel('θ̇ (rad/s)')
    ax1.set_title('Standard Controller Trajectories')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(np.pi, 0, 'r*', markersize=12, label='Equilibrium')
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('θ̇ (rad/s)')
    ax2.set_title('FPL Controller Trajectories')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 2. Lyapunov function evolution
    ax3 = fig.add_subplot(gs[1, 0])
    x0 = test_x0s[0]
    
    t_std, y_std, _ = simulate_trajectory_analytical_fast(
        plant, controller_std, x_equilibrium, u_equilibrium, u_lo, u_up, x0, T=25.0
    )
    t_fpl, y_fpl, _ = simulate_trajectory_analytical_fast(
        plant, controller_fpl, x_equilibrium, u_equilibrium, u_lo, u_up, x0, T=25.0
    )
    
    # Vectorized Lyapunov computation
    with torch.no_grad():
        y_std_tensor = torch.tensor(y_std.T, dtype=dtype)
        y_fpl_tensor = torch.tensor(y_fpl.T, dtype=dtype)
        
        V_std = compute_lyapunov_value_batch(
            y_std_tensor, lyapunov_std, x_equilibrium, V_lambda, R_std
        ).cpu().numpy()
        V_fpl = compute_lyapunov_value_batch(
            y_fpl_tensor, lyapunov_fpl, x_equilibrium, V_lambda, R_fpl
        ).cpu().numpy()
    
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
    
    test_angles = np.linspace(np.pi - 0.2*bound_level, np.pi + 0.2*bound_level, 2)
    test_velocities = np.linspace(-0.3*bound_level, 0.3*bound_level, 2)
    
    for controller, name in [(controller_std, 'Standard'), (controller_fpl, 'FPL')]:
        settling_list = []
        effort_list = []
        
        for theta0 in test_angles:
            for theta_dot0 in test_velocities:
                x0 = np.array([theta0, theta_dot0])
                t, y, u = simulate_trajectory_analytical_fast(
                    plant, controller, x_equilibrium, u_equilibrium,
                    u_lo, u_up, x0, T=25.0
                )
                
                errors = np.sqrt((y[0]-np.pi)**2 + y[1]**2)
                settling_idx = np.where(errors < 0.01)[0]
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
    
    t_std, y_std, u_std = simulate_trajectory_analytical_fast(
        plant, controller_std, x_equilibrium, u_equilibrium, u_lo, u_up, x0, T=25.0
    )
    t_fpl, y_fpl, u_fpl = simulate_trajectory_analytical_fast(
        plant, controller_fpl, x_equilibrium, u_equilibrium, u_lo, u_up, x0, T=25.0
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
    x_lo_test = torch.tensor(
        [np.pi - 0.15 * bound_level * np.pi, -0.6 * bound_level], 
        dtype=torch.float64
    )
    x_up_test = torch.tensor(
        [np.pi + 0.15 * bound_level * np.pi, 0.6 * bound_level], 
        dtype=torch.float64
    )
    
    # Adjust sample count for speed
    n_samples = min(500, 400 + 100 * bound_level)

    configurations = [
        (controller_std, 'Analytical', 'Std-Analytical'),
        (controller_std, 'ML', 'Std-ML'), 
        (controller_fpl, 'Analytical', 'FPL-Analytical'),
        (controller_fpl, 'ML', 'FPL-ML')
    ]
    
    # Compute convergence regions
    results = compute_convergence_regions_fast(
        configurations, n_samples, x_lo_test, x_up_test,
        plant, forward_system, x_equilibrium, 
        u_equilibrium, u_lo, u_up
    )
    
    # Plot scatter points
    for ax_idx, (_, _, title) in enumerate(configurations):
        theta_vals, theta_dot_vals, converged = results[title]
        
        # Create scatter plot
        mask_conv = converged
        mask_not_conv = ~converged
        
        # Plot points
        ax = axes_conv[ax_idx]
        ax.scatter(theta_vals[mask_not_conv], theta_dot_vals[mask_not_conv], 
                  c='red', s=10, alpha=0.7, edgecolors='none', label='Does not converge')
        ax.scatter(theta_vals[mask_conv], theta_dot_vals[mask_conv], 
                  c='green', s=10, alpha=0.7, edgecolors='none', label='Converges')
        
        # Add equilibrium point
        ax.plot(np.pi, 0, 'b*', markersize=10, zorder=5)
        
        # Add verification region box
        rect_width = 0.2 * bound_level * np.pi
        rect_height = bound_level
        rect = Rectangle((np.pi - rect_width/2, -rect_height/2), 
                        rect_width, rect_height, 
                        linewidth=2, edgecolor='black', facecolor='none', 
                        zorder=4, linestyle='--', label='Verification region')
        ax.add_patch(rect)
        
        ax.set_xlabel('θ (rad)')
        ax.set_ylabel('θ̇ (rad/s)')
        ax.set_title(f'{title}')
        ax.set_xlim(x_lo_test[0].item(), x_up_test[0].item())
        ax.set_ylim(x_lo_test[1].item(), x_up_test[1].item())
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
    
    fig.suptitle(f'Controller Comparison (Bound Level {bound_level})', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and compare pendulum controllers')
    parser.add_argument('--bound_level', type=int, default=3, 
                       help='Bound level of the trained controllers')
    parser.add_argument('--save_figs', action='store_true',
                       help='Save figures to files')
    args = parser.parse_args()
    
    # Use optimized version
    fig = create_full_comparison_plots_fast(bound_level=args.bound_level)
    
    if args.save_figs:
        fig.savefig(f'comparison_fast_bound{args.bound_level}.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved to comparison_fast_bound{args.bound_level}.png")
    
    plt.show()