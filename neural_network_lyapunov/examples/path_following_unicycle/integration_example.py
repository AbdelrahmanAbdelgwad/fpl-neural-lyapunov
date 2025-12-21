"""
Integration example: Using ReLU-based unicycle dynamics with neural_network_lyapunov

This shows how to replace:
    dynamics_relu = torch.load(dynamics_model_path, map_location=device)

With the analytical ReLU-based trigonometric approximation.
"""

import torch
import numpy as np

# Import the ReLU unicycle dynamics module
from relu_unicycle_dynamics import (
    ReLUUnicycleDynamics,
    convert_to_standard_relu_network,
    create_exact_relu_dynamics_network,
    load_relu_unicycle_dynamics
)


def example_integration():
    """
    Example showing how to integrate with your training script.
    """
    
    # ========================================
    # Configuration (match your script)
    # ========================================
    dt = 0.1  # time step
    v = 1.0   # constant forward velocity (from plant.v)
    dtype = torch.float64
    device = torch.device("cpu")
    
    # State bounds (from your script)
    x_lo = torch.tensor([-1.0, -np.pi], dtype=dtype)  # [e_y_min, e_theta_min]
    x_up = torch.tensor([1.0, np.pi], dtype=dtype)    # [e_y_max, e_theta_max]
    
    # Control bounds
    u_lo = torch.tensor([-10], dtype=dtype)
    u_up = torch.tensor([10], dtype=dtype)
    
    # Equilibrium
    q_equilibrium = torch.tensor([0.0, 0.0], dtype=dtype)
    u_equilibrium = torch.tensor([v], dtype=dtype)  # or [0.0] depending on your formulation
    
    # ========================================
    # Option 1: Quick replacement (for training only, not MIP-compatible)
    # ========================================
    print("Option 1: Direct ReLU dynamics (training only)")
    dynamics_direct = ReLUUnicycleDynamics(dt, v, nb_points=17, dtype=dtype)
    
    # Test
    xu_test = torch.tensor([0.1, 0.3, 0.5], dtype=dtype)
    x_next = dynamics_direct(xu_test)
    print(f"  Input: {xu_test}")
    print(f"  Output: {x_next}")
    
    # ========================================
    # Option 2: Standard ReLU network (MIP-compatible via training)
    # ========================================
    print("\nOption 2: Trained standard ReLU network (MIP-compatible)")
    dynamics_relu = convert_to_standard_relu_network(
        dt=dt,
        v=v,
        nb_points=17,
        hidden_sizes=(32, 32),
        dtype=dtype,
        num_epochs=50,
        verbose=True
    )
    
    # Verify dimensions
    print(f"  Input features: {dynamics_relu[0].in_features}")
    print(f"  Output features: {dynamics_relu[-1].out_features}")
    
    # ========================================
    # Option 3: Exact analytical ReLU network
    # ========================================
    print("\nOption 3: Exact analytical ReLU network")
    dynamics_exact = create_exact_relu_dynamics_network(dt, v, nb_points=17, dtype=dtype)
    x_next_exact = dynamics_exact(xu_test)
    print(f"  Output: {x_next_exact}")
    
    # ========================================
    # Integration with ReLUSystemGivenEquilibrium
    # ========================================
    print("\n" + "="*50)
    print("Integration with your training script:")
    print("="*50)
    
    integration_code = '''
# In your monotonic_train_path_following_demo.py, replace:

# OLD:
# dynamics_relu = torch.load(dynamics_model_path, map_location=device)

# NEW (Option A - trained approximation):
from relu_unicycle_dynamics import convert_to_standard_relu_network

dynamics_relu = convert_to_standard_relu_network(
    dt=dt,
    v=plant.v,
    nb_points=17,
    hidden_sizes=(32, 32),
    dtype=torch.float64,
    num_epochs=100
)

# Then continue with:
forward_system = relu_system.ReLUSystemGivenEquilibrium(
    torch.float64,
    x_lo,
    x_up,
    u_lo,
    u_up,
    dynamics_relu,  # <-- Use the ReLU approximation
    q_equilibrium,
    u_equilibrium,
    dt,
)
'''
    print(integration_code)
    
    # ========================================
    # Accuracy comparison
    # ========================================
    print("\n" + "="*50)
    print("Accuracy comparison:")
    print("="*50)
    
    # Generate test points
    test_points = torch.tensor([
        [0.0, 0.0, 0.0],      # equilibrium
        [0.1, np.pi/6, 0.5],  # small angle
        [0.5, np.pi/2, 1.0],  # 90 degrees
        [-0.3, -np.pi/4, -0.5],
        [0.2, np.pi, 0.0],    # 180 degrees (edge case)
    ], dtype=dtype)
    
    print(f"{'Input':<35} {'ReLU Approx':<25} {'True Value':<25} {'Error':<15}")
    print("-" * 100)
    
    for xu in test_points:
        e_y, e_theta, omega = xu.tolist()
        
        # ReLU approximation
        x_next_approx = dynamics_direct(xu)
        
        # True dynamics
        e_y_true = e_y + dt * v * np.sin(e_theta)
        e_theta_true = e_theta + dt * omega
        
        error_y = abs(x_next_approx[0].item() - e_y_true)
        error_theta = abs(x_next_approx[1].item() - e_theta_true)
        
        print(f"[{e_y:.2f}, {e_theta:.3f}, {omega:.2f}]" + " "*15 + 
              f"[{x_next_approx[0].item():.4f}, {x_next_approx[1].item():.4f}]" + " "*5 +
              f"[{e_y_true:.4f}, {e_theta_true:.4f}]" + " "*5 +
              f"[{error_y:.2e}, {error_theta:.2e}]")


if __name__ == "__main__":
    example_integration()
