import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict
from dataclasses import dataclass
import numpy as np



def p_mean(
    values: torch.Tensor,
    p: float,
    slack: float = 1e-15,
    default_val: float = 0.0,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Generalized mean (power mean) - the mathematical heart of FPL.

    This function creates non-linear compositions of objectives:
    - p < 0: Emphasizes poorly-performing objectives (AND-like)
    - p = 0: Geometric mean (multiplicative balance)
    - p > 0: Emphasizes well-performing objectives (OR-like)

    The slack parameter prevents numerical instability when values approach 0.
    """
    # Ensure numerical stability
    p = max(p, 1e-5) if abs(p) < 1e-5 else p

    # Handle empty tensor edge case
    if values.numel() == 0:
        return torch.tensor(default_val, dtype=values.dtype, device=values.device)

    # Add slack to prevent division by zero with negative p
    values_safe = values + slack

    # Compute the p-mean
    if abs(p) < 1e-4:  # Approximate geometric mean for p ≈ 0
        result = torch.exp(torch.mean(torch.log(values_safe), dim=dim))
    else:
        result = torch.mean(values_safe**p, dim=dim) ** (1.0 / p)

    return result - slack


def build_piecewise(
    breakpoints: List[tuple], value: torch.Tensor, clipped: bool = False
) -> torch.Tensor:
    """
    Constructs piecewise linear transformations matching TensorFlow implementation.
    """
    # Start with the last segment
    x0, y0 = breakpoints[-2]
    x1, y1 = breakpoints[-1]
    
    # Linear interpolation for last segment (applies to all initially)
    result = (value - x0) / (x1 - x0) * (y1 - y0) + y0
    
    if clipped:
        result = torch.clamp(result, min(y0, y1), max(y0, y1))
    
    # Work backwards through remaining segments
    for i in range(len(breakpoints) - 2, 0, -1):
        x0, y0 = breakpoints[i - 1]
        x1, y1 = breakpoints[i]
        
        # Linear interpolation in this segment
        segment_value = (value - x0) / (x1 - x0) * (y1 - y0) + y0
        
        if clipped:
            segment_value = torch.clamp(segment_value, min(y0, y1), max(y0, y1))
        
        # Overwrite where value < x1 (upper bound of segment)
        result = torch.where(value < x1, segment_value, result)
    
    return result

def angular_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Angular similarity returns 1 if v1 == v2 (same angle), 0 if opposite.
    Only considers the position (first 2 components), not velocity.
    
    Args:
        v1: [..., state_dim] where first 2 dims are cos(theta), sin(theta)
        v2: [..., state_dim] where first 2 dims are cos(theta), sin(theta)
    """
    # Extract angles from trajectory representation state = [theta, theta_dot]
    v1_angle = v1[..., 0]  # theta
    v2_angle = v2[..., 0]  # theta_equilibrium
    
    # Compute similarity (1 when same, 0 when opposite)
    return torch.abs(torch.abs(v1_angle - v2_angle) - np.pi) / np.pi

@dataclass
class FPLConstraint:
    """
    Recursive constraint structure for hierarchical objective composition.

    This allows building complex behavioral specifications through
    nested compositions of simpler objectives.
    """

    p_value: float  # The p-value for p_mean
    constraints: Dict[str, Union["FPLConstraint", torch.Tensor]]

    def evaluate(self) -> torch.Tensor:
        """Recursively evaluate the constraint hierarchy."""
        values = []
        for name, constraint in self.constraints.items():
            if isinstance(constraint, FPLConstraint):
                values.append(constraint.evaluate())
            else:
                values.append(constraint)

        if not values:
            return torch.tensor(1.0)

        stacked = torch.stack(values)
        return p_mean(stacked, self.p_value, default_val=1.0)

    def __str__(self) -> str:
        """Human-readable representation of the constraint hierarchy."""
        constraint_strs = []
        for name, c in self.constraints.items():
            if isinstance(c, torch.Tensor):
                constraint_strs.append(f"{name}:{c.item():.3f}")
            else:
                constraint_strs.append(f"{name}:{c}")
        return f"p={self.p_value:.1f}<{', '.join(constraint_strs)}>"


class FPLMonotonicLyapunovTrainer:
    """
    Enhanced trainer combining FPL objective composition with
    monotonic Lyapunov architecture for path following.
    """

    def __init__(
        self,
        lyapunov_system,
        closed_loop_system,
        V_lambda: float,
        x_equilibrium: torch.Tensor,
        R_options,
        x_lo,
        x_up,
    ):
        self.lyapunov_system = lyapunov_system
        self.closed_loop_system = closed_loop_system
        self.V_lambda = V_lambda
        self.x_equilibrium = x_equilibrium
        self.R_options = R_options
        self.x_lo = x_lo
        self.x_up = x_up

    def rollout_trajectory(self, initial_states: torch.Tensor, horizon: int) -> tuple:
        batch_size, state_dim = initial_states.shape
        device = initial_states.device

        trajectory = torch.zeros(
            batch_size,
            horizon + 1,
            state_dim,
            device=device,
            dtype=initial_states.dtype,
        )
        V_values = torch.zeros(
            batch_size, horizon + 1, device=device, dtype=initial_states.dtype
        )
        u_values = torch.zeros(
            batch_size, horizon, 2, device=device, dtype=initial_states.dtype
        )

        trajectory[:, 0] = initial_states
        current_states = initial_states

        for t in range(horizon):
            # V(x_t)
            V_values[:, t] = self.compute_lyapunov_value(current_states).squeeze()

            # u_t (controller is differentiable)
            u_t = self.closed_loop_system.compute_u_pre(current_states)
            u_values[:, t] = u_t

            # x_{t+1} = f(x_t, u_t)   (NO .detach(); NO in-place)
            ns = self.closed_loop_system.step_forward(current_states)  # [B, 3]
            next_states = ns.clone()
            # # wrap angle to [-pi, pi] in-place on index 2
            # next_states[:, 2:3] = (
            #     (next_states[:, 2:3] + torch.pi) % (2 * torch.pi)
            # ) - torch.pi

            trajectory[:, t + 1] = next_states
            current_states = next_states

        # V(x_{horizon})
        V_values[:, -1] = self.compute_lyapunov_value(current_states).squeeze()
        return trajectory, V_values, u_values


    def compute_lyapunov_value(self, states: torch.Tensor) -> torch.Tensor:
        # ensure 2D: (B,2)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        xeq = self.x_equilibrium
        if xeq.dim() == 1:
            xeq = xeq.unsqueeze(0)

        relu_output = self.lyapunov_system.lyapunov_relu(states)
        relu_equilibrium = self.lyapunov_system.lyapunov_relu(xeq)

        R = self.R_options.R() if hasattr(self.R_options, "R") else self.R_options
        delta = states - xeq  # (B,2)
        # use .mT (batch-friendly) and avoid .reshape(-1,1) surprises
        V = (
            relu_output
            - relu_equilibrium
            + self.V_lambda * torch.norm(R @ delta.mT, p=1, dim=0).unsqueeze(1)
        )
        return V

    def compute_fpl_loss(
        self, batch_states: torch.Tensor, min_horizon: int = 3, max_horizon: int = 20
    ) -> tuple:
        horizon = torch.randint(min_horizon, max_horizon + 1, (1,)).item()

        trajectories, V_values, u_values = self.rollout_trajectory(
            batch_states, horizon
        )

        V_initial = V_values[:, 0]
        V_final = V_values[:, -1]

        # distances
        initial_distances = torch.norm(batch_states - self.x_equilibrium, p=2, dim=1)
        final_states = trajectories[:, -1]
        final_distances = torch.norm(final_states - self.x_equilibrium, p=2, dim=1)

        # Lyapunov decrease
        diff = V_initial - V_final
        
        decrease_by = (
            1.0 / 100.0
        )  # should arrive to the target within 100 steps
        line = torch.minimum(
            torch.tensor(decrease_by * horizon, dtype=V_initial.dtype, device=V_initial.device),
            V_initial
        )
        proof_of_performance = p_mean(
            build_piecewise(
                [(-1.0, 0.0), (-0.1, 0.001), (0.0, 0.01), (line, 0.9), (1.0, 1.0)],
                diff,
                clipped=True,
            ),
            -2.0,
        )

        # Compute angular similarities for all states in trajectory
        # trajectories shape: [batch_size, horizon+1, state_dim]
        # Broadcast equilibrium to match trajectory shape
        equilibrium_broadcast = self.x_equilibrium.unsqueeze(0).unsqueeze(0).expand_as(trajectories)
        
        # Compute angular similarity for all states
        as_all = angular_similarity(trajectories, equilibrium_broadcast)
        # as_all shape: [batch_size, horizon+1]
        
        # Flatten to compute p_mean across all timesteps and batch
        as_all_flat = as_all.reshape(-1)

        close_angle_fpl = build_piecewise(
                    [(0.0, 0.0), (0.6, 0.01), (0.7, 0.9), (1.0, 1.0)],
                    p_mean(as_all_flat, -2.0),
                )
        
        # control effort (should be small)
        control_effort = torch.norm(u_values, p=2, dim=2)  # (B, horizon)
        control_effort_fpl_batch = p_mean(
            build_piecewise(
                [(0.0, 1.0), (5.0, 0.9), (8.0, 0.5), (15.0, 0.1), (20.0, 0.0), (40.0, 0.0)],
                control_effort,
                clipped=True,
            ),
            -2.0,
            dim=1,
        )
        control_effort_fpl = p_mean(control_effort_fpl_batch, 1.0)

        # Build FPL tree with close_angles constraint
        fpl_structure = FPLConstraint(
            0.0,
            {
                "performance": FPLConstraint(
                    0.0,
                    {
                        "close_angles": close_angle_fpl,
                        "control_effort": control_effort_fpl,
                    },
                ),

                "lyapunov": FPLConstraint(
                    0.0,
                    {
                        "pop": proof_of_performance,
                    },
                ),
            },
        )

        # Use this for the actual loss (keeps gradients)
        fulfillment = fpl_structure.evaluate()
        loss = 1.0 - fulfillment

        # Build a DETACHED copy purely for logging/printing
        def _detached_tree(c):
            if isinstance(c, torch.Tensor):
                return c.detach()
            if isinstance(c, FPLConstraint):
                return FPLConstraint(c.p_value, {k: _detached_tree(v) for k, v in c.constraints.items()})
            return c

        fpl_structure_log = _detached_tree(fpl_structure)

        return loss, fpl_structure_log



def train_with_fpl(trainer, state_samples, args):
    """
    Enhanced training loop using FPL composition.
    """

    # AFTER
    params = []
    params += list(trainer.lyapunov_system.lyapunov_relu.parameters())
    params += list(trainer.R_options.variables())  # <- wrap in list
    params += trainer.closed_loop_system.controller_variables()  # <- already a list

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Convert to batched dataset
    dataset = torch.utils.data.TensorDataset(state_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )


    for epoch in range(args.pretrain_num_epochs):
        epoch_losses = []

        for batch_idx, (batch_states,) in enumerate(dataloader):
            optimizer.zero_grad()

            # generate random samples from a gussian distribution around the equilibrium
            batch_states = torch.randn(batch_states.shape[0], batch_states.shape[1], dtype=torch.float64)
            batch_states = (
                batch_states
                * (1 * (trainer.x_up - trainer.x_lo)).reshape((1, -1))
                + trainer.closed_loop_system.x_equilibrium.reshape((1, -1))
            )
            batch_states = torch.clamp(batch_states, trainer.x_lo, trainer.x_up)

            # Compute FPL loss with trajectory rollout
            loss, fpl_structure = trainer.compute_fpl_loss(
                batch_states,
                min_horizon=1,
                max_horizon=15,
            )

            loss.backward(retain_graph=True)
            # loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                g = sum(
                    (
                        p.grad.norm().item()
                        for p in trainer.closed_loop_system.controller_variables()
                        if p.grad is not None
                    )
                )
                # print(f"[debug] controller grad-norm: {g:.3e}")

            epoch_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Fulfillment: {1 - loss.item():.4f}")
                print(f"  FPL Structure: {fpl_structure}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} Average Fulfillment: {1 - avg_loss:.4f}")

        # Early stopping if converged
        # if (1 - avg_loss) >= (min(0.94 + args.bound_level / 100, 0.995)):
        if (1 - avg_loss) >= 0.98:
            print("Converged!")
            break

    return trainer
