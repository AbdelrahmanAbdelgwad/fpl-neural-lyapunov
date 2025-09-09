import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict
from dataclasses import dataclass


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
    Constructs piecewise linear transformations for constraint shaping.

    This allows creating sophisticated reward landscapes that guide
    optimization through different regimes.
    """
    result = torch.zeros_like(value)

    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]

        # Linear interpolation in this segment
        segment_value = (value - x0) / (x1 - x0) * (y1 - y0) + y0

        if clipped:
            segment_value = torch.clamp(segment_value, min(y0, y1), max(y0, y1))

        # Apply this segment where value falls within [x0, x1)
        if i == len(breakpoints) - 2:
            # Last segment
            mask = value >= x0
        else:
            mask = (value >= x0) & (value < x1)

        result = torch.where(mask, segment_value, result)

    return result


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
            # wrap angle to [-pi, pi] in-place on index 2
            next_states[:, 2:3] = (
                (next_states[:, 2:3] + torch.pi) % (2 * torch.pi)
            ) - torch.pi
            trajectory[:, t + 1] = next_states
            current_states = next_states

        # V(x_{horizon})
        V_values[:, -1] = self.compute_lyapunov_value(current_states).squeeze()
        return trajectory, V_values, u_values

    def _effort_fulfillment(self, u_values):
        # u_values: [B, H, u_dim] -> scale to [0,1], average over time & controls
        u_max = 10.0  # set to your clamp in the controller
        scaled = torch.clamp(torch.abs(u_values) / u_max, 0, 1)
        return 1.0 - scaled.mean(dim=(1, 2))  # -> [B]

    # fpl.py  (helper inside the class)
    def _in_bounds_fulfillment(self, traj):
        """
        traj: [B, H+1, state_dim]
        returns [B] in [0,1] high when all states stay inside [x_lo, x_up]
        """
        if (self.x_lo is None) or (self.x_up is None):
            return torch.ones(traj.shape[0], dtype=traj.dtype, device=traj.device)

        # per-step, per-dim violation (>=0 outside, 0 inside)
        over = torch.clamp(traj - self.x_up, min=0.0)
        under = torch.clamp(self.x_lo - traj, min=0.0)
        viol = over + under  # [B, H+1, D]

        # aggregate across dims by infinity-norm (max violation per step)
        step_viol = viol.abs().amax(dim=2)  # [B, H+1]

        # turn into fulfillment in [0,1]: 1/(1+α*viol), then AND over time (p=-2)
        alpha = 10.0
        per_step_F = 1.0 / (1.0 + alpha * step_viol)  # [B, H+1]
        # geometric/AND-like aggregate across time
        F_bounds = (per_step_F.clamp(min=1e-6).prod(dim=1)) ** (
            1.0 / (per_step_F.shape[1] + 1e-9)
        )
        return F_bounds

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

        F_bounds = self._in_bounds_fulfillment(trajectories)  # [B]

        V_initial = V_values[:, 0]
        V_final = V_values[:, -1]

        # # DEBUG: Check what's happening with trajectories
        # print(f"\n=== Debug Info ===")
        # print(f"Horizon: {horizon}")
        # print(f"Batch size: {batch_states.shape[0]}")

        # # Check a single trajectory
        # idx = 0  # First trajectory
        # print(f"\nTrajectory {idx}:")
        # print(f"  Initial state: {batch_states[idx].tolist()}")
        # # print intermediate states
        # for t in range(horizon + 1):
        #     print(f"  State at t={t}: {trajectories[idx, t].tolist()}")
        # print(f"  Control inputs: {u_values[idx].squeeze().tolist()}")
        # print(f"  Final state: {trajectories[idx, -1].tolist()}")
        # print(f"  Initial V: {V_initial[idx].item():.6f}")
        # print(f"  Final V: {V_final[idx].item():.6f}")
        # print(f"  V_decrease: {(V_initial[idx] - V_final[idx]).item():.6f}")

        # # Check if states are actually moving toward equilibrium
        # initial_dist = torch.norm(batch_states[idx] - self.x_equilibrium).item()
        # final_dist = torch.norm(trajectories[idx, -1] - self.x_equilibrium).item()
        # print(f"  Initial distance from eq: {initial_dist:.6f}")
        # print(f"  Final distance from eq: {final_dist:.6f}")
        # print(f"  Distance decrease: {(initial_dist - final_dist):.6f}")

        # # Print first few V values along trajectory
        # print(f"  V trajectory: {V_values[idx, :5].tolist()}")
        # print("==================\n")

        # distances
        initial_distances = torch.norm(batch_states - self.x_equilibrium, p=2, dim=1)
        final_states = trajectories[:, -1]
        final_distances = torch.norm(final_states - self.x_equilibrium, p=2, dim=1)

        # # 1) Lyapunov decrease
        V_decrease = V_initial - V_final
        # required_decrease = torch.minimum(
        #     V_initial,
        #     torch.tensor(
        #         horizon / 50.0, dtype=V_initial.dtype, device=V_initial.device
        #     ),
        # )
        # decrease_satisfaction = build_piecewise(
        #     [
        #         (-1.0, 0.0),
        #         (-0.05, 0.001),
        #         (0.0, 0.01),
        #         (required_decrease, 0.9),
        #         (1.0, 1.0),
        #     ],
        #     V_decrease,
        #     clipped=True,
        # )

        # 1) Lyapunov exponential decrease (not finite time decay with build_piecewise)
        # V_dot <= -K * V
        # Discrete condition: V_final - V_initial <= -K * V_initial  ⇔  V_final <= (1 - K) * V_initial
        # # When computing decrease satisfaction, normalize V values first
        # V_initial_norm = V_initial / (self.V_scale if hasattr(self, "V_scale") else 1.0)
        # V_final_norm = V_final / (self.V_scale if hasattr(self, "V_scale") else 1.0)

        # Use normalized values for exponential decay check
        K = 0.2
        eps = 1e-4  # Increased from 1e-8
        # mask = V_initial_norm > eps
        mask = V_initial > eps  # Use original values for masking

        # V_i = V_initial_norm[mask]
        # V_f = V_final_norm[mask]
        V_i = V_initial[mask]
        V_f = V_final[mask]

        # Target upper bound for the next-step Lyapunov value
        target_final = (1.0 - K) * V_i

        # Hinge penalty for violations of V_f <= target_final
        # (no penalty when the constraint is satisfied)
        violation = torch.relu(V_f - target_final)

        # Smooth, stronger penalty (squared); use .mean() for batch aggregation
        lyap_decay_loss = (
            (violation**2).mean()
            if violation.numel() > 0
            else V_initial.new_tensor(0.0)
        )
        lyap_decay_loss = lyap_decay_loss

        # saturate the penalty to be betwenen 0 and 1 without clipping gradients
        lyap_decay_loss = torch.tanh(lyap_decay_loss)
        decrease_satisfaction = 1.0 - lyap_decay_loss

        # (Optional) metric: fraction of states satisfying the constraint
        frac_satisfied = (
            (V_f <= target_final).float().mean()
            if V_i.numel() > 0
            else V_initial.new_tensor(1.0)
        )

        # 2) Lyapunov Positivity Constraint
        # V should be positive away from equilibrium
        distance_threshold = 0.01
        away_from_eq = initial_distances > distance_threshold
        V_positive = torch.where(
            away_from_eq,
            torch.sigmoid(V_initial * 10),  # Sigmoid shaping for smooth gradient
            torch.ones_like(V_initial),
        )

        # 3) Zero at Equilibrium Constraint
        V_at_eq = self.compute_lyapunov_value(self.x_equilibrium.unsqueeze(0)).squeeze()
        zero_constraint = 1.0 - torch.sqrt(V_at_eq + 1e-9)

        # 4) Progress toward eq
        progress = torch.exp(-final_distances)

        # 5) NEW: control-effort fulfillment in [0,1]
        effort_fulfillment = self._effort_fulfillment(u_values)  # [batch]
        effort_fulfillment = effort_fulfillment**2.0

        # Build FPL tree
        fpl_structure = FPLConstraint(
            p_value=-0.0,
            constraints={
                "stability": FPLConstraint(
                    p_value=-2.0,
                    constraints={
                        "decrease": p_mean(decrease_satisfaction, -6.0),
                        "v_dot": p_mean(torch.sigmoid(V_decrease * 10), -10.0),
                    },
                ),
                # "performance": FPLConstraint(
                #     p_value=-2.0,
                #     constraints={
                #         "effort": p_mean(effort_fulfillment, -2.0),
                #     },
                # ),
                "in_bounds": p_mean(F_bounds, -2.0),
            },
        )
        # print(f"Effort Fulfillment: {effort_fulfillment}")

        fulfillment = fpl_structure.evaluate()
        loss = 1.0 - fulfillment
        return loss, fpl_structure


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

            # Compute FPL loss with trajectory rollout
            loss, fpl_structure = trainer.compute_fpl_loss(
                batch_states, min_horizon=3, max_horizon=20
            )

            # loss.backward(retain_graph=True)
            loss.backward()
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
        if (1 - avg_loss) >= 0.95:
            print("Converged!")
            break

    return trainer
