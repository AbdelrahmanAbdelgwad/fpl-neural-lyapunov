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
    ):
        self.lyapunov_system = lyapunov_system
        self.closed_loop_system = closed_loop_system
        self.V_lambda = V_lambda
        self.x_equilibrium = x_equilibrium
        self.R_options = R_options

    def rollout_trajectory(self, initial_states: torch.Tensor, horizon: int) -> tuple:
        """
        Simulate forward and return (trajectory, V_values, u_values).
        """
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
            batch_size, horizon, 1, device=device, dtype=initial_states.dtype
        )  # assuming 1D u

        trajectory[:, 0] = initial_states
        current_states = initial_states

        for t in range(horizon):
            # V(x_t)
            V_values[:, t] = self.compute_lyapunov_value(current_states).squeeze()

            # u_t (use the same controller used by step_forward)
            u_t = self.closed_loop_system.compute_u(current_states)
            u_values[:, t, 0] = u_t.squeeze(-1)

            # x_{t+1}
            next_states = self.closed_loop_system.step_forward(current_states)
            trajectory[:, t + 1] = next_states
            current_states = next_states

        V_values[:, -1] = self.compute_lyapunov_value(current_states).squeeze()
        return trajectory, V_values, u_values

    def _effort_fulfillment(self, u_values: torch.Tensor) -> torch.Tensor:
        """
        Map control effort to [0,1] fulfillment."""
        # Normalize u_values to [0, 1] range
        u_min = torch.min(u_values)
        u_max = torch.max(u_values)

        if u_max - u_min < 1e-6:
            # Avoid division by zero if all u_values are the same
            return torch.ones(u_values.shape[0], device=u_values.device)
        else:
            normalized_u = (u_values - u_min) / (u_max - u_min)
            return torch.clamp(normalized_u, 0.0, 1.0).squeeze(-1)

    def compute_lyapunov_value(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute V(x) using the monotonic Lyapunov network.
        """
        relu_output = self.lyapunov_system.lyapunov_relu(states)
        relu_equilibrium = self.lyapunov_system.lyapunov_relu(self.x_equilibrium)

        R = self.R_options.R() if hasattr(self.R_options, "R") else self.R_options

        V = (
            relu_output
            - relu_equilibrium
            + self.V_lambda
            * torch.norm(R @ (states - self.x_equilibrium).T, p=1, dim=0).reshape(-1, 1)
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

        # 1) Lyapunov decrease
        V_decrease = V_initial - V_final
        required_decrease = torch.minimum(
            V_initial,
            torch.tensor(
                horizon / 50.0, dtype=V_initial.dtype, device=V_initial.device
            ),
        )
        decrease_satisfaction = build_piecewise(
            [
                (-1.0, 0.0),
                (-0.05, 0.001),
                (0.0, 0.01),
                (required_decrease, 0.9),
                (1.0, 1.0),
            ],
            V_decrease,
            clipped=True,
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
        V_at_eq = self.compute_lyapunov_value(self.x_equilibrium).squeeze()
        zero_constraint = 1.0 - torch.sqrt(V_at_eq + 1e-9)

        # 4) Progress toward eq
        progress = torch.exp(-final_distances)

        # 5) NEW: control-effort fulfillment in [0,1]
        effort_fulfillment = self._effort_fulfillment(u_values)  # [batch]

        # Build FPL tree
        fpl_structure = FPLConstraint(
            p_value=0.0,  # geometric mean at the top
            constraints={
                "stability": FPLConstraint(
                    p_value=-1.0,  # AND-like: emphasize the weakest term
                    constraints={
                        "decrease": p_mean(decrease_satisfaction, -1.0),
                        "positive": p_mean(
                            V_positive, 0.0
                        ),  # Always satisfied due to monotonic architecture
                        "zero_at_eq": zero_constraint,  # Always satisfied due to monotonic architecture
                    },
                ),
                "performance": FPLConstraint(
                    p_value=0.0,  # geometric mean
                    constraints={
                        "progress": p_mean(progress, -2.0),
                        "v_dot": p_mean(torch.sigmoid(V_decrease * 10), 0.0),
                        "effort": p_mean(effort_fulfillment, -2.0),
                    },
                ),
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
    optimizer = torch.optim.Adam(
        list(trainer.lyapunov_system.lyapunov_relu.parameters())
        + trainer.R_options.variables(),
        lr=args.learning_rate,
    )

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

            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Fulfillment: {1 - loss.item():.4f}")
                print(f"  FPL Structure: {fpl_structure}")

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} Average Fulfillment: {1 - avg_loss:.4f}")

        # Early stopping if converged
        if avg_loss < 1e-2:
            print("Converged!")
            break

    return trainer
