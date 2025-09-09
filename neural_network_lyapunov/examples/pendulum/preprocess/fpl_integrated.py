"""
Enhanced FPL Integration for Lyapunov Neural Network with MILP Violation Handling
==================================================================================

This module addresses the local optima issue by incorporating MILP violation costs
into the FPL framework while maintaining logical semantics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class MILPViolationFulfillment:
    """
    Converts MILP violation costs into fulfillment values that FPL can reason about.

    The key insight: MILP violations should be visible to FPL as unfulfilled objectives,
    not just as numerical penalties. This allows FPL's logical operators to properly
    balance constraint satisfaction with performance objectives.
    """

    def __init__(self, alpha_pos: float = 10.0, beta_deriv: float = 10.0):
        """
        Args:
            alpha_pos: Scaling factor for positivity violations (higher = stricter)
            beta_deriv: Scaling factor for derivative violations (higher = stricter)
        """
        self.alpha_pos = alpha_pos
        self.beta_deriv = beta_deriv

    def compute_fulfillment(
        self, positivity_mip_obj: float, derivative_mip_obj: float
    ) -> Tuple[float, float]:
        """
        Transform MILP objective values into fulfillment values [0,1].

        Key principle: Violations (positive objectives) map to low fulfillment,
        satisfied constraints (negative objectives) map to high fulfillment.
        """
        # Sigmoid-like transformation ensures smooth gradients
        f_pos = 1.0 / (1.0 + self.alpha_pos * max(positivity_mip_obj, 0))
        f_deriv = 1.0 / (1.0 + self.beta_deriv * max(derivative_mip_obj, 0))

        return f_pos, f_deriv


class AdaptiveFPLScheduler:
    """
    Implements curriculum learning through FPL's priority offset mechanism.

    Early training: Prioritize Lyapunov conditions (stability)
    Later training: Balance with performance objectives
    """

    def __init__(
        self, warmup_iterations: int = 2000, transition_iterations: int = 1000
    ):
        self.warmup_iterations = warmup_iterations
        self.transition_iterations = transition_iterations

    def get_priority_offsets(self, iteration: int) -> Dict[str, float]:
        """
        Returns priority offsets for different objectives based on training progress.

        The offset [φ]_δ in FPL shifts the baseline fulfillment, creating
        lexicographic-like ordering without hard constraints.
        """
        if iteration < self.warmup_iterations:
            # Strong priority on Lyapunov conditions
            return {"lyapunov": 0.8, "performance": 0.2, "efficiency": 0.1}
        elif iteration < self.warmup_iterations + self.transition_iterations:
            # Smooth transition period
            progress = (iteration - self.warmup_iterations) / self.transition_iterations
            lyap_priority = 0.8 - 0.5 * progress  # 0.8 -> 0.3
            perf_priority = 0.2 + 0.5 * progress  # 0.2 -> 0.7
            return {
                "lyapunov": lyap_priority,
                "performance": perf_priority,
                "efficiency": 0.2,
            }
        else:
            # Performance-focused with constraint maintenance
            return {"lyapunov": 0.3, "performance": 0.7, "efficiency": 0.3}


class EnhancedFPLMonotonicLyapunovTrainer:
    """
    Enhanced trainer that integrates MILP violation handling with FPL composition.
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
        # New parameters for enhanced training
        alpha_pos: float = 10.0,
        beta_deriv: float = 10.0,
        warmup_iterations: int = 2000,
    ):

        # Original parameters
        self.lyapunov_system = lyapunov_system
        self.closed_loop_system = closed_loop_system
        self.V_lambda = V_lambda
        self.x_equilibrium = x_equilibrium
        self.R_options = R_options
        self.x_lo = x_lo
        self.x_up = x_up

        # Enhanced components
        self.violation_fulfillment = MILPViolationFulfillment(alpha_pos, beta_deriv)
        self.scheduler = AdaptiveFPLScheduler(warmup_iterations)
        self.iteration = 0

        # Gradient normalization parameters
        self.gradient_ema_alpha = 0.99
        self.gradient_norm_history = {"fpl": 1.0, "milp": 1.0}

    def compute_composed_fulfillment(
        self,
        f_pos: float,
        f_deriv: float,
        f_performance: torch.Tensor,
        f_efficiency: torch.Tensor,
        p: float = -2.0,
    ) -> torch.Tensor:
        """
        Compose all fulfillment values using FPL operators with adaptive priorities.

        The composition structure:
        (f_pos ∧_p f_deriv) ∧_p ([f_performance]_δ1 ∧_p [f_efficiency]_δ2)

        This ensures Lyapunov conditions are satisfied before optimizing performance.
        """
        priorities = self.scheduler.get_priority_offsets(self.iteration)

        # Apply priority offsets using FPL's [φ]_δ operator
        def apply_offset(f_val, delta):
            """Implements [φ]_δ from FPL semantics"""
            return (f_val + max(delta, 0)) / (1 + delta)

        # Compose Lyapunov conditions (must both be satisfied)
        f_lyapunov = self.power_mean(
            torch.tensor([f_pos, f_deriv]), p=p  # Negative p enforces AND-like behavior
        )

        # Apply priorities
        f_lyapunov_prioritized = apply_offset(f_lyapunov, priorities["lyapunov"])
        f_performance_prioritized = apply_offset(
            f_performance, priorities["performance"]
        )
        f_efficiency_prioritized = apply_offset(f_efficiency, priorities["efficiency"])

        # Final composition
        return self.power_mean(
            torch.stack(
                [
                    f_lyapunov_prioritized,
                    f_performance_prioritized,
                    f_efficiency_prioritized,
                ]
            ),
            p=p,
        )

    def power_mean(self, values: torch.Tensor, p: float) -> torch.Tensor:
        """
        Generalized mean with numerical stability.
        p < 0: AND-like (pessimistic, all must be satisfied)
        p > 0: OR-like (optimistic, any can be satisfied)
        """
        if abs(p) < 1e-6:  # Geometric mean for p ≈ 0
            return torch.exp(torch.mean(torch.log(values + 1e-10)))
        else:
            return torch.pow(torch.mean(torch.pow(values + 1e-10, p)), 1 / p)

    def normalize_gradients(self, loss_fpl: torch.Tensor, loss_milp: torch.Tensor):
        """
        Adaptive gradient normalization to balance FPL and MILP optimization.

        Problem: MILP gradients can dominate, causing FPL objectives to be ignored.
        Solution: Track gradient magnitudes and normalize to maintain balance.
        """
        # Compute gradient norms
        grad_fpl = torch.autograd.grad(
            loss_fpl, self.get_trainable_params(), retain_graph=True, create_graph=False
        )
        grad_milp = torch.autograd.grad(
            loss_milp,
            self.get_trainable_params(),
            retain_graph=True,
            create_graph=False,
        )

        norm_fpl = sum(g.norm().item() for g in grad_fpl if g is not None)
        norm_milp = sum(g.norm().item() for g in grad_milp if g is not None)

        # Update EMA of gradient norms
        self.gradient_norm_history["fpl"] = (
            self.gradient_ema_alpha * self.gradient_norm_history["fpl"]
            + (1 - self.gradient_ema_alpha) * norm_fpl
        )
        self.gradient_norm_history["milp"] = (
            self.gradient_ema_alpha * self.gradient_norm_history["milp"]
            + (1 - self.gradient_ema_alpha) * norm_milp
        )

        # Compute scaling factors to balance gradients
        scale_fpl = 1.0
        scale_milp = self.gradient_norm_history["fpl"] / (
            self.gradient_norm_history["milp"] + 1e-8
        )

        return loss_fpl * scale_fpl + loss_milp * scale_milp

    def get_trainable_params(self):
        """Get all trainable parameters."""
        params = []
        params += list(self.lyapunov_system.lyapunov_relu.parameters())
        params += list(self.R_options.variables())
        if hasattr(self.closed_loop_system, "controller_variables"):
            params += self.closed_loop_system.controller_variables()
        return params

    def train_step(
        self,
        positivity_state_samples: torch.Tensor,
        derivative_state_samples: torch.Tensor,
        positivity_mip_obj: float,
        derivative_mip_obj: float,
    ) -> Dict:
        """
        Single training step with enhanced FPL-MILP integration.
        """
        # Convert MILP violations to fulfillment values
        f_pos, f_deriv = self.violation_fulfillment.compute_fulfillment(
            positivity_mip_obj, derivative_mip_obj
        )

        # Compute performance and efficiency fulfillments (example)
        # These would come from your actual objectives
        f_performance = self.compute_performance_fulfillment(derivative_state_samples)
        f_efficiency = self.compute_efficiency_fulfillment(derivative_state_samples)

        # Compose all fulfillments with adaptive priorities
        overall_fulfillment = self.compute_composed_fulfillment(
            f_pos, f_deriv, f_performance, f_efficiency
        )

        # Convert fulfillment to loss (we minimize 1 - fulfillment)
        loss_fpl = 1.0 - overall_fulfillment

        # Original MILP loss (for gradient balancing)
        loss_milp = max(positivity_mip_obj, 0) + max(derivative_mip_obj, 0)
        loss_milp = torch.tensor(loss_milp, requires_grad=True)

        # Balance gradients between FPL and MILP objectives
        total_loss = self.normalize_gradients(loss_fpl, loss_milp)

        self.iteration += 1

        return {
            "total_loss": total_loss.item(),
            "f_pos": f_pos,
            "f_deriv": f_deriv,
            "overall_fulfillment": overall_fulfillment.item(),
            "priorities": self.scheduler.get_priority_offsets(self.iteration),
        }

    def compute_performance_fulfillment(self, states: torch.Tensor) -> torch.Tensor:
        """Placeholder for actual performance metric."""
        # Example: Distance from goal
        goal = torch.tensor([np.pi, 0], dtype=torch.float64)
        distances = torch.norm(states - goal, dim=1)
        return torch.exp(-0.1 * distances.mean())

    def compute_efficiency_fulfillment(self, states: torch.Tensor) -> torch.Tensor:
        """Placeholder for actual efficiency metric."""
        # Example: Control effort minimization
        controls = self.closed_loop_system.compute_u_pre(states)
        effort = torch.norm(controls, dim=1).mean()
        return torch.exp(-0.01 * effort)


def train_with_enhanced_fpl(trainer, state_samples, args):
    """
    Modified training loop with MILP violation handling.
    """
    import torch.optim as optim

    # Setup optimizer with all parameters
    params = trainer.get_trainable_params()
    optimizer = optim.Adam(params, lr=args.learning_rate)

    # Increase pool solutions for better exploration
    trainer.lyapunov_system.lyapunov_positivity_mip_pool_solutions = 5
    trainer.lyapunov_system.lyapunov_derivative_mip_pool_solutions = 5

    dataset = torch.utils.data.TensorDataset(state_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    for epoch in range(args.pretrain_num_epochs):
        epoch_metrics = []

        for batch_idx, (batch_states,) in enumerate(dataloader):
            optimizer.zero_grad()

            # Solve MILPs for current batch
            positivity_mip_obj = solve_positivity_mip(trainer, batch_states)
            derivative_mip_obj = solve_derivative_mip(trainer, batch_states)

            # Enhanced training step
            metrics = trainer.train_step(
                batch_states, batch_states, positivity_mip_obj, derivative_mip_obj
            )

            # Backward pass
            total_loss = torch.tensor(metrics["total_loss"], requires_grad=True)
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()
            epoch_metrics.append(metrics)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(
                    f"  Fulfillments - Pos: {metrics['f_pos']:.4f}, "
                    f"Deriv: {metrics['f_deriv']:.4f}, "
                    f"Overall: {metrics['overall_fulfillment']:.4f}"
                )
                print(f"  Priorities: {metrics['priorities']}")

        # Check convergence
        avg_fulfillment = np.mean([m["overall_fulfillment"] for m in epoch_metrics])
        if avg_fulfillment > 0.95:
            print(f"Converged at epoch {epoch} with fulfillment {avg_fulfillment:.4f}")
            break

    return trainer


def solve_positivity_mip(trainer, states):
    """Placeholder for actual MILP solving."""
    # This would call your actual MILP solver
    # Return the objective value
    return np.random.uniform(-0.1, 0.5)  # Simulated value


def solve_derivative_mip(trainer, states):
    """Placeholder for actual MILP solving."""
    # This would call your actual MILP solver
    # Return the objective value
    return np.random.uniform(-0.2, 0.3)  # Simulated value
