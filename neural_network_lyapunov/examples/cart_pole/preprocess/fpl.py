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
        print(f"FPL Trainer: Equilibrium at {self.x_equilibrium}")
        self.R_options = R_options
        self.x_lo = x_lo
        print(f"FPL Trainer: State lower bounds {self.x_lo}")
        self.x_up = x_up
        print(f"FPL Trainer: State upper bounds {self.x_up}")

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
       # respect the true control dimension (cart-pole u_dim = 1)
        u_dim = self.closed_loop_system.forward_system.u_dim
        u_values = torch.zeros(
        batch_size, horizon, u_dim, device=device, dtype=initial_states.dtype
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
            # wrap cart-pole angle θ to [-π, π]; θ is state index 1
            next_states[:, 1:2] = (
                (next_states[:, 1:2] + torch.pi) % (2 * torch.pi)
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

    def _monotonic_fulfillment(self) -> torch.Tensor:
        """
        Returns a scalar in [0,1] that rewards being comfortably above the
        layer constraints:
        - case=1: b_input >= 0.1  (we reward margin = b_input - 0.1)
        - case=2: a_input >= 0    (we reward a_input), and also the
            *effective* slopes a >= epsilon  (we reward margin = min(a) - epsilon)
        """
        relu = self.lyapunov_system.lyapunov_relu
        layer_scores = []

        for layer in relu:
            # Only care about our custom monotonic layers
            if not hasattr(layer, "case"):
                continue

            # CASE 1 (bias/partition layer): reward b_input − 0.1
            if getattr(layer, "case", None) == 1 and hasattr(layer, "b_input"):
                # margin per element
                margin_b = layer.b_input - 0.1
                # map margin -> [0,1] smoothly; higher margin -> closer to 1
                F_b = torch.sigmoid(10.0 * margin_b).mean()
                layer_scores.append(F_b)

            # CASE 2 (slope layer): reward both raw and effective slopes
            if getattr(layer, "case", None) == 2 and hasattr(layer, "a_input"):
                # raw nonnegativity (how far above 0 the unconstrained params sit)
                F_raw = torch.sigmoid(10.0 * layer.a_input).mean()
                # effective slopes 'a' are computed by the layer; reward min(a) − epsilon
                if hasattr(layer, "a") and hasattr(layer, "epsilon"):
                    margin_eff = layer.a.min() - layer.epsilon
                    F_eff = torch.sigmoid(10.0 * margin_eff)
                    F_layer = 0.5 * (F_raw + F_eff)
                else:
                    F_layer = F_raw
                layer_scores.append(F_layer)

        if not layer_scores:
            # If no monotonic layers are present, treat as satisfied.
            return torch.tensor(1.0, dtype=self.x_lo.dtype)

        return torch.stack(layer_scores).mean()


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

    def compute_fpl_loss(self, batch_states: torch.Tensor, min_horizon: int = 3, max_horizon: int = 20, deriv_viol: torch.Tensor = None) -> tuple:
        horizon = torch.randint(min_horizon, max_horizon + 1, (1,)).item()
        trajectories, V_values, u_values = self.rollout_trajectory(batch_states, horizon)
        
        V_initial = V_values[:, 0]
        V_final = V_values[:, -1]
        final_states = trajectories[:, -1]
        
        # For cartpole: state is [x, theta, x_dot, theta_dot]
        # We care about theta converging to 0 (upright position)
        
        # 1. Angular similarity for cartpole (theta should be near 0)
        theta_final = final_states[:, 1]  # theta is at index 1
        theta_equilibrium = self.x_equilibrium[1]  # should be 0
        
        # Map theta difference to similarity (1 when theta=0, 0 when far)
        angular_similarities = torch.exp(-10 * (theta_final - theta_equilibrium)**2)
        
        # All trajectory angular similarities for p_mean
        all_theta = trajectories[:, :, 1]  # [B, H+1]
        as_all = torch.exp(-10 * (all_theta - theta_equilibrium)**2)  # [B, H+1]
        as_all_mean = as_all.mean(dim=1)  # [B]
        
        # Build piecewise for close_angles matching TF version
        close_angles = build_piecewise(
            [(0.0, 0.0), (0.6, 0.01), (0.7, 0.9), (1.0, 1.0)],
            p_mean(as_all_mean, 2.0),  # Use p=2.0 as in TF
            clipped=False
        )

        # 2. X position closeness (x should be near 0)
        x_final = final_states[:, 0]  # x is at index 0
        x_equilibrium = self.x_equilibrium[0]  # should be 0

        # Map x difference to similarity (1 when x=0, 0 when far)
        x_similarities = torch.exp(-1 * (x_final - x_equilibrium)**2)

        all_x = trajectories[:, :, 0]  # [B, H+1]
        xs_all = torch.exp(-1 * (all_x - x_equilibrium)**2)
        xs_all_mean = xs_all.mean(dim=1)  # [B]

        close_x = build_piecewise(
            [(0.0, 0.0), (0.6, 0.01), (0.7, 0.9), (1.0, 1.0)],
            p_mean(xs_all_mean, 2.0),  # Use p=2.0 as in TF
            clipped=False
        )
        
        # 3. Proof of Performance - matching TF's line-based approach
        V_decrease = V_initial - V_final
        decrease_by = 1.0 / 100.0  # arrive at target within 100 steps
        repetitionsf = torch.tensor(horizon, dtype=V_initial.dtype, device=V_initial.device)
        line = torch.minimum(
            decrease_by * repetitionsf,
            V_initial
        )
        
        # Build piecewise needs line to be a scalar or match batch size
        # We'll apply it element-wise
        proof_of_performance = []
        for i in range(V_decrease.shape[0]):
            line_i = line[i] if line.dim() > 0 else line
            pop_i = build_piecewise(
                [(-1.0, 0.0), (-0.1, 0.001), (0.0, 0.01), (line_i.item(), 0.9), (1.0, 1.0)],
                V_decrease[i:i+1],
                clipped=True
            )
            proof_of_performance.append(pop_i)
        proof_of_performance = torch.cat(proof_of_performance)

        # 4. Exponential Lyapunov decrease per step V[t+1] < (1 - K) * V[t]
        K = 0.001  # 0.1% decrease per step
        steps = 1
        V_final_i = V_values[:, steps:]
        V_initial_i = V_values[:, :-steps]
        # Ensure same length by trimming
        if V_final_i.shape[1] != V_initial_i.shape[1]:
            min_len = min(V_final_i.shape[1], V_initial_i.shape[1])
            V_final_i = V_final_i[:, :min_len]
            V_initial_i = V_initial_i[:, :min_len]
        # Compute v_dot for each step
        v_dot = V_final_i - (1.0 - K) * V_initial_i  # [B, H]
        v_dot_mean = v_dot.mean(dim=1)  # [B]

        exp_decrease_fulfillment = build_piecewise(
            [(-1.0, 0.0), (-0.1, 0.001), (0.0, 0.01), (1.0, 0.9), (10.0, 1.0)],
            -v_dot_mean,  # negate to turn decrease into increase
            clipped=True
        )


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
            (violation**0.5).mean()
            if violation.numel() > 0
            else V_initial.new_tensor(0.0)
        )

        lyap_decay_loss = lyap_decay_loss

        # saturate the penalty to be betwenen 0 and 1 without clipping gradients
        lyap_decay_loss = torch.tanh(0.005 * lyap_decay_loss)
        decrease_satisfaction = 1.0 - lyap_decay_loss
        v_dot_fulfillment = torch.sigmoid(V_decrease * 100)

        # Scale the derivative violation to [0,1] (high when small)
        # 1 should be the convergence threshold which is 2.5e-6 and lowest when the violation is 1 or more

        if deriv_viol is not None:
            # make it a tensor if not already
            if not isinstance(deriv_viol, torch.Tensor):
                deriv_viol = torch.tensor(deriv_viol, dtype=V_initial.dtype, device=V_initial.device)

            deriv_viol = torch.clamp(deriv_viol, min=0.0, max=1.0)
            # scale to [0,1]
            deriv_viol = 1.0 - deriv_viol
            deriv_viol = deriv_viol ** 2 # make it sharper

        # Build FPL structure matching TF version
        fpl_structure = FPLConstraint(
            p_value=-2.0,  # TF uses 0.0 at top level (geometric mean)
            constraints={
                # "close_angles": close_angles**2,  # Direct tensor, not p_mean wrapped
                # "close_x" : close_x**2,          # Direct tensor, not p_mean wrapped

                "lyapunov": FPLConstraint(
                    p_value=-0.0,  # TF uses 0.0 here too
                    constraints={
                        # "pop": p_mean(proof_of_performance, -2.0),  # TF uses -1.0
                        # "exp_decrease": p_mean(exp_decrease_fulfillment, -2.0),  # TF uses -1.0
                        "decrease_satisfaction": p_mean(decrease_satisfaction, -6.0),  # TF uses -1.0
                        "v_dot": p_mean(v_dot_fulfillment, -2.0),
                        # "deriv_viol": p_mean(deriv_viol, -6.0) if deriv_viol is not None else V_initial.new_tensor(1.0),
                    },
                ),
            },
        )
        
        fulfillment = fpl_structure.evaluate()
        loss = 1.0 - fulfillment
        
        # Detached copy for logging
        def _detached_tree(c):
            if isinstance(c, torch.Tensor):
                return c.detach()
            if isinstance(c, FPLConstraint):
                return FPLConstraint(c.p_value, {k: _detached_tree(v) for k, v in c.constraints.items()})
            return c
        
        fpl_structure_log = _detached_tree(fpl_structure)
        
        return loss, fpl_structure_log


def train_with_fpl_milp(trainer, dut, args, x_lo, x_up):
    """
    FPL training using MILP-found adversarial states instead of random sampling.
    """
    import copy

    # Control pool size for MILP solutions
    dut.lyapunov_derivative_mip_pool_solutions = 100  # Increased to get top 100

    # Setup parameters like in original train_with_fpl
    params = []
    params += list(trainer.lyapunov_system.lyapunov_relu.parameters())
    params += list(trainer.R_options.variables())
    params += trainer.closed_loop_system.controller_variables()
    
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Initialize adversarial state pools
    derivative_adversarial_pool = torch.empty((0, trainer.lyapunov_system.system.x_dim), dtype=torch.float64)
    
    violation_threshold = 1e-4  # Threshold to switch to regular training
    max_adversarial_per_iter = 100  # Top N violations to collect
    max_total_adversarials = 1000  # Cap total pool size
    
    epoch = 0
    while epoch < args.pretrain_num_epochs:
        # Step 1: Find adversarial states using MILP
        print(f"\n=== MILP Adversarial Search at Epoch {epoch} ===")
        
        # Get derivative violations
        derivative_mip, derivative_obj, derivative_adversarial, derivative_adversarial_next = dut.solve_lyap_derivative_mip()
        
        print(f"Derivative violation: {derivative_obj:.6f}")
        
        # Check if violations are below threshold
        if derivative_obj < violation_threshold:
            print(f"Violations below threshold {violation_threshold}, switching to regular training")
            break
            
        if derivative_adversarial.shape[0] > 0:
            derivative_adversarial_pool = torch.cat([
                derivative_adversarial_pool,
                derivative_adversarial[:max_adversarial_per_iter]
            ], dim=0)
            if derivative_adversarial_pool.shape[0] > max_total_adversarials:
                derivative_adversarial_pool = derivative_adversarial_pool[-max_total_adversarials:]

        print(f"Collected {derivative_adversarial_pool.shape[0]} adversarial states for training")
        
        # Step 2: Train with FPL on adversarial states
        if derivative_adversarial_pool.shape[0] > 0:

            # Create batched dataset
            dataset = torch.utils.data.TensorDataset(derivative_adversarial_pool)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=min(args.batch_size, derivative_adversarial_pool.shape[0]), shuffle=True
            )
            
            # Run FPL training for a few sub-epochs on these adversarial states
            sub_epochs = derivative_adversarial_pool.shape[0] // 5
            sub_epochs = min(sub_epochs, 30)  

            for sub_epoch in range(sub_epochs):
                epoch_losses = []
                
                for batch_idx, (batch_states,) in enumerate(dataloader):
                    optimizer.zero_grad()

                    # Augment the batch with random states occasionally
                    if batch_idx % 4 == 0:
                        state_dim = batch_states.shape[1]
                        random_states = torch.empty((batch_states.shape[0], state_dim)).uniform_(0, 1)
                        random_states = x_lo + (x_up - x_lo) * random_states
                        batch_states = torch.cat([batch_states, random_states], dim=0)
                    
                    # Compute FPL loss with trajectory rollout
                    loss, fpl_structure = trainer.compute_fpl_loss(
                        batch_states,
                        min_horizon=min(sub_epoch // 3, 30),
                        max_horizon=40,
                    )
                    
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                print(f"  Sub-epoch {sub_epoch} Average Fulfillment: {1 - avg_loss:.4f}")

        epoch += 1
    
    print("\n=== Switching to regular gradient-based training ===")
    return trainer


def adaptive_fpl_milp_training(trainer, dut, args, x_lo, x_up):
    """
    Adaptive training that uses FPL on MILP violations, then switches to regular training.
    """
    switch_threshold = 5e-5  # Target convergence threshold
    patience = 5
    no_improve_count = 0
    best_violation = float('inf')
    
    # Phase 1: FPL training on MILP adversarial states
    print("=== Phase 1: FPL on MILP adversarial states ===")
    
    for epoch in range(args.pretrain_num_epochs):
        # Get current violations
        _, deriv_viol, deriv_adversarial, _ = dut.solve_lyap_derivative_mip()

        current_violation = deriv_viol
        
        print(f"Epoch {epoch}: Violation = {current_violation:.8f}")
        
        # Check if we should switch to regular training
        if current_violation < switch_threshold:
            print(f"✓ Violations below threshold {switch_threshold}")
            break
            
        # Check for improvement
        if current_violation < best_violation - 1e-7:  # Small tolerance for numerical noise
            best_violation = current_violation
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If no improvement for 'patience' epochs, switch strategy
        if no_improve_count >= patience:
            print(f"No improvement for {patience} epochs. Switching to MILP training.")
            dut.learning_rate = 5e-4  # args.learning_rate
            dut.lyapunov_positivity_mip_cost_weight = 0.0  # None
            dut.patience = patience
            dut.no_improve_count = 0
            dut.best_violation = float('inf')
            dut.train(torch.empty((0, 4), dtype=torch.float64))
            no_improve_count = 0  # Reset counter
            best_violation = float('inf')  # Reset best violation
            # reset the pool of adversarial states
            deriv_adversarial = torch.empty((0, trainer.lyapunov_system.system.x_dim), dtype=torch.float64)
            # reset epoch to continue FPL training
            epoch = 0
            continue

        # Continue FPL training on adversarial states
        if deriv_adversarial.shape[0] > 0:
            top_adversarial = deriv_adversarial 

            # Quick FPL training on these states
            params = list(trainer.lyapunov_system.lyapunov_relu.parameters()) + \
                    list(trainer.R_options.variables()) + \
                    trainer.closed_loop_system.controller_variables()
            optimizer = torch.optim.Adam(params, lr=args.learning_rate)
            
            for sub_iter in range(64):  # Few sub-iterations per epoch
                optimizer.zero_grad()
                # # Compute FPL loss using the MILP deriv_viol as one of the fulfillments
                # state_dim = top_adversarial.shape[1]
                # random_states = torch.empty((top_adversarial.shape[0], state_dim)).uniform_(0, 1)
                # random_states = x_lo + (x_up - x_lo) * random_states
                # # augment with random states to maintain diversity
                # top_adversarial = torch.cat([top_adversarial, random_states], dim=0)

                loss, _ = trainer.compute_fpl_loss(top_adversarial, min_horizon=3, max_horizon=30, deriv_viol=deriv_viol)
                loss.backward()
                optimizer.step()
    
    # Phase 2: Regular gradient-based training for final convergence
    print("\n=== Phase 2: Regular gradient-based training ===")
    dut.learning_rate = 3e-3
    dut.max_iterations = 1  # Or whatever you want
    success, _, _, final_deriv_viol = dut.train(torch.empty((0, 4), dtype=torch.float64))
    
    return success, final_deriv_viol

def train_with_fpl(trainer, state_samples, args, randomize, x_lo, x_up):
    """
    Enhanced training loop using FPL composition.
    """
    import copy
    
    params = []
    params += list(trainer.lyapunov_system.lyapunov_relu.parameters())
    params += list(trainer.R_options.variables())
    params += trainer.closed_loop_system.controller_variables()

    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Initialize best model tracking
    best_fulfillment = -float('inf')
    best_epoch = -1
    best_model_state = {
        'lyapunov_relu': None,
        'controller': None,
        'R_options': None,
        'fulfillment': None,
        'epoch': None
    }

    # Convert to batched dataset
    dataset = torch.utils.data.TensorDataset(state_samples)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    for epoch in range(args.pretrain_num_epochs):
        epoch_losses = []

        for batch_idx, (batch_states,) in enumerate(dataloader):
            optimizer.zero_grad()

            # create new randomized batch_states if randomize is True
            if randomize and batch_idx % 4 == 0:
                state_dim = batch_states.shape[1]
                random_states = torch.empty((args.batch_size, state_dim)).uniform_(0, 1)
                batch_states = x_lo + (x_up - x_lo) * random_states

            # Compute FPL loss with trajectory rollout
            loss, fpl_structure = trainer.compute_fpl_loss(
                batch_states,
                min_horizon=min(epoch // 3 + 3, 30),
                max_horizon=30,
            )

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

            epoch_losses.append(loss.item())

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  Fulfillment: {1 - loss.item():.4f}")
                print(f"  FPL Structure: {fpl_structure}")

        # Calculate average fulfillment for this epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_fulfillment = 1 - avg_loss
        
        print(f"Epoch {epoch} Average Fulfillment: {avg_fulfillment:.4f}")
        
        # Check if this is the best model so far
        if avg_fulfillment > best_fulfillment:
            best_fulfillment = avg_fulfillment
            best_epoch = epoch
            
            # Save the best model state (deep copy to preserve)
            best_model_state['lyapunov_relu'] = copy.deepcopy(trainer.lyapunov_system.lyapunov_relu.state_dict())
            best_model_state['controller'] = copy.deepcopy(trainer.closed_loop_system.controller_network.state_dict())
            
            # Handle R_options based on its type
            if hasattr(trainer.R_options, 'R'):
                if callable(trainer.R_options.R):
                    best_model_state['R_options'] = copy.deepcopy(trainer.R_options.R().detach())
                else:
                    best_model_state['R_options'] = copy.deepcopy(trainer.R_options.R.detach())
            else:
                best_model_state['R_options'] = copy.deepcopy(trainer.R_options.detach())
            
            best_model_state['fulfillment'] = best_fulfillment
            best_model_state['epoch'] = best_epoch
            
            print(f"  >> New best model! Fulfillment: {best_fulfillment:.4f}")

        # Early stopping if converged
        if avg_fulfillment >= 0.9:
            print("Converged!")
            break

    # Load the best model at the end
    print(f"\nRestoring best model from epoch {best_epoch} with fulfillment {best_fulfillment:.4f}")
    # trainer.lyapunov_system.lyapunov_relu.load_state_dict(best_model_state['lyapunov_relu'])
    # trainer.closed_loop_system.controller_network.load_state_dict(best_model_state['controller'])
    
    # Restore R_options
    if hasattr(trainer.R_options, 'set_variable_value'):
        trainer.R_options.set_variable_value(best_model_state['R_options'].numpy())
    
    # Optionally save to disk
    if hasattr(trainer, 'save_network_path') or hasattr(args, 'save_path'):
        save_path = getattr(trainer, 'save_network_path', getattr(args, 'save_path', './'))
        torch.save(trainer.lyapunov_system.lyapunov_relu, f"{save_path}/best_lyapunov.pt")
        torch.save(trainer.closed_loop_system.controller_network, f"{save_path}/best_controller.pt")
        torch.save(best_model_state['R_options'], f"{save_path}/best_R.pt")
        
        # Save metadata
        torch.save({
            'epoch': best_epoch,
            'fulfillment': best_fulfillment,
        }, f"{save_path}/best_model_info.pt")
        
        print(f"Best model saved to {save_path}")

    return trainer
