# teleop_cartpole.py
# Keyboard teleoperation for cart-pole system with model comparison
# State: x = [cart_pos, pole_angle, cart_vel, pole_angular_vel]
# Control: u = [force]

import argparse
import math
import numpy as np
import pygame
import torch
import sys
import os

# Add path if cart_pole.py is in the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cart_pole import Cart_Pole

# -------------- Helpers --------------


def wrap_angle(th: float) -> float:
    """Wrap angle to [-pi, pi]"""
    while th <= -math.pi:
        th += 2 * math.pi
    while th > math.pi:
        th -= 2 * math.pi
    return th


def clamp_state(x, x_lo, x_hi):
    """Clamp state within bounds"""
    x_clamped = x.copy()
    x_clamped[0] = np.clip(x[0], x_lo[0], x_hi[0])  # cart position
    x_clamped[1] = wrap_angle(x[1])  # pole angle (wrap)
    x_clamped[2] = np.clip(x[2], x_lo[2], x_hi[2])  # cart velocity
    x_clamped[3] = np.clip(x[3], x_lo[3], x_hi[3])  # pole angular velocity
    return x_clamped


# -------------- Network wrapper --------------


class ForwardModel:
    """Wrapper for learned forward dynamics model"""

    def __init__(
        self,
        model_path,
        device="cpu",
        outputs_delta=False,
        outputs_accelerations=False,
        dtype=torch.float32,
    ):
        self.outputs_delta = outputs_delta
        self.outputs_accelerations = outputs_accelerations
        self.device = torch.device(device)
        self.dtype = dtype

        if model_path is None or (
            isinstance(model_path, str) and model_path.lower() == "none"
        ):
            self.model = None
        else:
            self.model = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model.eval()

        # Define equilibrium point for cart-pole
        self.x_eq = np.array([0.0, 0.0, 0.0, 0.0])  # Upright at origin
        self.u_eq = np.array([0.0])

        if self.model is not None:
            # Test model output dimension
            test_input = torch.tensor(
                np.concatenate([self.x_eq, self.u_eq]), dtype=self.dtype
            )
            with torch.no_grad():
                test_output = self.model(test_input)
                self.output_dim = test_output.shape[0]

                # Auto-detect if model outputs accelerations (2D) or full state (4D)
                if self.output_dim == 2:
                    self.outputs_accelerations = True
                    print(f"Model outputs accelerations only (dim={self.output_dim})")
                    print(f"Equilibrium accelerations: {test_output.cpu().numpy()}")
                elif self.output_dim == 4:
                    print(f"Model outputs full state (dim={self.output_dim})")
                else:
                    print(
                        f"Warning: Unexpected model output dimension: {self.output_dim}"
                    )

                # Cache equilibrium output
                self.phi_eq = test_output

    def step(self, x, u, dt):
        """
        Step the model forward
        x: np.array(4,), u: np.array(1,)
        returns np.array(4,)
        """
        if self.model is None:
            return x

        xin = torch.tensor(np.concatenate([x, u]), dtype=self.dtype)
        with torch.no_grad():
            model_output = self.model(xin)

            if self.outputs_accelerations:
                # Model outputs [x_ddot, theta_ddot]
                # Subtract equilibrium accelerations
                x_ddot_pred = model_output[0].item() - self.phi_eq[0].item()
                theta_ddot_pred = model_output[1].item() - self.phi_eq[1].item()

                # Euler integration
                x_next_pred = np.array(
                    [
                        x[0] + x[2] * dt,  # x_next = x + x_dot * dt
                        x[1] + x[3] * dt,  # theta_next = theta + theta_dot * dt
                        x[2] + x_ddot_pred * dt,  # x_dot_next = x_dot + x_ddot * dt
                        x[3]
                        + theta_ddot_pred
                        * dt,  # theta_dot_next = theta_dot + theta_ddot * dt
                    ]
                )
            elif self.outputs_delta:
                # Model outputs Δx
                delta = model_output.cpu().numpy()
                x_next_pred = x + delta
            else:
                # Model outputs x_next (with equilibrium correction if 4D)
                if self.output_dim == 4:
                    x_next_pred = (
                        model_output
                        - self.phi_eq
                        + torch.tensor(self.x_eq, dtype=self.dtype)
                    )
                    x_next_pred = x_next_pred.cpu().numpy()
                else:
                    x_next_pred = model_output.cpu().numpy()

        # Wrap angle
        x_next_pred[1] = wrap_angle(x_next_pred[1])
        return x_next_pred


# -------------- Rendering --------------


class CartPoleRenderer:
    def __init__(self, screen_width=900, screen_height=600):
        self.W = screen_width
        self.H = screen_height
        self.world_width = 5.0  # meters shown (±2.5m)
        self.scale = self.W / self.world_width

        # Colors
        self.bg_color = (245, 245, 245)
        self.track_color = (100, 100, 100)
        self.cart_color_truth = (50, 50, 200)  # Blue for ground truth
        self.cart_color_net = (200, 50, 50)  # Red for network
        self.cart_color_single = (0, 160, 0)  # Green when not comparing
        self.pole_color_truth = (100, 100, 250)
        self.pole_color_net = (250, 100, 100)
        self.pole_color_single = (0, 200, 0)
        self.text_color = (0, 0, 0)
        self.grid_color = (220, 220, 220)

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates"""
        sx = int(x * self.scale + self.W / 2)
        sy = int(self.H / 2 - y * self.scale)  # flip y axis
        return sx, sy

    def draw_grid(self, screen):
        """Draw background grid"""
        # Vertical lines
        for x in np.arange(-2.5, 2.6, 0.5):
            sx, _ = self.world_to_screen(x, 0)
            pygame.draw.line(screen, self.grid_color, (sx, 0), (sx, self.H), 1)

        # Horizontal line at ground
        pygame.draw.line(
            screen, self.track_color, (0, self.H // 2), (self.W, self.H // 2), 3
        )

    def draw_cartpole(
        self,
        screen,
        state,
        cart_color,
        pole_color,
        cart_width=0.4,
        cart_height=0.2,
        pole_length=1.0,
        alpha=255,
        offset_y=0,
    ):
        """Draw the cart-pole system"""
        x_cart = state[0]
        theta = state[1]

        # Cart position
        cart_x, cart_y = self.world_to_screen(x_cart, offset_y)
        cart_w = int(cart_width * self.scale)
        cart_h = int(cart_height * self.scale)

        # Create surface for transparency if needed
        if alpha < 255:
            cart_surf = pygame.Surface((cart_w, cart_h))
            cart_surf.set_alpha(alpha)
            cart_surf.fill(cart_color)
            screen.blit(cart_surf, (cart_x - cart_w // 2, cart_y - cart_h // 2))
        else:
            # Draw cart
            cart_rect = pygame.Rect(
                cart_x - cart_w // 2, cart_y - cart_h // 2, cart_w, cart_h
            )
            pygame.draw.rect(screen, cart_color, cart_rect)
            pygame.draw.rect(screen, (0, 0, 0), cart_rect, 2)

        # Draw wheels
        wheel_radius = int(0.05 * self.scale)
        wheel_color = tuple(int(c * 0.3) for c in cart_color)
        pygame.draw.circle(
            screen,
            wheel_color,
            (cart_x - cart_w // 3, cart_y + cart_h // 2),
            wheel_radius,
        )
        pygame.draw.circle(
            screen,
            wheel_color,
            (cart_x + cart_w // 3, cart_y + cart_h // 2),
            wheel_radius,
        )

        # Draw pole
        pole_end_x = x_cart + pole_length * np.sin(theta)
        pole_end_y = pole_length * np.cos(theta) + offset_y
        pole_end_screen = self.world_to_screen(pole_end_x, pole_end_y)

        pygame.draw.line(screen, pole_color, (cart_x, cart_y), pole_end_screen, 8)

        # Draw pole mass
        mass_color = tuple(int(c * 0.8) for c in pole_color)
        pygame.draw.circle(screen, mass_color, pole_end_screen, 12)

        # Draw pivot point
        pygame.draw.circle(screen, (0, 0, 0), (cart_x, cart_y), 6)


# -------------- Main Application --------------


def main():
    parser = argparse.ArgumentParser(
        description="Cart-Pole Teleoperation with Model Comparison"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to torch model (φ: [x;u]->x_next or [x;u]->accelerations)",
    )
    parser.add_argument(
        "--model-outputs-delta",
        action="store_true",
        help="Set if model outputs Δx instead of x_next",
    )
    parser.add_argument(
        "--model-outputs-accelerations",
        action="store_true",
        help="Set if model outputs only accelerations [x_ddot, theta_ddot]",
    )
    parser.add_argument(
        "--dt", type=float, default=0.02, help="Time step for simulation"
    )
    parser.add_argument(
        "--force-max", type=float, default=20.0, help="Maximum force magnitude"
    )
    parser.add_argument(
        "--force-step", type=float, default=2.0, help="Force increment per key press"
    )
    parser.add_argument(
        "--x-init",
        type=float,
        nargs=4,
        default=[0.0, 0.1, 0.0, 0.0],
        help="Initial state [x, theta, x_dot, theta_dot]",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Start in compare mode (ground truth vs network)",
    )
    parser.add_argument(
        "--lqr", action="store_true", help="Enable LQR stabilization mode"
    )
    args = parser.parse_args()

    # Initialize pygame
    pygame.init()
    clock = pygame.time.Clock()
    renderer = CartPoleRenderer()
    screen = pygame.display.set_mode((renderer.W, renderer.H))
    pygame.display.set_caption("Cart-Pole: Ground Truth vs Network Model")
    font = pygame.font.SysFont("consolas", 16)

    # Initialize cart-pole system
    plant = Cart_Pole(torch.float64)

    # Initialize model - use float64 to match your model
    model = ForwardModel(
        args.model,
        outputs_delta=args.model_outputs_delta,
        outputs_accelerations=args.model_outputs_accelerations,
        dtype=torch.float64,
    )

    # States and control
    x_truth = np.array(args.x_init, dtype=float)
    x_net = np.array(args.x_init, dtype=float)
    u = np.array([0.0], dtype=float)

    # State bounds
    x_lo = np.array([-2.5, -np.pi, -10.0, -10.0])
    x_hi = np.array([2.5, np.pi, 10.0, 10.0])

    # LQR controller (if enabled)
    if args.lqr:
        x_eq = np.array([0.0, 0.0, 0.0, 0.0])
        u_eq = np.array([0.0])
        Q = np.diag([1.0, 10.0, 0.1, 1.0])
        R = np.array([[0.01]])
        K, S = plant.lqr_control(Q, R, x_eq, u_eq)
        print(f"LQR gain K: {K}")

    # Traces for visualization
    truth_trace = []
    net_trace = []
    max_trace_len = 500

    # Simulation variables
    running = True
    paused = False
    lqr_active = args.lqr
    mode_compare = args.compare and (model.model is not None)

    # Main loop
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        # Exit
        if keys[pygame.K_ESCAPE]:
            running = False

        # Pause
        if keys[pygame.K_SPACE]:
            paused = not paused
            pygame.time.wait(200)

        # Reset
        if keys[pygame.K_r]:
            x_truth = np.array(args.x_init, dtype=float)
            x_net = np.array(args.x_init, dtype=float)
            u = np.array([0.0], dtype=float)
            truth_trace.clear()
            net_trace.clear()

        # Toggle compare mode
        if keys[pygame.K_m] and model.model is not None:
            mode_compare = not mode_compare
            pygame.time.wait(200)

        # Toggle LQR
        if keys[pygame.K_l] and args.lqr:
            lqr_active = not lqr_active
            pygame.time.wait(200)

        # Manual control
        if not lqr_active:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                u[0] = max(-args.force_max, u[0] - args.force_step)
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                u[0] = min(args.force_max, u[0] + args.force_step)
            else:
                # Decay force towards zero
                u[0] *= 0.9

            # Quick force set
            if keys[pygame.K_z]:
                u[0] = 0.0
            if keys[pygame.K_q]:
                u[0] = -args.force_max
            if keys[pygame.K_e]:
                u[0] = args.force_max

        # LQR control (use ground truth state)
        if lqr_active and args.lqr:
            u = -K @ x_truth
            u = np.clip(u, -args.force_max, args.force_max)

        # Step simulation (if not paused)
        if not paused:
            # Ground truth dynamics
            x_truth_next = plant.next_pose(x_truth, u, args.dt)
            x_truth = clamp_state(x_truth_next, x_lo, x_hi)

            # Network dynamics
            if model.model is not None:
                x_net_next = model.step(x_net, u, args.dt)
                x_net = clamp_state(x_net_next, x_lo, x_hi)
            else:
                x_net = x_truth.copy()

            # Store traces
            truth_trace.append((x_truth[0], x_truth[1]))
            net_trace.append((x_net[0], x_net[1]))
            if len(truth_trace) > max_trace_len:
                truth_trace.pop(0)
            if len(net_trace) > max_trace_len:
                net_trace.pop(0)

        # Render
        screen.fill(renderer.bg_color)
        renderer.draw_grid(screen)

        # Draw traces
        if mode_compare:
            # Truth trace (blue)
            if len(truth_trace) > 1:
                for i in range(1, len(truth_trace)):
                    sx1, sy1 = renderer.world_to_screen(truth_trace[i - 1][0], 0)
                    sx2, sy2 = renderer.world_to_screen(truth_trace[i][0], 0)
                    pygame.draw.line(screen, (150, 150, 250), (sx1, sy1), (sx2, sy2), 2)

            # Network trace (red)
            if len(net_trace) > 1:
                for i in range(1, len(net_trace)):
                    sx1, sy1 = renderer.world_to_screen(net_trace[i - 1][0], 0)
                    sx2, sy2 = renderer.world_to_screen(net_trace[i][0], 0)
                    pygame.draw.line(screen, (250, 150, 150), (sx1, sy1), (sx2, sy2), 2)
        else:
            # Single trace (green)
            trace = net_trace if model.model is not None else truth_trace
            if len(trace) > 1:
                for i in range(1, len(trace)):
                    sx1, sy1 = renderer.world_to_screen(trace[i - 1][0], 0)
                    sx2, sy2 = renderer.world_to_screen(trace[i][0], 0)
                    pygame.draw.line(screen, (150, 250, 150), (sx1, sy1), (sx2, sy2), 2)

        # Draw cart-poles
        if mode_compare:
            # Draw both systems with slight vertical offset for clarity
            renderer.draw_cartpole(
                screen,
                x_truth,
                renderer.cart_color_truth,
                renderer.pole_color_truth,
                offset_y=-0.05,
                alpha=200,
            )
            renderer.draw_cartpole(
                screen,
                x_net,
                renderer.cart_color_net,
                renderer.pole_color_net,
                offset_y=0.05,
                alpha=200,
            )
        else:
            # Draw single system
            state = x_net if model.model is not None else x_truth
            renderer.draw_cartpole(
                screen, state, renderer.cart_color_single, renderer.pole_color_single
            )

        # Draw force arrow
        if abs(u[0]) > 0.1:
            ref_state = (
                x_truth
                if mode_compare
                else (x_net if model.model is not None else x_truth)
            )
            cart_x, cart_y = renderer.world_to_screen(ref_state[0], 0)
            arrow_len = int(abs(u[0]) * 3)
            arrow_dir = 1 if u[0] > 0 else -1
            arrow_end = (cart_x + arrow_dir * arrow_len, cart_y)
            pygame.draw.line(screen, (0, 200, 0), (cart_x, cart_y), arrow_end, 4)
            # Arrowhead
            pygame.draw.polygon(
                screen,
                (0, 200, 0),
                [
                    arrow_end,
                    (arrow_end[0] - arrow_dir * 8, arrow_end[1] - 5),
                    (arrow_end[0] - arrow_dir * 8, arrow_end[1] + 5),
                ],
            )

        # Calculate errors if in compare mode
        if mode_compare:
            pos_error = abs(x_net[0] - x_truth[0])
            angle_error = abs(wrap_angle(x_net[1] - x_truth[1]))
            state_error = np.linalg.norm(x_net - x_truth)

        # HUD text
        lines = []

        if mode_compare:
            lines.extend(
                [
                    f"Mode: COMPARE (Truth=Blue, Network=Red) | {'PAUSED' if paused else 'RUNNING'}",
                    f"Truth: x={x_truth[0]:+.3f}m, θ={math.degrees(x_truth[1]):+.1f}°, "
                    f"ẋ={x_truth[2]:+.3f}m/s, θ̇={x_truth[3]:+.3f}rad/s",
                    f"Network: x={x_net[0]:+.3f}m, θ={math.degrees(x_net[1]):+.1f}°, "
                    f"ẋ={x_net[2]:+.3f}m/s, θ̇={x_net[3]:+.3f}rad/s",
                    f"Errors: Δx={pos_error:.3e}m, Δθ={math.degrees(angle_error):.2f}°, "
                    f"||Δstate||={state_error:.3e}",
                ]
            )
        else:
            state = x_net if model.model is not None else x_truth
            mode_str = "Network" if model.model is not None else "Ground Truth"
            lines.extend(
                [
                    f"Mode: {mode_str} | {'LQR' if lqr_active else 'Manual'} | "
                    f"{'PAUSED' if paused else 'RUNNING'}",
                    f"State: x={state[0]:+.3f}m, θ={math.degrees(state[1]):+.1f}°, "
                    f"ẋ={state[2]:+.3f}m/s, θ̇={state[3]:+.3f}rad/s",
                ]
            )

        lines.extend(
            [
                f"Force: {u[0]:+.2f}N",
                "",
                "Controls:",
                "  ←/→ or A/D: Apply force | Z: Zero | Q/E: Max left/right",
                "  R: Reset | SPACE: Pause | ESC: Quit",
            ]
        )

        if model.model is not None:
            lines.append("  M: Toggle compare mode")
        if args.lqr:
            lines.append("  L: Toggle LQR mode")

        y = 10
        for line in lines:
            txt = font.render(line, True, renderer.text_color)
            screen.blit(txt, (10, y))
            y += 20

        # Update display
        pygame.display.flip()
        clock.tick(int(1.0 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
