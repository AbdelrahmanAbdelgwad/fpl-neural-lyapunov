# teleop_pendulum.py
# Keyboard teleop / visualizer for PENDULUM dynamics with animation.
# State: x = [theta, thetadot]; Control: u = [torque]
# Analytic dynamics:
#   d/dt theta    = thetadot
#   d/dt thetadot = (u - mgl*sin(theta) - b*thetadot) / (ml^2)
#
# Features:
# - Dual visualization: pendulum animation + phase portrait
# - Keyboard control or custom controller function
# - Compare analytic vs learned forward model
#
# Controls:
#   A / D : decrease / increase torque (keyboard mode)
#   X     : zero torque
#   R     : reset state to --init
#   M     : toggle compare mode (draw analytic trace too)
#   V     : toggle view (pendulum/phase/both)
#   C     : toggle clamping to bounds
#   T / G : shorten / lengthen trace
#   ESC   : quit
#
# Example:
#   # Keyboard control
#   python teleop_pendulum.py --dt 0.02
#
#   # With learned model
#   python teleop_pendulum.py --model path/to/model.pt --compare
#
#   # With LQR controller
#   python teleop_pendulum.py --controller lqr --lqr-Q 10,1 --lqr-R 1
#
# Requirements: pygame, torch, numpy, scipy

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Callable

import numpy as np
import pygame
import scipy.linalg
import torch
import torch.nn as nn

import neural_network_lyapunov.utils as utils


def wrap_angle(th: float) -> float:
    # wrap to [-pi, pi]
    while th <= -math.pi:
        th += 2 * math.pi
    while th > math.pi:
        th -= 2 * math.pi
    return th


def clamp_state(
    x: np.ndarray, bounds_lo: np.ndarray, bounds_hi: np.ndarray
) -> np.ndarray:
    # x = [theta, thetadot]
    x[0] = wrap_angle(x[0])  # keep θ wrapped
    x[1] = max(bounds_lo[1], min(bounds_hi[1], x[1]))
    return x


def euler_step_pendulum(
    x: np.ndarray,
    u: float,
    mass: float,
    gravity: float,
    length: float,
    damping: float,
    dt: float,
) -> np.ndarray:
    """
    One Euler step of the analytic pendulum dynamics.
    x = [theta, thetadot], u = torque (float).
    """
    theta, thetadot = float(x[0]), float(x[1])

    # Dynamics
    theta_dot = thetadot
    thetadot_dot = (
        float(u) - mass * gravity * length * math.sin(theta) - damping * thetadot
    ) / (mass * length * length)

    x_next = np.array(
        [
            wrap_angle(theta + theta_dot * dt),
            thetadot + thetadot_dot * dt,
        ],
        dtype=float,
    )
    return x_next


class ForwardModelPendulum:
    """
    Thin wrapper around a Torch model φ: [x;u] -> x_next.
    Applies equilibrium subtraction if requested:
       x_next = φ([x;u]) - φ([x_eq;u_eq]) + x_eq
    """

    def __init__(
        self,
        model_path: Optional[str],
        u_eq: float = 0.0,
        x_eq=(math.pi, 0.0),
        device="cpu",
        dt: float = 0.01,
        output_mode: str = "xnext",
    ):
        self.device = torch.device(device)
        self.x_eq = np.array(x_eq, dtype=float)
        self.u_eq = float(u_eq)
        self.dt = float(dt)
        assert output_mode in ("xnext", "vnext", "thetaddot")
        self.output_mode = output_mode

        if model_path is None or (
            isinstance(model_path, str) and model_path.lower() == "none"
        ):
            self.model = None
            self.phi_eq = None
        else:
            dynamics_model_data = torch.load(model_path)
            self.model = utils.setup_relu(
                dynamics_model_data["linear_layer_width"],
                params=None,
                negative_slope=dynamics_model_data["negative_slope"],
                bias=True,
                dtype=torch.float64,
            )
            self.model.load_state_dict(dynamics_model_data["state_dict"])
            self.model.eval()
            xeu = torch.tensor(
                np.array([self.x_eq[0], self.x_eq[1], self.u_eq]), dtype=torch.float64
            )
            with torch.no_grad():
                phi_out = self.model(xeu)
                # Handle both 1D and 2D output
                if len(phi_out.shape) == 1 and phi_out.shape[0] == 1:
                    # Model outputs theta_ddot only
                    self.phi_eq = torch.tensor([0, phi_out[0]], dtype=torch.float64)
                    self.outputs_thetaddot_only = True
                else:
                    # Model outputs full state
                    self.phi_eq = phi_out
                    self.outputs_thetaddot_only = False

    def step(self, x: np.ndarray, u: float) -> np.ndarray:
        if self.model is None:
            return x.copy()
        xin = torch.tensor(np.array([x[0], x[1], float(u)]), dtype=torch.float64)
        with torch.no_grad():
            phi_out = self.model(xin)

            if phi_out.ndim == 0:  # ensure shape [1]
                phi_out = phi_out.view(1)
            if phi_out.shape[0] == 1:
                val = (phi_out[0] - self.phi_eq[1]).item()
                if self.output_mode == "thetaddot":
                    # xdot = [thetadot, thetaddot]; Euler integrate
                    theta_next = wrap_angle(x[0] + x[1] * self.dt)
                    v_next = x[1] + val * self.dt
                    x_next_pred = torch.tensor(
                        [theta_next, v_next], dtype=torch.float64
                    )
                else:  # "vnext": model predicts thetadot_{n+1}
                    v_next = val
                    theta_next = wrap_angle(x[0] + v_next * self.dt)
                    x_next_pred = torch.tensor(
                        [theta_next, v_next], dtype=torch.float64
                    )
            else:
                # full x_{n+1}
                x_next_pred = (
                    phi_out - self.phi_eq + torch.tensor(self.x_eq, dtype=torch.float64)
                )

        x_next = x_next_pred.detach().cpu().numpy().astype(float)
        x_next[0] = wrap_angle(x_next[0])
        return x_next


# Controllers
def create_lqr_controller(
    Q: np.ndarray,
    R: np.ndarray,
    mass: float,
    gravity: float,
    length: float,
    damping: float,
) -> Callable:
    """Create an LQR controller around upright equilibrium (π, 0)."""
    # Linearize around (π, 0)
    A = np.array([[0, 1], [gravity / length, -damping / (mass * length * length)]])
    B = np.array([[0], [1 / (mass * length * length)]])

    # Solve Riccati equation
    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ S)

    def controller(x: np.ndarray) -> float:
        # Control around upright
        x_err = np.array([wrap_angle(x[0] - math.pi), x[1]])
        u = float(K @ x_err)
        return u

    return controller


def create_energy_controller(
    gain: float, mass: float, gravity: float, length: float
) -> Callable:
    """Create an energy-shaping swing-up controller."""

    def controller(x: np.ndarray) -> float:
        theta, thetadot = x[0], x[1]

        # Desired energy (upright position)
        E_des = mass * gravity * length  # PE at top, KE = 0

        # Current energy
        ke = 0.5 * mass * (length * thetadot) ** 2
        pe = -mass * gravity * length * math.cos(theta)
        E = ke + pe

        # Energy-based control
        if abs(thetadot) < 0.01:
            u = -gain * (E - E_des)
        else:
            u = -gain * thetadot * (E - E_des)

        return float(u)

    return controller


def create_pid_controller(
    kp: float, ki: float, kd: float, target: float = math.pi
) -> Callable:
    """Create a PID controller."""
    integral = 0.0
    prev_error = 0.0

    def controller(x: np.ndarray) -> float:
        nonlocal integral, prev_error

        error = wrap_angle(target - x[0])
        integral += error * 0.01  # assuming small dt
        derivative = (error - prev_error) / 0.01
        prev_error = error

        u = kp * error + ki * integral - kd * x[1]  # using thetadot for D term
        return float(u)

    return controller


def create_monotonic_controller(model: nn.Module) -> Callable:
    """Create a monotonic controller using a neural network."""

    def controller(x: np.ndarray) -> float:
        # Forward pass through the model
        with torch.no_grad():
            input_tensor = torch.tensor(x, dtype=torch.double).unsqueeze(0)
            output = model(input_tensor)
            return float(output)

    return controller


@dataclass
class View:
    W: int = 1400
    H: int = 700

    # Split screen: pendulum on left, phase portrait on right
    pendulum_center = (350, 350)
    pendulum_scale = 250  # pixels per meter

    phase_center = (1050, 350)
    phase_width = 600
    phase_height = 600
    theta_half_extent: float = math.pi
    thetadot_half_extent: float = 10.0

    bg_color = (245, 245, 245)
    grid_color = (220, 220, 220)
    truth_color = (250, 100, 100)  # analytic
    net_color = (20, 100, 220)  # network
    same_color = (0, 160, 0)
    text_color = (0, 0, 0)
    pendulum_color = (50, 50, 50)

    def world_to_phase_screen(self, p: Tuple[float, float]) -> Tuple[int, int]:
        # p: (theta, thetadot)
        sx = int(
            (p[0] / self.theta_half_extent) * (self.phase_width / 2)
            + self.phase_center[0]
        )
        sy = int(
            self.phase_center[1]
            - (p[1] / self.thetadot_half_extent) * (self.phase_height / 2)
        )
        return sx, sy


def draw_pendulum(screen, view, x, color, length=1.0):
    """Draw the pendulum at state x."""
    theta = x[0]

    # Pendulum pivot
    pivot_x, pivot_y = view.pendulum_center

    # Pendulum bob position
    bob_x = pivot_x + view.pendulum_scale * length * math.sin(theta)
    bob_y = pivot_y + view.pendulum_scale * length * math.cos(theta)

    # Draw rod
    pygame.draw.line(screen, color, (pivot_x, pivot_y), (int(bob_x), int(bob_y)), 4)

    # Draw pivot
    pygame.draw.circle(screen, (100, 100, 100), (pivot_x, pivot_y), 8)

    # Draw bob
    pygame.draw.circle(screen, color, (int(bob_x), int(bob_y)), 20)


def draw_phase_portrait(screen, view):
    """Draw phase portrait axes and grid."""
    cx, cy = view.phase_center
    hw = view.phase_width // 2
    hh = view.phase_height // 2

    # Background
    pygame.draw.rect(
        screen, (250, 250, 250), (cx - hw, cy - hh, view.phase_width, view.phase_height)
    )

    # Grid
    for t in np.linspace(-view.theta_half_extent, view.theta_half_extent, 13):
        sx1, _ = view.world_to_phase_screen((t, -view.thetadot_half_extent))
        sx2, _ = view.world_to_phase_screen((t, view.thetadot_half_extent))
        pygame.draw.line(screen, view.grid_color, (sx1, cy - hh), (sx2, cy + hh), 1)

    for t in np.linspace(-view.thetadot_half_extent, view.thetadot_half_extent, 13):
        _, sy1 = view.world_to_phase_screen((-view.theta_half_extent, t))
        _, sy2 = view.world_to_phase_screen((view.theta_half_extent, t))
        pygame.draw.line(screen, view.grid_color, (cx - hw, sy1), (cx + hw, sy2), 1)

    # Axes
    pygame.draw.line(screen, (180, 180, 180), (cx - hw, cy), (cx + hw, cy), 2)
    pygame.draw.line(screen, (180, 180, 180), (cx, cy - hh), (cx, cy + hh), 2)

    # Equilibrium points
    for eq_theta in [0, math.pi, -math.pi]:
        sx, sy = view.world_to_phase_screen((eq_theta, 0))
        pygame.draw.circle(screen, (100, 100, 100), (sx, sy), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="neural_network_lyapunov/examples/pendulum/data/pendulum_second_order_forward_relu.pt",
        help="Path to torch model (φ: [theta, thetadot, u]->x_next).",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    parser.add_argument(
        "--model-output",
        type=str,
        default="vnext",
        choices=["xnext", "vnext", "thetaddot"],
        help="What a 1-D model predicts.",
    )
    parser.add_argument(
        "--init",
        type=float,
        nargs=2,
        default=[0.5, 0.0],
        help="Initial [theta, thetadot].",
    )

    # Pendulum parameters
    parser.add_argument("--mass", type=float, default=1.0, help="Pendulum mass (kg).")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity (m/s^2).")
    parser.add_argument(
        "--length", type=float, default=1.0, help="Pendulum length (m)."
    )
    parser.add_argument(
        "--damping", type=float, default=0.1, help="Damping coefficient."
    )

    # Control options
    parser.add_argument(
        "--controller",
        type=str,
        default="NN",
        choices=["keyboard", "lqr", "energy", "pid", "zero", "NN"],
        help="Controller type.",
    )
    parser.add_argument("--umax", type=float, default=20.0, help="Max |torque| (N⋅m).")
    parser.add_argument(
        "--du", type=float, default=0.5, help="Torque increment (keyboard)."
    )

    # Controller-specific parameters
    parser.add_argument(
        "--lqr-Q",
        type=str,
        default="10,1",
        help="LQR Q matrix diagonal (comma-separated).",
    )
    parser.add_argument("--lqr-R", type=float, default=1.0, help="LQR R value.")
    parser.add_argument(
        "--energy-gain", type=float, default=2.0, help="Energy controller gain."
    )
    parser.add_argument("--pid-kp", type=float, default=10.0, help="PID P gain.")
    parser.add_argument("--pid-ki", type=float, default=0.1, help="PID I gain.")
    parser.add_argument("--pid-kd", type=float, default=2.0, help="PID D gain.")

    # Display options
    parser.add_argument(
        "--view",
        type=str,
        default="both",
        choices=["pendulum", "phase", "both"],
        help="Initial view mode.",
    )
    parser.add_argument(
        "--clamp-bounds", action="store_true", help="Clamp thetadot to ±10 rad/s."
    )
    parser.add_argument("--compare", action="store_true", help="Start in compare mode.")

    # Model equilibrium
    parser.add_argument(
        "--x-eq",
        type=float,
        nargs=2,
        default=[math.pi, 0.0],
        help="Equilibrium state for model.",
    )
    parser.add_argument(
        "--u-eq", type=float, default=0.0, help="Equilibrium control for model."
    )

    args = parser.parse_args()

    # Create controller
    controller = None
    if args.controller == "lqr":
        Q_diag = [float(x) for x in args.lqr_Q.split(",")]
        Q = np.diag(Q_diag)
        R = np.array([[args.lqr_R]])
        controller = create_lqr_controller(
            Q, R, args.mass, args.gravity, args.length, args.damping
        )
    elif args.controller == "energy":
        controller = create_energy_controller(
            args.energy_gain, args.mass, args.gravity, args.length
        )
    elif args.controller == "pid":
        controller = create_pid_controller(args.pid_kp, args.pid_ki, args.pid_kd)
    elif args.controller == "zero":
        controller = lambda x: 0.0

    elif args.controller == "NN":
        # "neural_network_lyapunov/examples/pendulum/data/monotonic_bound10_controller.pt"
        # "neural_network_lyapunov/examples/pendulum/data/pendulum_controller4.pt"
        # "neural_network_lyapunov/examples/examples_in_paper/pendulum/controller19.pt"
        controller_path = "neural_network_lyapunov/examples/pendulum/data/monotonic_bound5/monotonic_bound10_controller.pt"
        try:

            controller_relu = torch.load(controller_path)
            controller = create_monotonic_controller(controller_relu)
            # test out the model on dummy input
            with torch.no_grad():
                dummy_input = torch.zeros(1, 2)
                dummy_output = controller(dummy_input)
                print("Dummy output:", dummy_output)
        except:
            dynamics_controller_data = torch.load(controller_path)
            controller = utils.setup_relu(
                dynamics_controller_data["linear_layer_width"],
                params=None,
                negative_slope=dynamics_controller_data["negative_slope"],
                bias=True,
                dtype=torch.float64,
            )
            controller.load_state_dict(dynamics_controller_data["state_dict"])
            controller = create_monotonic_controller(controller)
            # test out the model on dummy input
            with torch.no_grad():
                dummy_input = torch.zeros(1, 2)
                dummy_output = controller(dummy_input)
                print("Dummy output:", dummy_output)

    pygame.init()
    clock = pygame.time.Clock()
    view = View()
    screen = pygame.display.set_mode((view.W, view.H))
    pygame.display.set_caption("Pendulum Tele-Op")

    font = pygame.font.SysFont("consolas", 16)

    # Control (torque)
    u = 0.0

    # States
    x_truth = np.array(args.init, dtype=float)
    x_net = np.array(args.init, dtype=float)

    # Bounds
    x_lo = np.array([0, -5.0], dtype=float)
    x_hi = np.array([+2 * math.pi, +5.0], dtype=float)

    # Model
    model = ForwardModelPendulum(
        args.model,
        u_eq=args.u_eq,
        x_eq=tuple(args.x_eq),
        dt=args.dt,
        output_mode=args.model_output,
    )

    # Traces
    trace_len = 1500
    truth_trace = []
    net_trace = []

    view_mode = args.view
    mode_compare = args.compare
    running = True

    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # Keyboard control
        if args.controller == "keyboard":
            if keys[pygame.K_a]:
                u = max(-args.umax, u - args.du)
            if keys[pygame.K_d]:
                u = min(args.umax, u + args.du)
            if keys[pygame.K_x]:
                u = 0.0
        else:
            # Use controller
            # u = controller(x_truth)
            u = controller(x_net)
            u = np.clip(u, -args.umax, args.umax)

        if keys[pygame.K_r]:
            x_truth[:] = args.init
            x_net[:] = args.init
            truth_trace.clear()
            net_trace.clear()
        if keys[pygame.K_m]:
            mode_compare = not mode_compare
            pygame.time.wait(150)
        if keys[pygame.K_v]:
            modes = ["pendulum", "phase", "both"]
            view_mode = modes[(modes.index(view_mode) + 1) % 3]
            pygame.time.wait(150)
        if keys[pygame.K_c]:
            args.clamp_bounds = not args.clamp_bounds
            pygame.time.wait(150)
        if keys[pygame.K_t]:
            trace_len = max(200, trace_len - 200)
            pygame.time.wait(150)
        if keys[pygame.K_g]:
            trace_len = min(4000, trace_len + 200)
            pygame.time.wait(150)

        # Step dynamics
        x_truth_next = euler_step_pendulum(
            x_truth, u, args.mass, args.gravity, args.length, args.damping, args.dt
        )

        try:
            x_net_next = model.step(x_net, u)
        except Exception:
            x_net_next = x_net.copy()

        if args.clamp_bounds:
            x_truth_next = clamp_state(x_truth_next, x_lo, x_hi)
            x_net_next = clamp_state(x_net_next, x_lo, x_hi)

        x_truth = x_truth_next
        x_net = x_net_next

        # Update traces
        truth_trace.append((x_truth[0], x_truth[1]))
        net_trace.append((x_net[0], x_net[1]))
        if len(truth_trace) > trace_len:
            truth_trace.pop(0)
        if len(net_trace) > trace_len:
            net_trace.pop(0)

        # Draw
        screen.fill(view.bg_color)

        # Draw pendulum view
        if view_mode in ["pendulum", "both"]:
            # Draw ground
            pygame.draw.line(
                screen,
                (150, 150, 150),
                (view.pendulum_center[0] - 150, view.pendulum_center[1]),
                (view.pendulum_center[0] + 150, view.pendulum_center[1]),
                2,
            )

            if mode_compare:
                draw_pendulum(screen, view, x_truth, view.truth_color, args.length)
            draw_pendulum(
                screen,
                view,
                x_net,
                view.net_color if mode_compare else view.same_color,
                args.length,
            )

            # Label
            label = font.render("Pendulum View", True, view.text_color)
            screen.blit(label, (view.pendulum_center[0] - 50, 50))

        # Draw phase portrait
        if view_mode in ["phase", "both"]:
            draw_phase_portrait(screen, view)

            # Draw traces
            if len(truth_trace) > 1 and mode_compare:
                for i in range(1, len(truth_trace)):
                    pygame.draw.line(
                        screen,
                        view.truth_color,
                        view.world_to_phase_screen(truth_trace[i - 1]),
                        view.world_to_phase_screen(truth_trace[i]),
                        2,
                    )

            if len(net_trace) > 1:
                for i in range(1, len(net_trace)):
                    pygame.draw.line(
                        screen,
                        view.net_color if mode_compare else view.same_color,
                        view.world_to_phase_screen(net_trace[i - 1]),
                        view.world_to_phase_screen(net_trace[i]),
                        3 if not mode_compare else 2,
                    )

            # Draw current points
            if mode_compare:
                sx, sy = view.world_to_phase_screen((x_truth[0], x_truth[1]))
                pygame.draw.circle(screen, view.truth_color, (sx, sy), 6)

            sx, sy = view.world_to_phase_screen((x_net[0], x_net[1]))
            pygame.draw.circle(
                screen, view.net_color if mode_compare else view.same_color, (sx, sy), 8
            )

            # Label
            label = font.render("Phase Portrait", True, view.text_color)
            screen.blit(label, (view.phase_center[0] - 50, 50))

        # HUD
        ke = 0.5 * args.mass * (args.length * x_truth[1]) ** 2
        pe = -args.mass * args.gravity * args.length * math.cos(x_truth[0])
        energy = ke + pe

        err = float(np.linalg.norm(x_net - x_truth)) if mode_compare else 0.0

        lines = [
            f"Controller: {args.controller}   u={u:+.3f} N⋅m   Energy={energy:.2f} J",
            f"Mode: {'COMPARE' if mode_compare else 'NETWORK'}   View: {view_mode}   Clamp: {'ON' if args.clamp_bounds else 'OFF'}",
            f"x_truth=(θ={x_truth[0]:+.3f}, θ̇={x_truth[1]:+.3f})",
            f"x_net  =(θ={x_net[0]:+.3f}, θ̇={x_net[1]:+.3f})"
            + (f"  ‖Δx‖={err:.3e}" if mode_compare else ""),
            "",
            "Keys: [A/D] torque±, [X] zero, [R] reset, [M] compare, [V] view, [C] clamp, [T/G] trace±",
        ]

        y = view.H - 120
        for s in lines:
            txt = font.render(s, True, view.text_color)
            screen.blit(txt, (20, y))
            y += 18

        pygame.display.flip()
        clock.tick(int(1.0 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
