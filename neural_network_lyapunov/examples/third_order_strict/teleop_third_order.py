# Keyboard teleop / visualizer for THIRD-ORDER STRICT FEEDBACK dynamics with animation-like projections.
# State: x = [x1, x2, x3]; Control: u (scalar)
# Continuous-time dynamics (strict-feedback form):
#   x1_dot = e1 * x2
#   x2_dot = e2 * x3
#   x3_dot = e3 * x1^2 + e4 * u
# Integrated forward with simple Euler step.
#
# Features (inspired by teleop_pendulum.py):
# - Keyboard control or custom controller (LQR or NN)
# - Compare analytic vs learned forward model ("compare" mode)
# - Three 2D projections: (x1,x2), (x2,x3), (x1,x3)
# - Bounds clamping to the paper's box via --bound-level (x1,x2 in ±0.03*b; x3 in ±0.04*b)
#
# Controls:
#   A / D : decrease / increase u (keyboard mode)
#   X     : zero u
#   R     : reset state to --init
#   M     : toggle compare mode (draw analytic trace too)
#   C     : toggle clamping to bounds
#   T / G : shorten / lengthen trace
#   ESC   : quit
#
# Example usages:
#   # Keyboard control in the paper's domain (b=50 -> x1,x2∈[-1.5,1.5], x3∈[-2,2])
#   python teleop_third_order.py --controller keyboard --bound-level 50
#
#   # LQR around the origin with Q=diag(10,1,1), R=1, u∈[-30,30]
#   python teleop_third_order.py --controller lqr --lqr-Q 10,1,1 --lqr-R 1 --umax 30
#
#   # With a learned forward model (phi: [x;u] -> x_next) and NN controller (psi: x -> u)
#   python teleop_third_order.py --model path/to/phi.pt --controller NN --controller-model path/to/psi.pt --compare
#
# Requirements: pygame, torch, numpy, scipy

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Callable, List

import numpy as np
import pygame
import scipy.linalg
import torch
import torch.nn as nn

try:
    import neural_network_lyapunov.utils as utils  # for loading saved ReLU models
except Exception:
    utils = None  # optional


# ========================= Dynamics helpers ========================= #

def euler_step_third_order(x: np.ndarray, u: float, e1: float, e2: float, e3: float, e4: float, dt: float) -> np.ndarray:
    """One Euler step of the third-order strict feedback dynamics."""
    x1, x2, x3 = float(x[0]), float(x[1]), float(x[2])

    x1_dot = e1 * x2
    x2_dot = e2 * x3
    x3_dot = e3 * (x1 ** 2) + e4 * float(u)

    return np.array([x1 + x1_dot * dt, x2 + x2_dot * dt, x3 + x3_dot * dt], dtype=float)


class ForwardModelThirdOrder:
    """
    Thin wrapper around a Torch model φ: [x;u] -> x_next OR xdot.
    Applies equilibrium subtraction if requested:
       y = φ([x;u]) - φ([x_eq;u_eq]) + y_eq
    where y is x_next for output_mode="xnext" or xdot for "xdot".
    """

    def __init__(
        self,
        model_path: Optional[str],
        x_eq=(0.0, 0.0, 0.0),
        u_eq: float = 0.0,
        device: str = "cpu",
        dt: float = 0.02,
        output_mode: str = "xnext",
    ):
        self.device = torch.device(device)
        self.x_eq = np.array(x_eq, dtype=float)
        self.u_eq = float(u_eq)
        self.dt = float(dt)
        assert output_mode in ("xnext", "xdot")
        self.output_mode = output_mode

        self.model = None
        self.phi_eq = None

        if model_path is None or (isinstance(model_path, str) and model_path.lower() == "none"):
            return

        data = torch.load(model_path)
        # Two common save formats supported:
        #  (1) Direct nn.Module
        #  (2) Saved dict with state_dict + architecture hints (as used in the repo)
        if isinstance(data, nn.Module):
            self.model = data.to(self.device).eval()
        elif isinstance(data, dict) and "state_dict" in data:
            if utils is None:
                raise RuntimeError("Model file is a state_dict blob but utils.setup_relu is unavailable.")
            self.model = utils.setup_relu(
                data["linear_layer_width"],
                params=None,
                negative_slope=data.get("negative_slope", 0.0),
                bias=True,
                dtype=torch.float64,
            )
            self.model.load_state_dict(data["state_dict"])
            self.model.eval()
        else:
            raise RuntimeError("Unrecognized model checkpoint format.")

        with torch.no_grad():
            xeu = torch.tensor(np.concatenate([self.x_eq, [self.u_eq]]), dtype=torch.float64).to(self.device)
            phi_out = self.model(xeu)
            # Normalize to 1-D tensors
            if phi_out.ndim == 0:
                phi_out = phi_out.view(1)
            self.phi_eq = phi_out.clone().detach()

    def step(self, x: np.ndarray, u: float) -> np.ndarray:
        if self.model is None:
            return x.copy()
        xin = torch.tensor(np.concatenate([x.astype(float), [float(u)]]), dtype=torch.float64).to(self.device)
        with torch.no_grad():
            y = self.model(xin)
            if y.ndim == 0:
                y = y.view(1)
            y = y - self.phi_eq + self.phi_eq  # keep center stable but allow nonzero eq if model has biases

            if self.output_mode == "xnext":
                x_next = y
            else:  # xdot
                x_next = torch.tensor(x, dtype=torch.float64).to(self.device) + y * self.dt

        return x_next.detach().cpu().numpy().astype(float)


# ========================= Controllers ========================= #

def create_lqr_controller(Q: np.ndarray, R: np.ndarray, e1: float, e2: float, e3: float, e4: float) -> Callable[[np.ndarray], float]:
    """Create an LQR controller around the origin for the third-order strict-feedback system.
    Linearization at x=0 yields A=[[0,e1,0],[0,0,e2],[0,0,0]], B=[[0],[0],[e4]].
    """
    A = np.array([[0.0, e1, 0.0], [0.0, 0.0, e2], [0.0, 0.0, 0.0]], dtype=float)
    B = np.array([[0.0], [0.0], [e4]], dtype=float)

    S = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = -np.linalg.solve(R, B.T @ S)

    def controller(x: np.ndarray) -> float:
        u = float(K @ x.reshape(3, 1))
        return u

    return controller


def create_nn_controller(model_path: str, x_eq=(0.0, 0.0, 0.0), u_eq: float = 0.0) -> Callable[[np.ndarray], float]:
    """Create a monotonic NN controller centered at equilibrium: u = ψ(x) - ψ(x*) + u*."""
    data = torch.load(model_path)
    if isinstance(data, nn.Module):
        net = data
    elif isinstance(data, dict) and "state_dict" in data:
        if utils is None:
            raise RuntimeError("Controller is a state_dict blob but utils.setup_relu is unavailable.")
        net = utils.setup_relu(
            data["linear_layer_width"],
            params=None,
            negative_slope=data.get("negative_slope", 0.0),
            bias=True,
            dtype=torch.float64,
        )
        net.load_state_dict(data["state_dict"])
    else:
        raise RuntimeError("Unrecognized controller checkpoint format.")

    net.eval()
    with torch.no_grad():
        phi_eq = net(torch.tensor(x_eq, dtype=torch.double).unsqueeze(0)).squeeze()

    def controller(x: np.ndarray) -> float:
        with torch.no_grad():
            xin = torch.tensor(x, dtype=torch.double).unsqueeze(0)
            u_pre = net(xin).squeeze() - phi_eq + u_eq
            return float(u_pre)

    return controller


# ========================= Visualization helpers ========================= #

@dataclass
class View:
    W: int = 1400
    H: int = 700

    bg = (245, 245, 245)
    grid = (220, 220, 220)
    truth_color = (250, 100, 100)  # analytic
    net_color = (20, 100, 220)     # network
    same_color = (0, 160, 0)
    text = (0, 0, 0)

    # Three projection rectangles (x1,x2), (x2,x3), (x1,x3)
    margin = 40
    plot_w = 420
    plot_h = 420

    def rects(self) -> List[pygame.Rect]:
        left = self.margin
        top = 60
        r1 = pygame.Rect(left, top, self.plot_w, self.plot_h)  # (x1,x2)
        r2 = pygame.Rect(left + self.plot_w + self.margin, top, self.plot_w, self.plot_h)  # (x2,x3)
        r3 = pygame.Rect(left + 2*(self.plot_w + self.margin), top, self.plot_w, self.plot_h)  # (x1,x3)
        return [r1, r2, r3]


def world_to_screen(p: Tuple[float, float], rect: pygame.Rect, xrange: Tuple[float, float], yrange: Tuple[float, float]) -> Tuple[int, int]:
    (x, y) = p
    (xmin, xmax) = xrange
    (ymin, ymax) = yrange
    # clamp ratios to [0,1]
    rx = 0.0 if xmax == xmin else (x - xmin) / (xmax - xmin)
    ry = 0.0 if ymax == ymin else (y - ymin) / (ymax - ymin)
    sx = rect.left + int(rx * rect.width)
    sy = rect.bottom - int(ry * rect.height)
    return sx, sy


def draw_axes(screen, rect: pygame.Rect, xrange, yrange, view: View, xlabel: str, ylabel: str, font):
    pygame.draw.rect(screen, (250, 250, 250), rect)
    # Grid lines
    for t in np.linspace(xrange[0], xrange[1], 9):
        sx1, _ = world_to_screen((t, yrange[0]), rect, xrange, yrange)
        sx2, _ = world_to_screen((t, yrange[1]), rect, xrange, yrange)
        pygame.draw.line(screen, view.grid, (sx1, rect.top), (sx2, rect.bottom), 1)
    for t in np.linspace(yrange[0], yrange[1], 9):
        _, sy1 = world_to_screen((xrange[0], t), rect, xrange, yrange)
        _, sy2 = world_to_screen((xrange[1], t), rect, xrange, yrange)
        pygame.draw.line(screen, view.grid, (rect.left, sy1), (rect.right, sy2), 1)
    # Axes
    _, y0 = world_to_screen((xrange[0], 0.0), rect, xrange, yrange)
    x0, _ = world_to_screen((0.0, yrange[0]), rect, xrange, yrange)
    pygame.draw.line(screen, (180, 180, 180), (rect.left, y0), (rect.right, y0), 2)
    pygame.draw.line(screen, (180, 180, 180), (x0, rect.top), (x0, rect.bottom), 2)

    # Labels
    label = font.render(f"{xlabel} vs {ylabel}", True, view.text)
    screen.blit(label, (rect.left, rect.top - 22))


def draw_trace(screen, rect: pygame.Rect, pts: List[Tuple[float, float]], xrange, yrange, color, width=2):
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        p1 = world_to_screen(pts[i-1], rect, xrange, yrange)
        p2 = world_to_screen(pts[i], rect, xrange, yrange)
        pygame.draw.line(screen, color, p1, p2, width)


# ========================= Main app ========================= #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")

    # Dynamics coefficients
    parser.add_argument("--e", type=str, default="1.0,1.0,1.0,1.0", help="Coefficients e1,e2,e3,e4 for the dynamics.")

    # Init conditions
    parser.add_argument("--init", type=float, nargs=3, default=[0.3, 0.0, 0.0], help="Initial [x1, x2, x3].")

    # Control options
    parser.add_argument("--controller", type=str, default="keyboard", choices=["keyboard", "lqr", "NN", "zero"], help="Controller type.")
    parser.add_argument("--umax", type=float, default=30.0, help="Max |u|.")
    parser.add_argument("--du", type=float, default=0.5, help="Control increment (keyboard).")

    # LQR params
    parser.add_argument("--lqr-Q", type=str, default="10,1,1", help="LQR Q matrix diagonal (comma-separated).")
    parser.add_argument("--lqr-R", type=float, default=1.0, help="LQR R scalar.")

    # NN controller
    parser.add_argument("--controller-model", type=str, default="none", help="Path to controller model (ψ: x -> u).")

    # Forward model for compare mode
    parser.add_argument("--model", type=str, default="none", help="Path to forward model (φ: [x;u]->x_next or xdot).")
    parser.add_argument("--model-output", type=str, default="xnext", choices=["xnext", "xdot"], help="What the forward model predicts.")
    parser.add_argument("--x-eq", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Equilibrium state for model/controller.")
    parser.add_argument("--u-eq", type=float, default=0.0, help="Equilibrium control for model/controller.")

    # Display / behavior
    parser.add_argument("--clamp-bounds", action="store_true", help="Clamp state to box defined by --bound-level.")
    parser.add_argument("--bound-level", type=float, default=50.0, help="Bound level b; x1,x2∈±0.03*b and x3∈±0.04*b.")
    parser.add_argument("--compare", action="store_true", help="Start in compare mode (draw analytic + model).")

    args = parser.parse_args()

    e1, e2, e3, e4 = [float(v) for v in args.e.split(",")]

    # Build controller
    controller: Optional[Callable[[np.ndarray], float]] = None
    if args.controller == "lqr":
        Q_diag = [float(x) for x in args.lqr_Q.split(",")]
        Q = np.diag(Q_diag)
        R = np.array([[args.lqr_R]], dtype=float)
        controller = create_lqr_controller(Q, R, e1, e2, e3, e4)
    elif args.controller == "NN":
        if args.controller_model.lower() == "none":
            raise SystemExit("--controller NN requires --controller-model path to a ψ: x->u network.")
        controller = create_nn_controller(args.controller_model, tuple(args.x_eq), args.u_eq)
    elif args.controller == "zero":
        controller = lambda x: 0.0

    pygame.init()
    clock = pygame.time.Clock()
    view = View()
    screen = pygame.display.set_mode((view.W, view.H))
    pygame.display.set_caption("Third-Order Strict-Feedback Tele-Op")

    font = pygame.font.SysFont("consolas", 16)

    # State + control
    u = 0.0
    x_truth = np.array(args.init, dtype=float)
    x_net = np.array(args.init, dtype=float)

    # Bounds from bound_level
    b = float(args.bound_level)
    x_lo = np.array([-0.03 * b, -0.03 * b, -0.04 * b], dtype=float)
    x_hi = np.array([+0.03 * b, +0.03 * b, +0.04 * b], dtype=float)

    # Forward model
    model = ForwardModelThirdOrder(
        args.model,
        x_eq=tuple(args.x_eq),
        u_eq=args.u_eq,
        dt=args.dt,
        output_mode=args.model_output,
    )

    # Traces
    trace_len = 1500
    truth_trace_x1x2: List[Tuple[float, float]] = []
    truth_trace_x2x3: List[Tuple[float, float]] = []
    truth_trace_x1x3: List[Tuple[float, float]] = []

    net_trace_x1x2: List[Tuple[float, float]] = []
    net_trace_x2x3: List[Tuple[float, float]] = []
    net_trace_x1x3: List[Tuple[float, float]] = []

    mode_compare = args.compare
    running = True

    # Axis ranges equal to bounds
    xrange_x1 = (x_lo[0], x_hi[0])
    xrange_x2 = (x_lo[1], x_hi[1])
    xrange_x3 = (x_lo[2], x_hi[2])

    rects = view.rects()

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
            # Use controller on the model state (like teleop_pendulum does)
            u = controller(x_net)
            u = float(np.clip(u, -args.umax, args.umax))

        # Hotkeys
        if keys[pygame.K_r]:
            x_truth[:] = args.init
            x_net[:] = args.init
            truth_trace_x1x2.clear(); truth_trace_x2x3.clear(); truth_trace_x1x3.clear()
            net_trace_x1x2.clear(); net_trace_x2x3.clear(); net_trace_x1x3.clear()
            pygame.time.wait(150)
        if keys[pygame.K_m]:
            mode_compare = not mode_compare
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
        x_truth_next = euler_step_third_order(x_truth, u, e1, e2, e3, e4, args.dt)
        try:
            x_net_next = model.step(x_net, u)
        except Exception:
            x_net_next = x_net.copy()

        if args.clamp_bounds:
            x_truth_next = np.maximum(x_lo, np.minimum(x_hi, x_truth_next))
            x_net_next = np.maximum(x_lo, np.minimum(x_hi, x_net_next))

        x_truth = x_truth_next
        x_net = x_net_next

        # Update traces
        truth_trace_x1x2.append((x_truth[0], x_truth[1]))
        truth_trace_x2x3.append((x_truth[1], x_truth[2]))
        truth_trace_x1x3.append((x_truth[0], x_truth[2]))

        net_trace_x1x2.append((x_net[0], x_net[1]))
        net_trace_x2x3.append((x_net[1], x_net[2]))
        net_trace_x1x3.append((x_net[0], x_net[2]))

        for lst in [truth_trace_x1x2, truth_trace_x2x3, truth_trace_x1x3,
                    net_trace_x1x2, net_trace_x2x3, net_trace_x1x3]:
            if len(lst) > trace_len:
                lst.pop(0)

        # Draw
        screen.fill(view.bg)

        # Axes and traces for each projection
        ranges = [ (xrange_x1, xrange_x2, "x1", "x2", truth_trace_x1x2, net_trace_x1x2),
                   (xrange_x2, xrange_x3, "x2", "x3", truth_trace_x2x3, net_trace_x2x3),
                   (xrange_x1, xrange_x3, "x1", "x3", truth_trace_x1x3, net_trace_x1x3) ]

        for rect, (xr, yr, xl, yl, t_trace, n_trace) in zip(rects, ranges):
            draw_axes(screen, rect, xr, yr, view, xl, yl, font)
            if mode_compare:
                draw_trace(screen, rect, t_trace, xr, yr, view.truth_color, width=2)
            draw_trace(screen, rect, n_trace, xr, yr, view.net_color if mode_compare else view.same_color, width=3 if not mode_compare else 2)
            # Current points
            if mode_compare and len(t_trace) > 0:
                pygame.draw.circle(screen, view.truth_color, world_to_screen(t_trace[-1], rect, xr, yr), 5)
            if len(n_trace) > 0:
                pygame.draw.circle(screen, view.net_color if mode_compare else view.same_color, world_to_screen(n_trace[-1], rect, xr, yr), 6)

        # HUD
        err = float(np.linalg.norm(x_net - x_truth)) if mode_compare else 0.0
        lines = [
            f"Controller: {args.controller}   u={u:+.3f}",
            f"Mode: {'COMPARE' if mode_compare else 'MODEL ONLY'}   Clamp: {'ON' if args.clamp_bounds else 'OFF'}",
            f"x_truth=({x_truth[0]:+.3f}, {x_truth[1]:+.3f}, {x_truth[2]:+.3f})" + (f"  ‖Δx‖={err:.3e}" if mode_compare else ""),
            f"x_model=({x_net[0]:+.3f}, {x_net[1]:+.3f}, {x_net[2]:+.3f})",
            "",
            "Keys: [A/D] u±, [X] zero, [R] reset, [M] compare, [C] clamp, [T/G] trace±",
        ]

        y = view.H - 120
        for s in lines:
            txt = font.render(s, True, view.text)
            screen.blit(txt, (20, y))
            y += 18

        pygame.display.flip()
        clock.tick(int(1.0 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
