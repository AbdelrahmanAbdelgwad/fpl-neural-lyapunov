# teleop_path_following.py
# Keyboard teleop / visualizer for PATH-FOLLOWING error dynamics.
# State: q = [dist_e, theta_e]; Control: u = [omega]
# Analytic dynamics (per your Path_Following model):
#   d/dt dist_e  = v * sin(theta_e)
#   d/dt theta_e = u - v*cos(theta_e)/(1 - dist_e)
#
# This script lets you (1) drive u from the keyboard, (2) compare analytic
# integration vs a learned forward model φ: [q;u] -> q_next.
#
# Notes on equilibrium subtraction (φ([q*,u*])):
# In some of your training code, the forward system uses q*=[0,0] and
# u* set to the constant speed v (u* = v). If your saved model did this,
# pass --use-v-as-u-eq (default ON) or explicitly --u-eq 6.0 (or your v).
# If your model used u*=0, pass --no-use-v-as-u-eq or --u-eq 0.0.
#
# Controls:
#   A / D : decrease / increase omega
#   X     : zero omega
#   R     : reset state to --init
#   M     : toggle compare mode (draw analytic trace too)
#   C     : toggle clamping to bounds
#   T / G : shorten / lengthen trace
#   ESC   : quit
#
# Example:
#   python teleop_path_following.py --model path/to/model.pt --dt 0.02 --v 6.0
#
# Requirements: pygame, torch, numpy

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pygame
import torch


def wrap_angle(th: float) -> float:
    # wrap to [-pi, pi]
    while th <= -math.pi:
        th += 2 * math.pi
    while th > math.pi:
        th -= 2 * math.pi
    return th


def clamp_state(
    q: np.ndarray, bounds_lo: np.ndarray, bounds_hi: np.ndarray
) -> np.ndarray:
    # q = [dist_e, theta_e]
    q[0] = max(bounds_lo[0], min(bounds_hi[0], q[0]))
    q[1] = wrap_angle(q[1])  # keep θ_e wrapped
    if q[1] < bounds_lo[1]:
        q[1] = q[1] + 2 * math.pi
    if q[1] > bounds_hi[1]:
        q[1] = q[1] - 2 * math.pi
    return q


def euler_step_path_following(
    q: np.ndarray, u: float, v: float, dt: float
) -> np.ndarray:
    """
    One Euler step of the analytic path-following error dynamics.
    q = [dist_e, theta_e], u = omega (float), v = forward speed (const).
    """
    dist_e, theta_e = float(q[0]), float(q[1])

    # Avoid singularity at (1 - dist_e) ~ 0
    denom = 1.0 - dist_e
    if abs(denom) < 1e-5:
        denom = 1e-5 if denom >= 0 else -1e-5

    dist_e_dot = v * math.sin(theta_e)
    theta_e_dot = float(u) - (v * math.cos(theta_e) / denom)

    q_next = np.array(
        [
            dist_e + dist_e_dot * dt,
            wrap_angle(theta_e + theta_e_dot * dt),
        ],
        dtype=float,
    )
    return q_next


class ForwardModelPF:
    """
    Thin wrapper around a Torch model φ: [q;u] -> q_next.
    Applies equilibrium subtraction if requested:
       q_next = φ([q;u]) - φ([q_eq;u_eq]) + q_eq
    """

    def __init__(
        self, model_path: Optional[str], u_eq: float, q_eq=(0.0, 0.0), device="cpu"
    ):

        self.device = torch.device(device)
        self.q_eq = np.array(q_eq, dtype=float)
        self.u_eq = float(u_eq)
        if model_path is None or (
            isinstance(model_path, str) and model_path.lower() == "none"
        ):
            self.model = None
            self.phi_eq = None
            self.tdtype = torch.float32
        else:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            xeu = torch.tensor(
                np.array([self.q_eq[0], self.q_eq[1], self.u_eq]), dtype=torch.float64
            )
            with torch.no_grad():
                self.phi_eq = self.model(xeu)

    def step(self, q: np.ndarray, u: float) -> np.ndarray:
        if self.model is None:
            return q.copy()
        xin = torch.tensor(np.array([q[0], q[1], float(u)]), dtype=torch.float64)
        with torch.no_grad():
            q_next_pred = (
                self.model(xin)
                - self.phi_eq
                + torch.tensor(self.q_eq, dtype=torch.float64)
            )
        q_next = q_next_pred.detach().cpu().numpy().astype(float)
        q_next[1] = wrap_angle(q_next[1])
        return q_next


@dataclass
class View:
    W: int = 900
    H: int = 900
    # We'll visualize in (dist_e, theta_e) space. X-axis = dist_e (m), Y = theta_e (rad).
    dist_half_extent: float = 1.2  # meters on ±X
    theta_half_extent: float = math.pi  # radians on ±Y

    bg_color = (245, 245, 245)
    grid_color = (220, 220, 220)
    truth_color = (50, 50, 50)  # analytic
    net_color = (20, 100, 220)  # network
    same_color = (0, 160, 0)
    text_color = (0, 0, 0)

    def world_to_screen(self, p: Tuple[float, float]) -> Tuple[int, int]:
        # p: (dist_e, theta_e)
        sx = int((p[0] / self.dist_half_extent) * (self.W / 2) + self.W / 2)
        sy = int(self.H / 2 - (p[1] / self.theta_half_extent) * (self.H / 2))
        return sx, sy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to torch model (φ: [dist_e, theta_e, u]->q_next).",
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Simulation timestep.")
    parser.add_argument(
        "--init",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Initial [dist_e, theta_e].",
    )
    parser.add_argument(
        "--v", type=float, default=6.0, help="Constant forward speed v (m/s)."
    )
    parser.add_argument("--umax", type=float, default=6.0, help="Max |omega| (rad/s).")
    parser.add_argument(
        "--du", type=float, default=0.05, help="Omega increment per key press."
    )
    parser.add_argument(
        "--clamp-bounds",
        action="store_true",
        help="Clamp q to dist_e∈[-0.9,0.9] and wrap θ_e∈[-π,π].",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Start in compare mode (analytic & network).",
    )

    # φ equilibrium options
    parser.add_argument(
        "--u-eq",
        type=float,
        default=None,
        help="Override u_eq used in φ([q*,u*]). If omitted and --use-v-as-u-eq is set, u_eq=v; else 0.0.",
    )
    parser.add_argument(
        "--use-v-as-u-eq",
        action="store_true",
        default=True,
        help="If set (default), use u_eq=v unless --u-eq is provided.",
    )

    args = parser.parse_args()

    # Determine u_eq
    if args.u_eq is not None:
        u_eq = float(args.u_eq)
    else:
        u_eq = args.v if args.use_v_as_u_eq else 0.0

    pygame.init()
    clock = pygame.time.Clock()
    view = View()
    screen = pygame.display.set_mode((view.W, view.H))
    pygame.display.set_caption("Path-Following Tele-Op (Analytic vs Network)")

    font = pygame.font.SysFont("consolas", 18)

    # Control (omega)
    u = 0.0

    # States
    q_truth = np.array(args.init, dtype=float)  # [dist_e, theta_e]
    q_net = np.array(args.init, dtype=float)

    # Bounds
    q_lo = np.array([-0.9, -math.pi], dtype=float)
    q_hi = np.array([+0.9, +math.pi], dtype=float)

    # Model
    model = ForwardModelPF(args.model, u_eq=u_eq, q_eq=(0.0, 0.0))

    # Traces
    trace_len = 1500
    truth_trace = []
    net_trace = []

    mode_compare = args.compare  # True: draw both; False: drive network only
    running = True

    while running:
        # ------------- Events -------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # Inputs
        if keys[pygame.K_a]:
            u = max(-args.umax, u - args.du)
        if keys[pygame.K_d]:
            u = min(args.umax, u + args.du)
        if keys[pygame.K_x]:
            u = 0.0
        if keys[pygame.K_r]:
            q_truth[:] = args.init
            q_net[:] = args.init
            truth_trace.clear()
            net_trace.clear()
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

        # ------------- Step -------------
        # Analytic
        q_truth_next = euler_step_path_following(q_truth, u, args.v, args.dt)

        # Network
        try:
            q_net_next = model.step(q_net, u)
        except Exception:
            q_net_next = q_net.copy()  # hold if model missing/broken

        if args.clamp_bounds:
            q_truth_next = clamp_state(q_truth_next, q_lo, q_hi)
            q_net_next = clamp_state(q_net_next, q_lo, q_hi)

        # Commit
        q_truth = q_truth_next
        q_net = q_net_next

        # Traces
        truth_trace.append((q_truth[0], q_truth[1]))
        net_trace.append((q_net[0], q_net[1]))
        if len(truth_trace) > trace_len:
            truth_trace.pop(0)
        if len(net_trace) > trace_len:
            net_trace.pop(0)

        # ------------- Draw -------------
        screen.fill(view.bg_color)

        # grid in (dist_e, theta_e) space
        for t in np.linspace(-view.dist_half_extent, view.dist_half_extent, 13):
            sx1, sy1 = view.world_to_screen((t, -view.theta_half_extent))
            sx2, sy2 = view.world_to_screen((t, view.theta_half_extent))
            pygame.draw.line(screen, view.grid_color, (sx1, sy1), (sx2, sy2), 1)
        for t in np.linspace(-view.theta_half_extent, view.theta_half_extent, 13):
            sx1, sy1 = view.world_to_screen((-view.dist_half_extent, t))
            sx2, sy2 = view.world_to_screen((view.dist_half_extent, t))
            pygame.draw.line(screen, view.grid_color, (sx1, sy1), (sx2, sy2), 1)

        # axes
        pygame.draw.line(
            screen,
            (180, 180, 180),
            view.world_to_screen((-view.dist_half_extent, 0.0)),
            view.world_to_screen((view.dist_half_extent, 0.0)),
            2,
        )
        pygame.draw.line(
            screen,
            (180, 180, 180),
            view.world_to_screen((0.0, -view.theta_half_extent)),
            view.world_to_screen((0.0, view.theta_half_extent)),
            2,
        )

        # traces
        if len(truth_trace) > 1 and mode_compare:
            for i in range(1, len(truth_trace)):
                pygame.draw.line(
                    screen,
                    view.truth_color,
                    view.world_to_screen(truth_trace[i - 1]),
                    view.world_to_screen(truth_trace[i]),
                    2,
                )
        if len(net_trace) > 1:
            for i in range(1, len(net_trace)):
                pygame.draw.line(
                    screen,
                    view.net_color if mode_compare else view.same_color,
                    view.world_to_screen(net_trace[i - 1]),
                    view.world_to_screen(net_trace[i]),
                    3 if not mode_compare else 2,
                )

        # draw current points
        def draw_point(q, color, radius=7):
            sx, sy = view.world_to_screen((q[0], q[1]))
            pygame.draw.circle(screen, color, (sx, sy), radius, 0)

        if mode_compare:
            draw_point(q_truth, view.truth_color, radius=6)
        draw_point(q_net, view.net_color if mode_compare else view.same_color, radius=8)

        # HUD
        err = float(np.linalg.norm(q_net - q_truth)) if mode_compare else 0.0
        lines = [
            f"dt={args.dt:.3f}  v={args.v:.2f} m/s   u=omega={u:+.2f} rad/s   u_eq={u_eq:+.2f}",
            f"mode={'COMPARE (analytic vs net)' if mode_compare else 'NETWORK ONLY'}   clamp={'ON' if args.clamp_bounds else 'OFF'}",
            f"q_truth=({q_truth[0]:+.3f}, {q_truth[1]:+.3f})",
            f"q_net  =({q_net[0]:+.3f}, {q_net[1]:+.3f})  "
            + (f"‖Δq‖={err:.3e}" if mode_compare else ""),
            "keys: A/D u±, X zero u, R reset, M mode, C clamp, T/G trace-,trace+, ESC quit",
        ]
        y = 10
        for s in lines:
            txt = font.render(s, True, view.text_color)
            screen.blit(txt, (10, y))
            y += 22

        pygame.display.flip()
        clock.tick(int(1.0 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
