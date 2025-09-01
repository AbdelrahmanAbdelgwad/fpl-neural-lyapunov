# teleop_unicycle.py
# Minimal keyboard teleop to compare analytic unicycle vs. learned forward model.
# State: x = [x, y, theta]; Control: u = [v, omega]
# Analytic dynamics per your point_navigation.Unicycle (x_dot = [v cosθ, v sinθ, ω]).
# References:
#   - Unicycle state/control & dynamics: point_navigation.py (dynamics)  [x,y,θ], [v,ω]
#   - Your forward model expects concat([x,u]) -> x_next by default.

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import pygame
import torch

# -------------- Helpers --------------


def wrap_angle(th: float) -> float:
    # wrap to [-pi, pi]
    while th <= -math.pi:
        th += 2 * math.pi
    while th > math.pi:
        th -= 2 * math.pi
    return th


def clamp_state(x, bounds_lo, bounds_hi):
    # x = [x, y, theta]
    x[0] = max(bounds_lo[0], min(bounds_hi[0], x[0]))
    x[1] = max(bounds_lo[1], min(bounds_hi[1], x[1]))
    # For theta we wrap, then (optionally) clamp if desired. Here we just wrap.
    x[2] = wrap_angle(x[2])
    return x


def euler_step_unicycle(x, u, dt):
    # x_next = x + f(x,u)*dt with f from analytic kinematics.
    v, w = u
    x_next = np.array(
        [
            x[0] + v * math.cos(x[2]) * dt,
            x[1] + v * math.sin(x[2]) * dt,
            wrap_angle(x[2] + w * dt),
        ],
        dtype=float,
    )
    return x_next


# -------------- Network wrapper --------------


class ForwardModel:
    def __init__(
        self, model_path, device="cpu", outputs_delta=False, dtype=torch.float32
    ):
        self.outputs_delta = outputs_delta
        self.device = torch.device(device)
        if model_path is None or (
            isinstance(model_path, str) and model_path.lower() == "none"
        ):
            self.model = None
        else:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
        self.dtype = dtype
        self.x_eq = np.array([0.0, 0.0, 0.0])
        self.u_eq = np.array([0.0, 0.0])
        xeu = torch.tensor(np.concatenate([self.x_eq, self.u_eq]), dtype=torch.float64)
        with torch.no_grad():
            self.phi_eq = self.model(xeu)  # cache φ([x*,u*])

    def step(self, x, u):
        """
        x: np.array(3,), u: np.array(2,)
        returns np.array(3,)
        """
        xin = torch.tensor(np.concatenate([x, u]), dtype=torch.float64)
        with torch.no_grad():
            x_next_pred = (
                self.model(xin)
                - self.phi_eq
                + torch.tensor(self.x_eq, dtype=torch.float64)
            )
        x_next_pred[2] = wrap_angle(x_next_pred[2].item())
        return x_next_pred.cpu().numpy()


# -------------- Rendering --------------


@dataclass
class View:
    W: int = 900
    H: int = 900
    world_half_extent: float = 1.2  # meters shown half-span (covers ±1.2 m)
    bg_color = (245, 245, 245)
    grid_color = (220, 220, 220)
    truth_color = (50, 50, 50)  # analytic
    net_color = (20, 100, 220)  # network
    same_color = (0, 160, 0)
    text_color = (0, 0, 0)

    def world_to_screen(self, p):
        # p: [x, y] in meters, origin in middle, +x right, +y up
        sx = int((p[0] / self.world_half_extent) * (self.W / 2) + self.W / 2)
        sy = int(self.H / 2 - (p[1] / self.world_half_extent) * (self.H / 2))
        return sx, sy


# -------------- Main app --------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to torch model (φ: [x;u]->x_next).",
    )
    parser.add_argument(
        "--model-outputs-delta",
        action="store_true",
        help="Set if your model outputs Δx instead of x_next.",
    )
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument(
        "--init",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Initial [x,y,theta].",
    )
    parser.add_argument("--vmax", type=float, default=10.0)
    parser.add_argument("--wmax", type=float, default=3.1416)
    parser.add_argument(
        "--clamp-bounds",
        action="store_true",
        help="Clamp state to x∈[-1,1]^2 and wrap θ.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Start in compare mode (analytic & network).",
    )
    args = parser.parse_args()

    pygame.init()
    clock = pygame.time.Clock()
    view = View()
    screen = pygame.display.set_mode((view.W, view.H))
    pygame.display.set_caption("Unicycle Tele-Op (Analytic vs Network)")

    font = pygame.font.SysFont("consolas", 18)

    # Controls
    v, w = 0.0, 0.0
    dv = 0.1  # per key press
    dw = 0.05

    # States
    x_truth = np.array(args.init, dtype=float)
    x_net = np.array(args.init, dtype=float)

    # Bounds
    x_lo = np.array([-1.0, -1.0, -math.pi], dtype=float)
    x_hi = np.array([1.0, 1.0, math.pi], dtype=float)

    # Model
    model = ForwardModel(
        args.model, outputs_delta=args.model_outputs_delta, dtype=torch.float32
    )

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
        if keys[pygame.K_w]:
            v = min(args.vmax, v + dv)
        if keys[pygame.K_s]:
            v = max(-args.vmax, v - dv)
        if keys[pygame.K_a]:
            w = min(args.wmax, w + dw)
        if keys[pygame.K_d]:
            w = max(-args.wmax, w - dw)
        if keys[pygame.K_x]:
            v, w = 0.0, 0.0
        if keys[pygame.K_r]:
            x_truth[:] = args.init
            x_net[:] = args.init
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

        u = np.array([v, w], dtype=float)

        # ------------- Step -------------
        # Analytic
        x_truth_next = euler_step_unicycle(x_truth, u, args.dt)

        # Network
        try:
            # x_net_next = model.step(x_net if not mode_compare else x_truth, u)
            x_net_next = model.step(x_net, u)
            # NOTE: In compare mode we feed the *same* state to both, so step-by-step deltas are directly comparable.
        except Exception as e:
            x_net_next = x_net.copy()  # hold if model missing
            # You can print once if needed:
            # print("Model step error:", e)

        if args.clamp_bounds:
            x_truth_next = clamp_state(x_truth_next, x_lo, x_hi)
            x_net_next = clamp_state(x_net_next, x_lo, x_hi)

        # Commit
        x_truth = x_truth_next
        x_net = x_net_next

        # Traces
        truth_trace.append((x_truth[0], x_truth[1]))
        net_trace.append((x_net[0], x_net[1]))
        if len(truth_trace) > trace_len:
            truth_trace.pop(0)
        if len(net_trace) > trace_len:
            net_trace.pop(0)

        # ------------- Draw -------------
        screen.fill(view.bg_color)

        # grid
        for t in np.linspace(-view.world_half_extent, view.world_half_extent, 13):
            sx1, sy1 = view.world_to_screen((t, -view.world_half_extent))
            sx2, sy2 = view.world_to_screen((t, view.world_half_extent))
            pygame.draw.line(screen, view.grid_color, (sx1, sy1), (sx2, sy2), 1)
            sx1, sy1 = view.world_to_screen((-view.world_half_extent, t))
            sx2, sy2 = view.world_to_screen((view.world_half_extent, t))
            pygame.draw.line(screen, view.grid_color, (sx1, sy1), (sx2, sy2), 1)

        # axes
        pygame.draw.line(
            screen,
            (180, 180, 180),
            view.world_to_screen((-view.world_half_extent, 0.0)),
            view.world_to_screen((view.world_half_extent, 0.0)),
            2,
        )
        pygame.draw.line(
            screen,
            (180, 180, 180),
            view.world_to_screen((0.0, -view.world_half_extent)),
            view.world_to_screen((0.0, view.world_half_extent)),
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

        # draw robots
        def draw_pose(x, color, radius=8):
            sx, sy = view.world_to_screen((x[0], x[1]))
            pygame.draw.circle(screen, color, (sx, sy), radius, 2)
            hd = (x[0] + 0.15 * math.cos(x[2]), x[1] + 0.15 * math.sin(x[2]))
            pygame.draw.line(screen, color, (sx, sy), view.world_to_screen(hd), 2)

        if mode_compare:
            draw_pose(x_truth, view.truth_color, radius=7)
        draw_pose(x_net, view.net_color if mode_compare else view.same_color, radius=9)

        # HUD text
        err = np.linalg.norm((x_net - x_truth)[:2]) if mode_compare else 0.0
        lines = [
            f"dt={args.dt:.3f}  v={v:+.2f} m/s  w={w:+.2f} rad/s",
            f"mode={'COMPARE (analytic vs net)' if mode_compare else 'NETWORK ONLY'}   clamp={'ON' if args.clamp_bounds else 'OFF'}",
            f"x_truth=({x_truth[0]:+.2f},{x_truth[1]:+.2f},{x_truth[2]:+.2f})",
            f"x_net  =({x_net[0]:+.2f},{x_net[1]:+.2f},{x_net[2]:+.2f})  "
            + (f"‖Δpos‖={err:.3e}" if mode_compare else ""),
            "keys: W/S v±, A/D ω±, X zero, R reset, M mode, C clamp, T/G trace-,trace+, ESC quit",
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
