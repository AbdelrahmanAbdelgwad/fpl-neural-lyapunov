# teleop_path_following_enhanced.py
# Keyboard teleop / visualizer for PATH-FOLLOWING error dynamics
# with controller testing and forward model comparison (parity with Pendulum).
#
# State: q = [dist_e, theta_e]; Control: u = [omega]
# Analytic continuous-time dynamics (error coordinates):
#   d/dt dist_e  = v * sin(theta_e)
#   d/dt theta_e = u - v*cos(theta_e)/(1 - dist_e)
# Equilibrium: q* = [0, 0], u* = v.
#
# Features
# --------
# • Manual teleop (keyboard) OR controller-driven (LQR / NN) control
# • Compare analytic integration vs learned forward model φ
# • Equilibrium subtraction for φ: q_next = φ([q;u]) - φ([q*;u*]) + q*
# • Flexible model output modes for φ with proper Euler integration when needed
# • Saturation, clamping, toggles, and HUD
#
# Example usages
# --------------
# 1) Manual teleop, compare to a learned q_next model, v used as u_eq:
#    python teleop_path_following_enhanced.py \
#       --model data/path_following_forward_model.pt --compare --controller MANUAL
#
# 2) LQR controller (around q*=[0,0], u* = v), no learned model, just analytic:
#    python teleop_path_following_enhanced.py --controller LQR --dt 0.02 --v 6.0
#
# 3) NN controller (ϕ_ctrl: q->u) with equilibrium shift and saturation; compare to φ:
#    python teleop_path_following_enhanced.py \
#       --model data/path_following_forward_model.pt \
#       --controller NN --controller-path data/monotonic_bound10_controller.pt \
#       --compare
#
# Controls (while running)
# ------------------------
#   A / D : decrease / increase omega (in MANUAL mode)
#   X     : zero omega (in MANUAL mode)
#   1/2/3 : switch controller -> 1:MANUAL, 2:LQR, 3:NN
#   R     : reset state to --init
#   M     : toggle compare mode (draw analytic trace too)
#   C     : toggle clamping to bounds
#   T / G : shorten / lengthen trace
#   ESC   : quit

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import numpy as np
import pygame
import torch
import scipy.linalg

# --------------------------- Utilities ---------------------------


def wrap_angle(th: float) -> float:
    """Wrap angle to [-pi, pi]."""
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
    q[1] = wrap_angle(
        q[1]
    )  # keep θ_e wrapped, then softly clamp into box if outside by >π
    if q[1] < bounds_lo[1]:
        q[1] = q[1] + 2 * math.pi
    if q[1] > bounds_hi[1]:
        q[1] = q[1] - 2 * math.pi
    return q


def euler_step_analytic(q: np.ndarray, u: float, v: float, dt: float) -> np.ndarray:
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


# ------------------------ Forward model φ ------------------------

ModelOutputMode = Literal["qnext", "distnext", "thetanext", "qdot"]


class ForwardModelPF:
    """
    Thin wrapper around a Torch model φ.

    Modes
    -----
    • qnext     : model outputs q_{n+1} (2D). We apply equilibrium subtraction.
    • distnext  : model outputs only dist_{n+1} (1D). We compute theta_{n+1}
                  via analytic Euler using current q and u.
    • thetanext : model outputs only theta_{n+1} (1D). We compute dist_{n+1}
                  via analytic Euler using current q and u.
    • qdot      : model outputs q̇ = [dist_dot, theta_dot] (2D). We integrate.

    In all cases where the model outputs 2D next-state, we apply
      q_next = φ([q;u]) - φ([q_eq;u_eq]) + q_eq.
    For 1D-output cases, we apply the same equilibrium subtraction to that
    component.
    """

    def __init__(
        self,
        model_path: Optional[str],
        u_eq: float,
        q_eq=(0.0, 0.0),
        device: str = "cpu",
        dt: float = 0.05,
        v: float = 6.0,
        output_mode: ModelOutputMode = "qnext",
    ):
        self.device = torch.device(device)
        self.q_eq = np.array(q_eq, dtype=float)
        self.u_eq = float(u_eq)
        self.dt = float(dt)
        self.v = float(v)
        self.output_mode: ModelOutputMode = output_mode

        self.model = None
        self.phi_eq = None
        if model_path is not None and not (
            isinstance(model_path, str) and model_path.lower() == "none"
        ):
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            xeu = torch.tensor(
                np.array([self.q_eq[0], self.q_eq[1], self.u_eq]), dtype=torch.float64
            )
            with torch.no_grad():
                self.phi_eq = self.model(xeu)

    def step(self, q: np.ndarray, u: float) -> np.ndarray:
        if self.model is None:
            # No model -> hold state (network trace mirrors last)
            return q.copy()

        xin = torch.tensor(np.array([q[0], q[1], float(u)]), dtype=torch.float64)
        with torch.no_grad():
            pred = self.model(xin)

        # Helper to subtract equilibrium on one component
        def shift1(val_t: torch.Tensor, idx: int) -> float:
            return float((val_t - self.phi_eq[idx]).item() + self.q_eq[idx])

        if pred.ndim == 0:
            pred = pred.view(1)

        if self.output_mode == "qnext":
            # expect 2D output
            if pred.numel() != 2:
                # degrade gracefully: copy analytic step
                return euler_step_analytic(q, u, self.v, self.dt)
            q_next = (
                (pred - self.phi_eq + torch.tensor(self.q_eq, dtype=torch.float64))
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            q_next[1] = wrap_angle(q_next[1])
            return q_next

        elif self.output_mode == "qdot":
            # expect 2D output of derivatives
            if pred.numel() != 2:
                return euler_step_analytic(q, u, self.v, self.dt)
            dist_e_dot = float(pred[0].item())
            theta_e_dot = float(pred[1].item())
            return np.array(
                [
                    q[0] + dist_e_dot * self.dt,
                    wrap_angle(q[1] + theta_e_dot * self.dt),
                ],
                dtype=float,
            )

        elif self.output_mode == "distnext":
            # expect 1D output for dist_{n+1}; theta via analytic Euler
            val = shift1(pred[0], idx=0) if pred.numel() >= 1 else q[0]
            q_analytic = euler_step_analytic(q, u, self.v, self.dt)
            q_analytic[0] = val
            return q_analytic

        elif self.output_mode == "thetanext":
            # expect 1D output for theta_{n+1}; dist via analytic Euler
            val = shift1(pred[0], idx=1) if pred.numel() >= 1 else q[1]
            q_analytic = euler_step_analytic(q, u, self.v, self.dt)
            q_analytic[1] = wrap_angle(val)
            return q_analytic

        else:
            # Fallback
            return euler_step_analytic(q, u, self.v, self.dt)


# --------------------------- Controllers ---------------------------

ControllerMode = Literal["MANUAL", "LQR", "NN"]


class Controller:
    def __init__(
        self,
        mode: ControllerMode,
        v: float,
        u_eq: float,
        u_sat: float,
        nn_path: Optional[str] = None,
        q_eq=(0.0, 0.0),
        Q_diag=(1.0, 1.0),
        R_scalar=1.0,
        device: str = "cpu",
    ):
        self.mode: ControllerMode = mode
        self.v = float(v)
        self.u_eq = float(u_eq)
        self.u_sat = abs(float(u_sat))
        self.q_eq = np.array(q_eq, dtype=float)
        self.device = torch.device(device)
        self.K = self._compute_lqr_gain(Q_diag, R_scalar)
        self.nn = None
        self.nn_eq = None
        if mode == "NN" and nn_path:
            self.nn = torch.load(nn_path, map_location=self.device)
            self.nn.eval()
            qe = torch.tensor(self.q_eq, dtype=torch.float64)
            with torch.no_grad():
                self.nn_eq = self.nn(qe)

        # For MANUAL, current u is driven by keyboard and kept here.
        self.manual_u: float = self.u_eq

    def _compute_lqr_gain(self, Q_diag, R_scalar):
        # Linearize analytic error dynamics at q=[0,0]
        v = self.v
        A = np.array([[0.0, v], [-v, 0.0]], dtype=float)
        B = np.array([[0.0], [1.0]], dtype=float)
        Q = np.diag(np.array(Q_diag, dtype=float))
        R = np.array([[float(R_scalar)]], dtype=float)
        S = scipy.linalg.solve_continuous_are(A, B, Q, R)
        K = -np.linalg.solve(R, B.T @ S)  # shape (1,2)
        return K

    def set_mode(self, mode: ControllerMode):
        self.mode = mode

    def set_manual_u(self, u: float):
        self.manual_u = float(u)

    def compute_u(self, q: np.ndarray) -> float:
        if self.mode == "MANUAL":
            u = self.manual_u
        elif self.mode == "LQR":
            dq = (q - self.q_eq).reshape(2)
            u = float(self.K @ dq) + self.u_eq
        elif self.mode == "NN":
            x = torch.tensor(q, dtype=torch.float64)
            with torch.no_grad():
                u = float((self.nn(x) - self.nn_eq).item() + self.u_eq)
        else:
            u = self.u_eq
        # Saturate
        u = max(-self.u_sat, min(self.u_sat, u))
        return u


# --------------------------- Visualization ---------------------------


@dataclass
class View:
    W: int = 900
    H: int = 900
    # We'll visualize in (dist_e, theta_e) space. X-axis = dist_e (m), Y = theta_e (rad).
    dist_half_extent: float = 1.2  # meters on ±X
    theta_half_extent: float = math.pi  # radians on ±Y

    bg_color = (245, 245, 245)
    grid_color = (220, 220, 220)
    truth_color = (250, 100, 100)  # analytic
    net_color = (20, 100, 220)  # network
    same_color = (0, 160, 0)
    text_color = (0, 0, 0)

    def world_to_screen(self, p: Tuple[float, float]) -> Tuple[int, int]:
        # p: (dist_e, theta_e)
        sx = int((p[0] / self.dist_half_extent) * (self.W / 2) + self.W / 2)
        sy = int(self.H / 2 - (p[1] / self.theta_half_extent) * (self.H / 2))
        return sx, sy


# ------------------------------ Main ------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Path-following teleop (controllers + model compare)"
    )

    # Simulation & plant
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    parser.add_argument(
        "--v", type=float, default=6.0, help="Constant forward speed v (m/s)."
    )
    parser.add_argument(
        "--init",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Initial [dist_e, theta_e].",
    )

    # Forward model φ([q;u]) options
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to torch model for q_next or qdot prediction.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="qnext",
        choices=["qnext", "distnext", "thetanext", "qdot"],
        help="What the model predicts (if provided).",
    )
    parser.add_argument(
        "--u-eq",
        type=float,
        default=None,
        help="Override u_eq used in φ([q*;u*]). If omitted and --use-v-as-u-eq is set, u_eq=v; else 0.0.",
    )
    parser.add_argument(
        "--use-v-as-u-eq",
        action="store_true",
        default=True,
        help="If set (default), use u_eq=v unless --u-eq is provided.",
    )

    # Controller options
    parser.add_argument(
        "--controller",
        type=str,
        default="MANUAL",
        choices=["MANUAL", "LQR", "NN"],
        help="Controller mode.",
    )
    parser.add_argument(
        "--controller-path",
        type=str,
        default=None,
        help="Path to a Torch NN controller (ϕ_ctrl: q->u).",
    )
    parser.add_argument(
        "--controller-sat",
        type=float,
        default=10.0,
        help="Absolute saturation for control |u|.",
    )
    parser.add_argument(
        "--lqr-q",
        type=float,
        nargs=2,
        default=[1.0, 1.0],
        help="Diagonal Q entries for LQR.",
    )
    parser.add_argument("--lqr-r", type=float, default=1.0, help="Scalar R for LQR.")

    # UI toggles
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

    args = parser.parse_args()

    # Equilibrium input used in φ eq-subtraction and in controller equilibrium
    if args.u_eq is not None:
        u_eq = float(args.u_eq)
    else:
        u_eq = args.v if args.use_v_as_u_eq else 0.0

    pygame.init()
    clock = pygame.time.Clock()
    view = View()
    screen = pygame.display.set_mode((view.W, view.H))
    pygame.display.set_caption("Path-Following Tele-Op (Controllers + Model Compare)")
    font = pygame.font.SysFont("consolas", 18)

    # States
    q_truth = np.array(args.init, dtype=float)
    q_net = np.array(args.init, dtype=float)

    # Bounds (for optional clamping)
    q_lo = np.array([-0.8, -0.8], dtype=float)
    q_hi = np.array([+0.8, +0.8], dtype=float)

    # Forward model
    fwd = ForwardModelPF(
        model_path=args.model,
        u_eq=u_eq,
        q_eq=(0.0, 0.0),
        device="cpu",
        dt=args.dt,
        v=args.v,
        output_mode=args.model_output,  # handles 1D/2D + integration as needed
    )

    # Controller
    ctrl = Controller(
        mode=args.controller,
        v=args.v,
        u_eq=u_eq,
        u_sat=args.controller_sat,
        nn_path=args.controller_path,
        q_eq=(0.0, 0.0),
        Q_diag=tuple(args.lqr_q),
        R_scalar=args.lqr_r,
        device="cpu",
    )

    # Manual control value (used only in MANUAL mode)
    u_manual = u_eq
    ctrl.set_manual_u(u_manual)

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

        # Controller switching
        if keys[pygame.K_1]:
            ctrl.set_mode("MANUAL")
            pygame.time.wait(150)
        if keys[pygame.K_2]:
            ctrl.set_mode("LQR")
            pygame.time.wait(150)
        if keys[pygame.K_3]:
            ctrl.set_mode("NN")
            pygame.time.wait(150)

        # Manual inputs (only effective in MANUAL mode)
        if ctrl.mode == "MANUAL":
            if keys[pygame.K_a]:
                u_manual = max(-args.controller_sat, u_manual - 0.05)
            if keys[pygame.K_d]:
                u_manual = min(args.controller_sat, u_manual + 0.05)
            if keys[pygame.K_x]:
                u_manual = 0.0
            ctrl.set_manual_u(u_manual)

        if keys[pygame.K_r]:
            q_truth[:] = args.init
            q_net[:] = args.init
            truth_trace.clear()
            net_trace.clear()
            u_manual = u_eq
            ctrl.set_manual_u(u_manual)
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

        # ------------- Control -------------
        u = ctrl.compute_u(q_truth)  # apply the same u to both branches

        # ------------- Step -------------
        # Analytic
        q_truth_next = euler_step_analytic(q_truth, u, args.v, args.dt)

        # Network
        try:
            q_net_next = fwd.step(q_net, u)
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
            f"dt={args.dt:.3f}  v={args.v:.2f} m/s   u_eq={u_eq:+.2f}   u={u:+.2f} (sat±{args.controller_sat:.1f})",
            f"controller={ctrl.mode}   model={args.model_output}{' (none)' if fwd.model is None else ''}   mode={'COMPARE' if mode_compare else 'NETWORK ONLY'}   clamp={'ON' if args.clamp_bounds else 'OFF'}",
            f"q_truth=({q_truth[0]:+.3f}, {q_truth[1]:+.3f})",
            f"q_net  =({q_net[0]:+.3f}, {q_net[1]:+.3f})  "
            + (f"‖Δq‖={err:.3e}" if mode_compare else ""),
            "keys: 1/2/3=MAN/LQR/NN,  A/D u±, X zero u, R reset, M mode, C clamp, T/G trace-,trace+, ESC quit",
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
