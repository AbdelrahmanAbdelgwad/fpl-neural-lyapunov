# Enhanced teleop/visualizer for CART-POLE with model + controller options
# State: x = [cart_pos, pole_angle, cart_vel, pole_ang_vel]
# Control: u ∈ R (cart force)
#
# Features (mirrors teleop_third_order.py UX):
# - Keyboard, LQR, or NN controller (u = ψ(x) − ψ(x*) + u*) with saturation
# - Compare analytic dynamics vs learned forward model (φ)
# - Robust model wrapper: supports φ([x;u]→x_next), φ([x;u]→Δx), or φ([θ,θ̇,u]→[ẍ,θ̈])
# - Bounds clamping in either a wide range or the paper-style box (bound_level)
# - Clean 2D cart-pole drawing + traces + HUD
#
# Example usages:
#   # Manual, wide bounds, compare against a learned x_next model
#   python teleop_cartpole_plus.py --model path/to/phi.pt --compare
#
#   # LQR stabilization with |u|≤30 N
#   python teleop_cartpole_plus.py --controller lqr --umax 30
#
#   # NN controller ψ(x→u) plus compare-mode using a model that outputs accelerations
#   python teleop_cartpole_plus.py --controller NN --controller-model path/to/psi.pt \
#       --model path/to/accel_model.pt --model-output accel --compare
#
#   # Paper-style small box like monotonic_train_cart_pole_demo.py
#   python teleop_cartpole_plus.py --paper-bounds --bound-level 50

import argparse
import math
from typing import Optional, Tuple, List

import numpy as np
import pygame
import torch
import torch.nn as nn

# Local dynamics implementation
from cart_pole import Cart_Pole  # provided alongside this script


# ---------------------------- Utils ---------------------------- #

def wrap_angle(th: float) -> float:
    while th <= -math.pi:
        th += 2.0 * math.pi
    while th > math.pi:
        th -= 2.0 * math.pi
    return th


def clamp_state(x: np.ndarray, x_lo: np.ndarray, x_hi: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[0] = float(np.clip(y[0], x_lo[0], x_hi[0]))     # cart position
    y[1] = wrap_angle(y[1])                           # angle
    y[2] = float(np.clip(y[2], x_lo[2], x_hi[2]))     # cart vel
    y[3] = float(np.clip(y[3], x_lo[3], x_hi[3]))     # ang vel
    return y


# ---------------------------- Model wrapper ---------------------------- #

class ForwardModelCartPole:
    """Wrap a Torch model φ for one-step prediction.

    Modes:
      - output="xnext":   φ returns x[n+1]; we apply equilibrium correction y' = y − y_eq + x_eq
      - output="delta":   φ returns Δx;   we set x_next = x + Δx
      - output="accel":   φ returns [ẍ, θ̈] given either [x,θ,ẋ,θ̇,u] or [θ,θ̇,u]; Euler integrate

    Input layout:
      - input="auto" tries [x;u] first (5-dim), then [θ,θ̇,u] (3-dim).
      - input="full" forces [x,θ,ẋ,θ̇,u].
      - input="theta" forces [θ,θ̇,u].
    """

    def __init__(
        self,
        model_path: Optional[str],
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        output: str = "xnext",      # "xnext" | "delta" | "accel"
        input_mode: str = "auto",    # "auto" | "full" | "theta"
        x_eq: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        u_eq: float = 0.0,
        dt: float = 0.01,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.output = output
        self.input_mode = input_mode
        self.x_eq = np.array(x_eq, dtype=float)
        self.u_eq = float(u_eq)
        self.dt = float(dt)

        self.model: Optional[nn.Module] = None
        self.y_eq: Optional[torch.Tensor] = None
        self._uses_theta_only: Optional[bool] = None

        if model_path and (str(model_path).lower() != "none"):
            data = torch.load(model_path, map_location=self.device)
            if isinstance(data, nn.Module):
                self.model = data.eval()
            elif isinstance(data, dict) and "state_dict" in data:
                # Blind-load a generic MLP if saved as a blob of weights
                # (User can adapt this branch if their checkpoints encode layers)
                raise RuntimeError(
                    "Checkpoint is a state_dict without architecture; please save nn.Module directly.")
            else:
                self.model = data  # hope it's a torchscript or similar

        if self.model is not None:
            # Cache equilibrium output for centering
            with torch.no_grad():
                xin_full = torch.tensor(np.r_[self.x_eq, self.u_eq], dtype=self.dtype)
                xin_theta = torch.tensor(np.r_[self.x_eq[1], self.x_eq[3], self.u_eq], dtype=self.dtype)
                self.y_eq = None
                self._uses_theta_only = None
                # Try full first unless forced otherwise
                tried = []
                for mode in ([self.input_mode] if self.input_mode in ("full", "theta") else ["full", "theta"]):
                    try:
                        y = self.model(xin_full if mode == "full" else xin_theta)
                        _ = y.reshape(-1)  # validate
                        self._uses_theta_only = (mode == "theta")
                        self.y_eq = y.detach()
                        break
                    except Exception as ex:
                        tried.append((mode, str(ex)))
                        continue
                if self.y_eq is None:
                    raise RuntimeError(f"Model forward failed for both inputs: {tried}")

    def step(self, x: np.ndarray, u: float) -> np.ndarray:
        if self.model is None:
            return x.copy()
        x = x.astype(float)
        u = float(u)
        with torch.no_grad():
            if self._uses_theta_only:
                xin = torch.tensor(np.array([x[1], x[3], u]), dtype=self.dtype)
            else:
                xin = torch.tensor(np.r_[x, u], dtype=self.dtype)
            y = self.model(xin).reshape(-1)

            if self.output == "accel":
                # Expect y = [ẍ, θ̈]
                if y.numel() != 2:
                    raise RuntimeError(f"Accel mode expects 2 outputs; got {y.numel()}.")
                # Center at equilibrium to enforce ẍ=θ̈=0 at (x*,u*)
                y = y - self.y_eq.reshape(-1)
                x_ddot, th_ddot = float(y[0]), float(y[1])
                x_next = np.array([
                    x[0] + x[2] * self.dt,
                    wrap_angle(x[1] + x[3] * self.dt),
                    x[2] + x_ddot * self.dt,
                    x[3] + th_ddot * self.dt,
                ])
            elif self.output == "delta":
                if y.numel() != 4:
                    raise RuntimeError(f"Delta mode expects 4 outputs; got {y.numel()}.")
                x_next = x + y.cpu().numpy()
                x_next[1] = wrap_angle(x_next[1])
            else:  # xnext
                if y.numel() != 4:
                    raise RuntimeError(f"xnext mode expects 4 outputs; got {y.numel()}.")
                # Enforce equilibrium exactly
                y_corr = y - self.y_eq.reshape(-1) + torch.tensor(self.x_eq, dtype=self.dtype)
                x_next = y_corr.cpu().numpy()
                x_next[1] = wrap_angle(x_next[1])
        return x_next


# ---------------------------- Controllers ---------------------------- #

def make_lqr_controller(plant: Cart_Pole, Q: np.ndarray, R: np.ndarray,
                        x_eq: np.ndarray, u_eq: np.ndarray):
    K, S = plant.lqr_control(Q, R, x_eq, u_eq)

    def ufun(x: np.ndarray) -> float:
        return float((-K @ (x - x_eq)).item() + u_eq.item())

    return ufun, K, S


def make_nn_controller(model_path: str, x_eq=(0, 0, 0, 0), u_eq=0.0,
                        device="cpu", dtype=torch.float64):
    data = torch.load(model_path, map_location=device)
    if isinstance(data, nn.Module):
        net = data.eval()
    else:
        net = data  # allow torchscript
    x_eq_t = torch.tensor(x_eq, dtype=dtype).unsqueeze(0)
    with torch.no_grad():
        phi_eq = net(x_eq_t).reshape(-1)

    def ufun(x: np.ndarray) -> float:
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=dtype).unsqueeze(0)
            u_hat = net(x_t).reshape(-1) - phi_eq + torch.tensor([u_eq], dtype=dtype)
        return float(u_hat[0].item())

    return ufun


# ---------------------------- Rendering ---------------------------- #

class CartPoleRenderer:
    def __init__(self, W=1000, H=600, world_width=5.0):
        self.W, self.H = W, H
        self.world_width = world_width
        self.scale = self.W / self.world_width
        self.bg = (245, 245, 245)
        self.grid = (220, 220, 220)
        self.track = (80, 80, 80)
        self.truth_cart = (50, 80, 220)
        self.truth_pole = (80, 120, 255)
        self.net_cart = (220, 80, 80)
        self.net_pole = (255, 120, 120)
        self.single_cart = (0, 160, 0)
        self.single_pole = (0, 200, 0)
        self.text = (0, 0, 0)

    def world_to_screen(self, x, y):
        sx = int(x * self.scale + self.W / 2)
        sy = int(self.H / 2 - y * self.scale)
        return sx, sy

    def draw_grid(self, screen):
        for x in np.arange(-self.world_width / 2, self.world_width / 2 + 1e-6, 0.5):
            sx, _ = self.world_to_screen(x, 0)
            pygame.draw.line(screen, self.grid, (sx, 0), (sx, self.H), 1)
        pygame.draw.line(screen, self.track, (0, self.H // 2), (self.W, self.H // 2), 3)

    def draw_cartpole(self, screen, state, cart_color, pole_color, *, offset_y=0.0, alpha=255):
        x_cart, theta = float(state[0]), float(state[1])
        cart_w, cart_h = 0.40, 0.20
        pole_L = 1.0
        cx, cy = self.world_to_screen(x_cart, offset_y)
        cw, ch = int(cart_w * self.scale), int(cart_h * self.scale)
        # Cart body
        surf = pygame.Surface((cw, ch), pygame.SRCALPHA)
        surf.fill((*cart_color, alpha))
        screen.blit(surf, (cx - cw // 2, cy - ch // 2))
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(cx - cw // 2, cy - ch // 2, cw, ch), 2)
        # Wheels
        r = max(2, int(0.05 * self.scale))
        wheel = tuple(int(0.4 * c) for c in cart_color)
        pygame.draw.circle(screen, wheel, (cx - cw // 3, cy + ch // 2), r)
        pygame.draw.circle(screen, wheel, (cx + cw // 3, cy + ch // 2), r)
        # Pole
        px = x_cart + pole_L * math.sin(theta)
        py = offset_y + pole_L * math.cos(theta)
        p_end = self.world_to_screen(px, py)
        pygame.draw.line(screen, pole_color, (cx, cy), p_end, 8)
        pygame.draw.circle(screen, tuple(int(0.8 * c) for c in pole_color), p_end, 10)
        pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 5)


# ---------------------------- App ---------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Cart-Pole teleop with model & controller options")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--umax", type=float, default=30.0)
    ap.add_argument("--du", type=float, default=20.0, help="Keyboard force increment")

    # Controller
    ap.add_argument("--controller", choices=["keyboard", "lqr", "NN", "zero"], default="keyboard")
    ap.add_argument("--lqr-Q", type=str, default="1,10,0.1,1", help="Diag entries for LQR Q")
    ap.add_argument("--lqr-R", type=float, default=0.01)
    ap.add_argument("--controller-model", type=str, default="none", help="ψ: x→u checkpoint")

    # Forward model
    ap.add_argument("--model", type=str, default="none", help="φ checkpoint path")
    ap.add_argument("--model-output", choices=["xnext", "delta", "accel"], default="xnext")
    ap.add_argument("--model-input", choices=["auto", "full", "theta"], default="auto")

    # Init + bounds
    ap.add_argument("--x-init", type=float, nargs=4, default=[0.0, 0.1, 0.0, 0.0])
    ap.add_argument("--paper-bounds", action="store_true", help="Use paper-style box via bound_level/100")
    ap.add_argument("--bound-level", type=int, default=50, help="For paper-bounds: scales the box (divided by 100)")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--clamp-bounds", action="store_true")

    args = ap.parse_args()

    pygame.init()
    clock = pygame.time.Clock()
    rnd = CartPoleRenderer()
    screen = pygame.display.set_mode((rnd.W, rnd.H))
    pygame.display.set_caption("Cart-Pole Tele-Op (Truth vs Model)")
    font = pygame.font.SysFont("consolas", 16)

    # Plant and model
    plant = Cart_Pole(torch.float64)
    model = ForwardModelCartPole(
        None if args.model.lower() == "none" else args.model,
        output=args.model_output,
        input_mode=args.model_input,
        dt=args.dt,
    )

    # Controller setup
    x_eq = np.zeros(4)
    u_eq = np.zeros(1)
    controller = None
    K = None
    if args.controller == "lqr":
        Qd = np.diag([float(v) for v in args.lqr_Q.split(",")])
        Rd = np.array([[args.lqr_R]])
        controller, K, _ = make_lqr_controller(plant, Qd, Rd, x_eq, u_eq)
    elif args.controller == "NN":
        if args.controller_model.lower() == "none":
            raise SystemExit("--controller NN requires --controller-model path")
        controller = make_nn_controller(args.controller_model, x_eq, float(u_eq[0]))
    elif args.controller == "zero":
        controller = lambda x: 0.0

    # Initial state and bounds
    x_truth = np.array(args.x_init, dtype=float)
    x_model = x_truth.copy()

    if args.paper_bounds:
        b = args.bound_level / 100.0
        # Matches monotonic_train_cart_pole_demo: x ∈ [±b, ±π/6 b, ±b, ±b]
        x_lo = np.array([-b, -math.pi / 6 * b, -b, -b])
        x_hi = np.array([+b, +math.pi / 6 * b, +b, +b])
    else:
        x_lo = np.array([-2.5, -math.pi, -10.0, -10.0])
        x_hi = np.array([+2.5, +math.pi, +10.0, +10.0])

    # Traces
    trace_truth: List[Tuple[float, float]] = []
    trace_model: List[Tuple[float, float]] = []
    max_trace = 800

    running, paused = True, False
    compare = args.compare and (model.model is not None)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False
        if keys[pygame.K_SPACE]:
            paused = not paused
            pygame.time.wait(150)
        if keys[pygame.K_r]:
            x_truth[:] = args.x_init
            x_model[:] = args.x_init
            trace_truth.clear(); trace_model.clear()
            pygame.time.wait(150)
        if keys[pygame.K_m] and model.model is not None:
            compare = not compare
            pygame.time.wait(150)
        if keys[pygame.K_c]:
            args.clamp_bounds = not args.clamp_bounds
            pygame.time.wait(150)

        # Control selection
        if args.controller == "keyboard":
            u = 0.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                u = -args.du
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                u = +args.du
            if keys[pygame.K_q]:
                u = -args.umax
            if keys[pygame.K_e]:
                u = +args.umax
            if keys[pygame.K_x] or keys[pygame.K_z]:
                u = 0.0
        else:
            u = controller(x_model)
        u = float(np.clip(u, -args.umax, args.umax))

        if not paused:
            # Integrate truth via plant
            x_truth_next = plant.next_pose(x_truth, np.array([u]), args.dt)
            x_truth = clamp_state(x_truth_next, x_lo, x_hi) if args.clamp_bounds else x_truth_next

            # Integrate model
            if model.model is not None:
                x_model_next = model.step(x_model, u)
                x_model = clamp_state(x_model_next, x_lo, x_hi) if args.clamp_bounds else x_model_next
            else:
                x_model = x_truth.copy()

            # Traces
            trace_truth.append((x_truth[0], x_truth[1]))
            trace_model.append((x_model[0], x_model[1]))
            if len(trace_truth) > max_trace: trace_truth.pop(0)
            if len(trace_model) > max_trace: trace_model.pop(0)

        # --- Render ---
        screen.fill(rnd.bg)
        rnd.draw_grid(screen)

        # Draw traces (cart x only to avoid clutter)
        def draw_trace(tr, color):
            if len(tr) < 2: return
            for i in range(1, len(tr)):
                x1, _ = tr[i-1]
                x2, _ = tr[i]
                p1 = rnd.world_to_screen(x1, 0.0)
                p2 = rnd.world_to_screen(x2, 0.0)
                pygame.draw.line(screen, color, p1, p2, 2)

        if compare:
            draw_trace(trace_truth, (140, 170, 255))
            draw_trace(trace_model, (255, 170, 170))
            rnd.draw_cartpole(screen, x_truth, rnd.truth_cart, rnd.truth_pole, offset_y=-0.05, alpha=220)
            rnd.draw_cartpole(screen, x_model, rnd.net_cart, rnd.net_pole, offset_y=+0.05, alpha=220)
        else:
            draw_trace(trace_model if model.model is not None else trace_truth, (160, 240, 160))
            state = x_model if model.model is not None else x_truth
            rnd.draw_cartpole(screen, state, rnd.single_cart, rnd.single_pole)

        # Force arrow
        if abs(u) > 1e-2:
            ref = x_truth if compare else (x_model if model.model is not None else x_truth)
            cx, cy = rnd.world_to_screen(ref[0], 0.0)
            L = int(abs(u) * 3)
            sgn = 1 if u > 0 else -1
            end = (cx + sgn * L, cy)
            pygame.draw.line(screen, (0, 180, 0), (cx, cy), end, 4)
            pygame.draw.polygon(screen, (0, 180, 0), [end, (end[0]-sgn*8, end[1]-5), (end[0]-sgn*8, end[1]+5)])

        # HUD
        lines = []
        mode = "COMPARE" if compare else ("MODEL" if model.model is not None else "TRUTH")
        lines.append(f"Mode: {mode}  |  Controller: {args.controller}  |  {'PAUSED' if paused else 'RUNNING'}")
        if compare:
            dtheta = wrap_angle(x_model[1]-x_truth[1])
            derr = np.linalg.norm(x_model - x_truth)
            lines.append(f"‖Δstate‖={derr:.3e}   Δθ={math.degrees(abs(dtheta)):.2f}°")
        lines.append(f"State (truth): x={x_truth[0]:+.3f} m, θ={math.degrees(x_truth[1]):+.1f}°, ẋ={x_truth[2]:+.3f}, θ̇={x_truth[3]:+.3f}")
        if model.model is not None:
            lines.append(f"State (model): x={x_model[0]:+.3f} m, θ={math.degrees(x_model[1]):+.1f}°, ẋ={x_model[2]:+.3f}, θ̇={x_model[3]:+.3f}")
        lines.append(f"u={u:+.2f} N   |   Bounds: {'PAPER' if args.paper_bounds else 'WIDE'}   Clamp: {'ON' if args.clamp_bounds else 'OFF'}")
        lines += [
            "Keys:",
            "  A/D or ←/→: ±force   Q/E: max left/right   X/Z: zero",
            "  M: compare   C: clamp   SPACE: pause   R: reset   ESC: quit",
        ]
        y = 10
        for s in lines:
            txt = font.render(s, True, rnd.text)
            screen.blit(txt, (10, y))
            y += 20

        pygame.display.flip()
        clock.tick(int(1.0 / args.dt))

    pygame.quit()


if __name__ == "__main__":
    main()
