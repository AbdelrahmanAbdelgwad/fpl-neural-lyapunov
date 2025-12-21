
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACC Visualization (unified, extended)
Author: Abdelrahman Abdelgawad + ChatGPT assistant

Adds over the previous version:
- Analytical *and* ML distance-to-equilibrium panels (semilog option), consistent IC colors, FPL solid vs No-FPL dashed.
- Verification rectangle + equilibrium overlay on ROA and phase plots.
- ROA: time-to-convergence bins with colorbar + on-plot stats (convergence rate, mean settle time).
- Lyapunov 3D surfaces with floor contours.
- Separate figure files (and a consolidated summary). --no_save to just display.
- Phase diagrams (x1 vs x2) for random ICs: analytical and ML subplots; FPL solid vs No-FPL dashed; colors per IC.
- Control effort scatter (u vs t) matching the phase-diagram selection; markers + connecting lines (solid/dashed).
- TQDM progress bars for long loops.
- GPU batching for ROA + V-grids; analytical ODE for curve-level fidelity.
"""

import os
import sys
import time
import json
import math
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Optional heavy libs
from concurrent.futures import ProcessPoolExecutor, as_completed

# tqdm fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

# Hygiene for big-CPU box
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

# Paper-ish defaults
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- helpers for per-IC coloring and env-specific "effort" ---

def ic_colors(n: int, cmap_name: str = "tab20"):
    """Return n distinct RGBA colors."""
    cmap = plt.get_cmap(cmap_name)
    if n <= cmap.N:
        return [cmap(i) for i in np.linspace(0, 1, n, endpoint=False)]
    # fallback if many ICs
    return [plt.cm.nipy_spectral(i) for i in np.linspace(0, 1, n, endpoint=False)]

def control_effort(env: str, U: np.ndarray, u_eq: torch.Tensor):
    """Quantity to plot on the y-axis for effort."""
    if env == "unicycle":
        return np.abs(U - float(u_eq))   # steering effort about cruise speed
    return np.abs(U)                      # pendulum has u_eq = 0

def sample_ic_list(n, mode, x_lo, x_up, rng):
    """
    Returns a list of np.ndarray initial conditions in [x_lo, x_up].
    2D state expected (pendulum/unicycle).
    """
    x_lo = np.asarray(x_lo, dtype=float); x_up = np.asarray(x_up, dtype=float)
    assert x_lo.shape[0] == 2 and x_up.shape[0] == 2, "IC sampler assumes 2D state."
    if n <= 0: return []
    if mode == "random":
        X = rng.uniform(low=x_lo, high=x_up, size=(n, 2))
        return [X[i] for i in range(n)]
    # grid
    nx = int(np.ceil(np.sqrt(n))); ny = int(np.ceil(n / nx))
    g0 = np.linspace(x_lo[0], x_up[0], nx); g1 = np.linspace(x_lo[1], x_up[1], ny)
    G0, G1 = np.meshgrid(g0, g1); X = np.stack([G0.ravel(), G1.ravel()], axis=1)
    return [X[i] for i in range(n)]

def effort_sum(u: np.ndarray, env: str, u_eq: torch.Tensor) -> float:
    """
    Scalar 'effort' for a trajectory: sum |u|.
    For unicycle, use sum |u - u_eq|.
    """
    if env == "unicycle":
        return float(np.sum(np.abs(u - float(u_eq))))
    return float(np.sum(np.abs(u)))

@torch.no_grad()
def lyap_on_traj(X_np: np.ndarray, lyapunov, x_eq: torch.Tensor, R: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Evaluate V(x) along a trajectory X_np (T×2) using your V definition:
        V(x) = lyapunov(x) - lyapunov(x_eq) + ||R(x-x_eq)||_1
    Uses the lyapunov module's dtype for consistency.
    """
    try:
        lyap_dtype = next(p.dtype for p in lyapunov.parameters())
    except StopIteration:
        lyap_dtype = torch.float32
    x = torch.as_tensor(X_np, dtype=lyap_dtype, device=device)
    xe = x_eq.to(device=device, dtype=lyap_dtype)
    R = R.to(device=device, dtype=lyap_dtype)
    v_net = lyapunov(x) - lyapunov(xe)
    v_l1  = torch.sum(torch.abs(R @ (x - xe).T), dim=0)
    return (v_net.squeeze() + v_l1).detach().cpu().numpy()

def settling_time_from_dist(t: np.ndarray, d: np.ndarray, eps: float) -> float:
    """First time index where distance < eps; returns +inf if never."""
    idx = np.where(d < eps)[0]
    return float(t[idx[0]]) if idx.size else float("inf")




# -----------------------------
# Run specification
# -----------------------------
@dataclass
class RunSpec:
    env: str                  # "pendulum" or "unicycle"
    bound_level: int          # integer bound level
    fpl: bool                 # True for FPL run, False otherwise
    root_dir: Optional[str] = None  # root where /data/... lives
    run_dir: Optional[str] = None    # explicit override dir (if provided)

    def model_base_dir(self) -> str:
        if self.run_dir is not None:
            return os.path.abspath(self.run_dir)
        here = os.path.dirname(os.path.realpath(__file__))
        base_root = self.root_dir if self.root_dir is not None else here
        suffix = "_fpl" if self.fpl else ""
        if self.env == "pendulum":
            return os.path.join(base_root, f"neural_network_lyapunov/examples/pendulum/data/monotonic_bound{self.bound_level}{suffix}")
        elif self.env == "unicycle":
            return os.path.join(base_root, f"neural_network_lyapunov/examples/path_following_unicycle/data/monotonic/monotonic_bound{self.bound_level}{suffix}")
        else:
            raise ValueError(f"Unknown env {self.env}")

    def model_paths(self) -> Dict[str, str]:
        base = self.model_base_dir()
        suffix = "_fpl" if self.fpl else ""
        def p(name):
            return os.path.join(base, f"monotonic_bound{self.bound_level}{suffix}_{name}.pt")
        return {
            "controller": p("controller"),
            "lyapunov": p("lyapunov"),
            "R": p("R"),
        }


# -----------------------------
# Devices & loaders
# -----------------------------
def setup_device(gpu_flag: bool) -> torch.device:
    return torch.device("cuda") if gpu_flag and torch.cuda.is_available() else torch.device("cpu")


def _move_obj_to_device(obj, device):
    import torch.nn as nn
    if isinstance(obj, nn.Module):
        obj.to(device)
    for name, val in vars(obj).items():
        if isinstance(val, torch.Tensor):
            setattr(obj, name, val.to(device))
        elif hasattr(val, "to"):  # nested modules
            try:
                val.to(device)
            except Exception:
                pass


def load_models(spec: RunSpec, device: torch.device):
    paths = spec.model_paths()
    controller = torch.load(paths["controller"], map_location="cpu")
    lyapunov = torch.load(paths["lyapunov"], map_location="cpu")
    R = torch.load(paths["R"], map_location="cpu")
    if isinstance(controller, torch.nn.Module):
        controller = controller.to(device).eval()
    if isinstance(lyapunov, torch.nn.Module):
        lyapunov = lyapunov.to(device).eval()
    if isinstance(R, torch.Tensor):
        R = R.to(device)
    return controller, lyapunov, R


def load_forward_system(env: str, device: torch.device, dt: float):
    import neural_network_lyapunov.utils as utils
    import neural_network_lyapunov.relu_system as relu_system
    here = os.path.dirname(os.path.realpath(__file__))

    if env == "pendulum":
        path = os.path.join(here, "neural_network_lyapunov/examples/pendulum/data/pendulum_second_order_forward_relu2.pt")
        data = torch.load(path, map_location="cpu")
        dynamics_model = utils.setup_relu(
            data["linear_layer_width"], params=None,
            negative_slope=data["negative_slope"], bias=True,
        )
        dynamics_model.load_state_dict(data["state_dict"])
        q_eq = torch.tensor([math.pi], )
        u_eq = torch.tensor([0.0], )
        x_lo = torch.tensor([0, -5.0], )
        x_up = torch.tensor([2 * np.pi,  5.0], )
        u_lo = torch.tensor([-20.0], )
        u_up = torch.tensor([ 20.0], )
        fwd = relu_system.ReLUSecondOrderSystemGivenEquilibrium(
            torch.float64, x_lo, x_up, u_lo, u_up, dynamics_model, q_eq, u_eq, dt
        )
        dynamics_model = dynamics_model.to(device).eval()
        _move_obj_to_device(fwd, device)
        x_eq = torch.tensor([math.pi, 0.0], )
        return dynamics_model, fwd, x_lo, x_up, u_lo.item(), u_up.item(), x_eq, u_eq

    # --- UNICYCLE ---
    elif env == "unicycle":
        import neural_network_lyapunov.examples.path_following_unicycle.path_following as pf
        # Pick a working dtype for the forward model: prefer user-selected DTYPE; cast the model to it.
        path = os.path.join(here, "neural_network_lyapunov/examples/path_following_unicycle/data/preprocess/path_following_unicycle_forward_model.pt")
        dynamics_model = torch.load(path, map_location="cpu").to(device).eval()
        try:
            DTYPE  # from your args.precision setup
        except NameError:
            DTYPE = next((p.dtype for p in dynamics_model.parameters()), torch.float64)
        dynamics_model = dynamics_model.to(DTYPE).to(device).eval()

        # Equilibria, bounds in the SAME dtype
        x_lo = torch.tensor([-0.8, -0.8], dtype=DTYPE)
        x_up = torch.tensor([ 0.8,  0.8], dtype=DTYPE)
        u_lo = torch.tensor([-10.0], dtype=DTYPE)
        u_up = torch.tensor([ 10.0], dtype=DTYPE)
        q_eq = torch.tensor([0.0, 0.0], dtype=DTYPE)       # distance
        u_eq = torch.tensor([6], dtype=DTYPE)
        x_eq_state = torch.tensor([0.0, 0.0], dtype=DTYPE)

        fwd = relu_system.ReLUSystemGivenEquilibrium(
        torch.float64,
        x_lo,
        x_up,
        u_lo,
        u_up,
        dynamics_model,
        q_eq,
        u_eq,
        dt,
    )
        dynamics_model = dynamics_model.to(device).eval()
        _move_obj_to_device(fwd, device)

        return dynamics_model, fwd, x_lo, x_up, u_lo.item(), u_up.item(), x_eq_state, u_eq

    else:
        raise ValueError(f"Unknown env {env}")


# -----------------------------
# Analytical RHS
# -----------------------------
def analytical_rhs(env: str, controller, x_eq: torch.Tensor, u_eq: torch.Tensor):
    # Use controller's parameter dtype (falls back to x_eq.dtype)
    ctrl_dtype = next((p.dtype for p in controller.parameters()), x_eq.dtype)
    x_eq = x_eq.to(ctrl_dtype)
    u_eq = u_eq.to(ctrl_dtype)
    dtype = ctrl_dtype


    if env == "pendulum":
        import neural_network_lyapunov.examples.pendulum.pendulum as pendulum_mod
        plant = pendulum_mod.Pendulum(dtype=dtype)  # <-- required
        m = plant.mass; L = plant.length; g = plant.gravity
        u_bias = (-controller(x_eq) + u_eq).detach().to(dtype)  # precompute once

        def rhs(t, x_np):
            x_t = torch.tensor(x_np, dtype=dtype)
            u = controller(x_t.unsqueeze(0)).squeeze() + u_bias
            u = torch.clamp(u, -20.0, 20.0)
            # Use the library dynamics exactly as in your previous scripts
            return plant.dynamics(x_t, u.detach().cpu().numpy())
        return rhs

    elif env == "unicycle":
        import neural_network_lyapunov.examples.path_following_unicycle.path_following as pf
        plant = pf.Path_Following(dtype=torch.float64)

        def rhs(t, x_np):
            x_t = torch.tensor(x_np, dtype=torch.float64)
            u = controller(x_t.unsqueeze(0)).squeeze() - controller(x_eq) + u_eq
            u = torch.clamp(u, -10.0, 10.0)
            # Use the library dynamics exactly as in your previous scripts
            return plant.dynamics(x_np, u.detach().cpu().numpy())
        return rhs



# -----------------------------
# Metrics
# -----------------------------
def wrap_angle(a: np.ndarray) -> np.ndarray:
    return ((a + np.pi) % (2*np.pi)) - np.pi

def state_distance(env: str, x: np.ndarray, w1: float, w2: float) -> float:
    if env == "pendulum":
        dth = wrap_angle(x[0] - np.pi)
        return float(np.sqrt(w1 * dth**2 + w2 * x[1]**2))
    elif env == "unicycle":
        return float(np.sqrt(w1 * x[0]**2 + w2 * x[1]**2))
    else:
        raise ValueError

def batched_state_distance(env: str, X: torch.Tensor, w1: float, w2: float) -> torch.Tensor:
    if env == "pendulum":
        dth = ((X[:,0] - math.pi + math.pi) % (2*math.pi)) - math.pi
        return torch.sqrt(w1 * dth**2 + w2 * X[:,1]**2)
    else:
        return torch.sqrt(w1 * X[:,0]**2 + w2 * X[:,1]**2)


# -----------------------------
# Lyapunov
# -----------------------------
@torch.no_grad()
def V_batch(X: torch.Tensor, lyapunov, x_eq: torch.Tensor, V_lambda: float, R: torch.Tensor, device: torch.device):
    # Use the Lyapunov module's parameter dtype (fallback to X.dtype)
    try:
        lyap_dtype = next(p.dtype for p in lyapunov.parameters())
    except StopIteration:
        lyap_dtype = X.dtype

    X = X.to(device=device, dtype=lyap_dtype)
    x_eq = x_eq.to(device=device, dtype=lyap_dtype)

    if isinstance(R, torch.Tensor):
        R = R.to(device=device, dtype=lyap_dtype)
    else:
        R = torch.as_tensor(R, dtype=lyap_dtype, device=device)

    relu_at_eq = lyapunov(x_eq)
    V = lyapunov(X) - relu_at_eq + V_lambda * torch.norm(R @ (X - x_eq).T, p=1, dim=0).unsqueeze(1)
    return V.squeeze()



# -----------------------------
# Simulation back-ends
# -----------------------------
def simulate_single_analytical(env: str, controller, x_eq, u_eq, x0: np.ndarray, T: float, dt: float,
                               w1: float, w2: float, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    from scipy.integrate import solve_ivp
    # --- cast controller inputs to controller dtype
    ctrl_dtype = next((p.dtype for p in controller.parameters()), x_eq.dtype)
    x_eq = x_eq.to(ctrl_dtype)
    u_eq = u_eq.to(ctrl_dtype)
    rhs = analytical_rhs(env, controller, x_eq, u_eq)
    t_eval = np.arange(0.0, T, dt)

    def event_converged(t, x):
        return state_distance(env, x, w1, w2) - eps
    event_converged.terminal = True
    event_converged.direction = -1

    sol = solve_ivp(rhs, (0.0, T), x0, method="DOP853", rtol=1e-8, atol=1e-10,
                    t_eval=t_eval, events=event_converged)

    U = np.zeros(len(sol.t))
    u_bias = (-controller(x_eq) + u_eq).detach()   # uses ctrl_dtype
    with torch.no_grad():
        for k, x in enumerate(sol.y.T):
            xt = torch.tensor(x, dtype=ctrl_dtype).unsqueeze(0)
            u = controller(xt).squeeze() + u_bias
            U[k] = float(u.cpu().numpy())

    settled = sol.t_events and len(sol.t_events[0]) > 0
    t_settle = float(sol.t_events[0][0]) if settled else float(T)
    return sol.t, sol.y.T, U, t_settle, settled


# --- fast batched step (works on GPU), with vmap fallback ---
def step_forward_batch(forward_system, X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Vectorized wrapper around forward_system.step_forward(x, u).
    Uses torch.vmap (or functorch.vmap) if available; otherwise falls back to a chunked loop.
    """
    # Try native torch.vmap (PyTorch ≥2.1)
    if hasattr(torch, "vmap"):
        return torch.vmap(lambda x, u: forward_system.step_forward(x, u), in_dims=(0, 0))(X, U)

    # Try functorch.vmap as a fallback
    try:
        from functorch import vmap
        return vmap(lambda x, u: forward_system.step_forward(x, u))(X, U)
    except Exception:
        pass  # fall back to chunked loop below

    # Chunked loop (bigger chunks reduce Python overhead)
    B = max(1024, 8)  # tune to your GPU memory
    outs = []
    for i in range(0, X.shape[0], B):
        xb = X[i:i+B]
        ub = U[i:i+B]
        out = torch.stack([forward_system.step_forward(xb[j], ub[j]) for j in range(xb.shape[0])], dim=0)
        outs.append(out)
    return torch.cat(outs, dim=0)


@torch.no_grad()
def simulate_batch_ml(env: str, forward_system, controller, x_eq: torch.Tensor, u_eq: torch.Tensor,
                      u_lo: float, u_up: float, X0: torch.Tensor, T: float, dt: float,
                      w1: float, w2: float, eps: float, device: torch.device,
                      stride: int = 1, progress: bool = False, progress_desc: Optional[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized ML rollout:
      - controller(X) batched on GPU
      - precompute controller(x_eq)+u_eq as a bias
      - vmap() stepping
      - only step active (not-yet-converged) trajectories
      - stride: evaluate & check convergence every `stride` steps to cut overhead
    """
    steps = int(T / dt)
    # forward system dtype = truth for state
    state_dtype = forward_system.x_equilibrium.dtype
    X = X0.to(device=device, dtype=state_dtype).clone()

    controller = controller.to(device).eval()
    ctrl_dtype = next((p.dtype for p in controller.parameters()), state_dtype)
    x_eq_c = x_eq.to(ctrl_dtype); u_eq_c = u_eq.to(ctrl_dtype)
    u_bias = (-controller(x_eq_c) + u_eq_c).detach()

    N = X.shape[0]
    converged = torch.zeros(N, dtype=torch.bool, device=device)
    t_converge = torch.full((N,), float("inf"), dtype=torch.float64, device=device)

    for k in range(steps):
        active = ~converged
        if not torch.any(active):
            break
        Xa = X[active]
        Ua_ctrl = controller(Xa.to(ctrl_dtype)) + u_bias
        Ua = torch.clamp(Ua_ctrl.to(state_dtype), u_lo, u_up)
        Xa_next = step_forward_batch(forward_system, Xa, Ua)
        X[active] = Xa_next
        if (k + 1) % stride == 0:
            dist = batched_state_distance(env, X[active], w1, w2)
            newly = dist < eps
            if torch.any(newly):
                idx = torch.nonzero(active, as_tuple=False).squeeze(1)[newly]
                t_converge[idx] = (k + 1) * dt
                converged[idx] = True
    return converged.cpu().numpy(), t_converge.cpu().numpy()




@torch.no_grad()
def simulate_single_ml(env, forward_system, controller, x_eq, u_eq, u_lo, u_up, x0, T, dt, device):
    steps = int(T / dt)
    t = np.arange(0.0, T, dt)

    # state dtype = forward system’s equilibrium dtype
    state_dtype = forward_system.x_equilibrium.dtype
    X = torch.tensor(x0, device=device, dtype=state_dtype)

    controller = controller.to(device).eval()
    # controller dtype (weights)
    ctrl_dtype = next((p.dtype for p in controller.parameters()), state_dtype)
    x_eq_c = x_eq.to(ctrl_dtype); u_eq_c = u_eq.to(ctrl_dtype)

    traj = np.zeros((len(t), 2), dtype=float)
    U = np.zeros(len(t), dtype=float)
    for k, tk in enumerate(t):
        traj[k] = X.detach().cpu().numpy()
        u_ctrl = controller(X.unsqueeze(0).to(ctrl_dtype)).squeeze() - controller(x_eq_c) + u_eq_c
        u = torch.clamp(u_ctrl.to(state_dtype), u_lo, u_up)  # cast back to state dtype
        U[k] = float(u.detach().cpu().numpy())
        X = forward_system.step_forward(X, u)
    return t, traj, U



# -----------------------------
# Plot helpers
# -----------------------------
def draw_verification_box(ax, x_lo, x_up, color="k", linestyle="--", linewidth=1.0):
    rx = Rectangle((x_lo[0], x_lo[1]), x_up[0]-x_lo[0], x_up[1]-x_lo[1],
                   fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7)
    ax.add_patch(rx)


def setup_bins(bin_edges: List[float]) -> Tuple[np.ndarray, ListedColormap, BoundaryNorm, List[str]]:
    """
    Build an *index-based* discrete colormap for convergence-time bins.
    We color by bin index (0..K-1), so the norm boundaries are [-0.5, 0.5, ..., K-0.5].
    The last index (K-1) is reserved for the FAIL bin.
    """
    edges = np.sort(np.asarray(bin_edges, dtype=float))   # e.g., [5, 10, 15]
    K = len(edges) + 1                                    # time bins + 1 FAIL bin

    # colormap: time bins from viridis + red for FAIL
    base = cm.get_cmap("viridis", K-1)
    colors = [base(i) for i in range(base.N)]
    colors.append((0.9, 0.1, 0.1, 1.0))                   # red = FAIL
    cmap = ListedColormap(colors)

    # index-based boundaries (all finite): [-0.5, 0.5, ..., K-0.5]
    boundaries_idx = np.arange(-0.5, K + 0.5, 1.0)
    norm = BoundaryNorm(boundaries_idx, cmap.N, clip=True)

    labels = [f"≤{edges[0]:g}s"] \
           + [f"({edges[i-1]:g},{edges[i]:g}]s" for i in range(1, len(edges))] \
           + [f">{edges[-1]:g}s / fail"]

    return edges, cmap, norm, labels

def times_to_bin_ids(t_sec: np.ndarray, converged_mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Map time-to-convergence (seconds) to bin indices.
    - t_sec: shape (N,), possibly contains inf for non-converged.
    - converged_mask: boolean (N,), True if converged within horizon.
    - edges: sorted 1D array of bin edges (seconds), length E.
    Returns: bin_ids in [0..K-1], where K=E+1 and the last index K-1 is FAIL.
    """
    E = len(edges)
    K = E + 1
    # digitize finite times into 0..E (≤e1 -> 0, (e1,e2] -> 1, ..., >eE -> E)
    ids = np.digitize(t_sec, edges, right=True)           # 0..E
    ids = np.clip(ids, 0, E)                              # safety
    # set FAIL index for non-converged
    ids[~converged_mask] = E                              # E == K-1
    return ids


def plot_distance_curves_per_ic(ax, env: str, curves_no_fpl, curves_fpl, ic_cols, semilog=False, title=""):
    # curves_* : list of (t, dist) in the same IC order
    for i, ((t0, d0), (t1, d1)) in enumerate(zip(curves_no_fpl, curves_fpl)):
        ax.plot(t0, d0, linestyle="--", color=ic_cols[i], alpha=0.9, label=f"IC {i} No-FPL" if i==0 else None)
        ax.plot(t1, d1, linestyle="-",  color=ic_cols[i], alpha=0.9, label=f"IC {i} FPL" if i==0 else None)
    if semilog: ax.set_yscale("log")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance to $x_{eq}$")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)


def plot_roa_bins(ax, X_ic: np.ndarray, bin_ids: np.ndarray, cmap, norm, title: str,
                  x_lo=None, x_up=None, eq=None, env=None):
    sc = ax.scatter(X_ic[:,0], X_ic[:,1], c=bin_ids, cmap=cmap, norm=norm, s=12, linewidths=0, alpha=0.95)
    if (x_lo is not None) and (x_up is not None):
        draw_verification_box(ax, x_lo, x_up, color="k", linestyle="--")
    if eq is not None:
        ax.plot(eq[0], eq[1], marker="*", color="red", markersize=8, label="eq")
    if env == "pendulum":
        ax.set_xlabel("$\\theta$ (rad)"); ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
    elif env == "unicycle":
        ax.set_xlabel("$d_e$ (m)"); ax.set_ylabel("$\\theta_e$ (rad)")
    ax.set_title(title); ax.grid(True, alpha=0.2)
    return sc


def plot_V_surface(ax, X1, X2, Vgrid, title: str, add_floor_contours=True, env=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    surf = ax.plot_surface(X1, X2, Vgrid, linewidth=0, antialiased=True, cmap="viridis", alpha=0.95)
    if add_floor_contours:
        zmin = np.nanmin(Vgrid)
        ax.contour(X1, X2, Vgrid, zdir='z', offset=zmin, cmap='viridis', levels=20)
        ax.set_zlim(zmin, np.nanmax(Vgrid))
    if env == "pendulum":
        ax.set_xlabel("$\\theta$ (rad)"); ax.set_ylabel("$\\dot{\\theta}$ (rad/s)"); ax.set_zlabel("$V(x)$")
    elif env == "unicycle":
        ax.set_xlabel("$d_e$ (m)"); ax.set_ylabel("$\\theta_e$ (rad)"); ax.set_zlabel("$V(x)$")
    ax.set_title(title)
    return surf


# ---------- module-scope worker for parallel analytical runs ----------
def run_analytical_job(job_tuple):
    """
    job_tuple: (env, bound_level, fpl_flag, x0, T, dt, w1, w2, eps)
    Loads the right controller on CPU, builds RHS, integrates, and returns (t, X, U).
    """
    (env, bound_level, fpl_flag, x0, T, dt, w1, w2, eps) = job_tuple

    device_cpu = torch.device("cpu")
    # We only need x_eq/u_eq for the RHS; load from forward_system
    _, _, _, _, _, _, x_eq, u_eq = load_forward_system(env, device_cpu, dt)

    spec = RunSpec(env=env, bound_level=bound_level, fpl=fpl_flag)
    controller, _, _ = load_models(spec, device=device_cpu)

    t, X, U, _, _ = simulate_single_analytical(
        env, controller, x_eq, u_eq, np.asarray(x0, dtype=float),
        T, dt, w1, w2, eps
    )
    return t, X, U


# -----------------------------
# Main figure builder
# -----------------------------
def build_figures(env: str, bound_level: int, use_gpu: bool,
                  roa_samples: int, bin_edges: List[float],
                  eps: float, T: float, dt: float, w1: float, w2: float,
                  save_dir: str, no_save: bool, semilog_dist: bool,
                  stride: int, analytical_workers: int, ic_seed: int,
                  ic_distance: int, sample_mode_distance: str,
                  ic_phase: int, sample_mode_phase: str,
                  ic_violin: int, sample_mode_violin: str,
                  ic_lyap: int, sample_mode_lyap: str,
                  ic_pareto: int, sample_mode_pareto: str):


    device = setup_device(use_gpu)
    os.makedirs(save_dir, exist_ok=True)

    # Load forward system and bounds
    dynamics_model, forward_system, x_lo_t, x_up_t, u_lo, u_up, x_eq_t, u_eq_t = load_forward_system(env, device, dt)
    x_lo = x_lo_t.detach().cpu().numpy(); x_up = x_up_t.detach().cpu().numpy()
    x_eq = x_eq_t.to(device); u_eq = u_eq_t.to(device)
    

    # IC samplers
    # Per-plot IC counts and sampling modes (only the new flags)
    rng = np.random.default_rng(ic_seed)

    dist_ic_list   = sample_ic_list(ic_distance, sample_mode_distance, x_lo, x_up, rng)
    phase_ic_list  = sample_ic_list(ic_phase,    sample_mode_phase,     x_lo, x_up, rng)
    violin_ic_list = sample_ic_list(ic_violin,   sample_mode_violin,    x_lo, x_up, rng)
    lyap_ic_list   = sample_ic_list(ic_lyap,     sample_mode_lyap,      x_lo, x_up, rng)
    pareto_ic_list = sample_ic_list(ic_pareto,   sample_mode_pareto,    x_lo, x_up, rng)

    # Colors per plot
    ic_cols_phase = ic_colors(len(phase_ic_list))
    ic_cols_dist  = ic_colors(len(dist_ic_list))



    # Build run specs
    runs = []
    for fpl_flag in [False, True]:
            runs.append(RunSpec(env=env, bound_level=bound_level, fpl=fpl_flag))

    # ---------------- Distance curves (analytical + ML) ----------------
    print("[*] Simulating distance-to-eq curves...")
    curves_analytical_nofpl = []
    curves_analytical_fpl = []
    curves_ml_nofpl = []
    curves_ml_fpl = []

    # For analytical use CPU (solve_ivp), for ML use device (GPU if available)
    spec_nofpl = next(r for r in runs if not r.fpl)
    spec_fpl = next(r for r in runs if r.fpl)
    controller_nf, _, _ = load_models(spec_nofpl, device=torch.device("cpu"))
    controller_f,  _, _ = load_models(spec_fpl,  device=torch.device("cpu"))
    controller_nf_gpu, _, _ = load_models(spec_nofpl, device=device)
    controller_f_gpu,  _, _ = load_models(spec_fpl,  device=device)

    from concurrent.futures import as_completed

    if analytical_workers > 0 and len(dist_ic_list) > 1:
        with ProcessPoolExecutor(max_workers=analytical_workers) as ex:
            # ----- No-FPL jobs -----
            jobs_nf = {
                ex.submit(
                    run_analytical_job,
                    (env, bound_level, False, x0, T, dt, w1, w2, eps)
                ): i
                for i, x0 in enumerate(dist_ic_list)
            }
            curves_analytical_nofpl = [None] * len(dist_ic_list)
            for fut in tqdm(as_completed(jobs_nf), total=len(jobs_nf),
                            desc="Analytical ICs (No-FPL, parallel)"):
                i = jobs_nf[fut]
                t0, X0, U0 = fut.result()
                d0 = np.array([state_distance(env, x, w1, w2) for x in X0])
                curves_analytical_nofpl[i] = (t0, d0)

            # ----- FPL jobs -----
            jobs_f = {
                ex.submit(
                    run_analytical_job,
                    (env, bound_level, True, x0, T, dt, w1, w2, eps)
                ): i
                for i, x0 in enumerate(dist_ic_list)
            }
            curves_analytical_fpl = [None] * len(dist_ic_list)
            for fut in tqdm(as_completed(jobs_f), total=len(jobs_f),
                            desc="Analytical ICs (FPL, parallel)"):
                i = jobs_f[fut]
                t1, X1, U1 = fut.result()
                d1 = np.array([state_distance(env, x, w1, w2) for x in X1])
                curves_analytical_fpl[i] = (t1, d1)
    else:

        for i, x0 in enumerate(tqdm(dist_ic_list, desc="Analytical ICs")):
            t0, X0, U0, ts0, ok0 = simulate_single_analytical(env, controller_nf, x_eq.cpu(), u_eq.cpu(), x0, T, dt, w1, w2, eps)
            d0 = np.array([state_distance(env, x, w1, w2) for x in X0])
            curves_analytical_nofpl.append((t0, d0))

            t1, X1, U1, ts1, ok1 = simulate_single_analytical(env, controller_f,  x_eq.cpu(), u_eq.cpu(), x0, T, dt, w1, w2, eps)
            d1 = np.array([state_distance(env, x, w1, w2) for x in X1])
            curves_analytical_fpl.append((t1, d1))

    for i, x0 in enumerate(tqdm(dist_ic_list, desc="ML ICs")):
        t0, X0, U0 = simulate_single_ml(env, forward_system, controller_nf_gpu, x_eq, u_eq, u_lo, u_up, x0, T, dt, device)
        d0 = np.array([state_distance(env, x, w1, w2) for x in X0])
        curves_ml_nofpl.append((t0, d0))

        t1, X1, U1 = simulate_single_ml(env, forward_system, controller_f_gpu, x_eq, u_eq, u_lo, u_up, x0, T, dt, device)
        d1 = np.array([state_distance(env, x, w1, w2) for x in X1])
        curves_ml_fpl.append((t1, d1))

    # Plot per-IC distance (two subplots)
    fig_dist, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    plot_distance_curves_per_ic(axA, env, curves_analytical_nofpl, curves_analytical_fpl, ic_cols_dist,
                            semilog=semilog_dist, title="Distance to $x_{eq}$ (Analytical)")
    plot_distance_curves_per_ic(axB, env, curves_ml_nofpl,         curves_ml_fpl,         ic_cols_dist,
                                semilog=semilog_dist, title="Distance to $x_{eq}$ (ML dynamics)")

    _save_or_show(fig_dist, save_dir, f"{env}_bound{bound_level}_distance_per_ic", no_save)

    # ---------------- ROA with bins + stats ----------------
    print("[*] ROA sampling + convergence-time bins...")
    N = roa_samples
    X0 = np.column_stack([rng.uniform(x_lo[0], x_up[0], size=N),
                          rng.uniform(x_lo[1], x_up[1], size=N)]).astype(np.float64)
    X0_t = torch.from_numpy(X0).to(torch.float64)

    edges, cmap, norm, labels = setup_bins(bin_edges)

    bin_maps = {}
    stats = {}

    for fpl_flag, label in [(False, "No-FPL"), (True, "FPL")]:
        spec = RunSpec(env=env, bound_level=bound_level, fpl=fpl_flag)
        controller, _, _ = load_models(spec, device=device)

        conv_mask, t_conv = simulate_batch_ml(
            env, forward_system, controller, x_eq, u_eq, u_lo, u_up,
            X0_t, T, dt, w1, w2, eps, device, stride=stride,
            progress=True, progress_desc=f"ROA steps ({label})"
        )



        # stats from the actual run (not from indices)
        rate = float(np.mean(conv_mask))
        mean_t = float(np.mean(t_conv[conv_mask])) if np.any(conv_mask) else float("nan")

        # map times to *bin indices* (last index is fail)
        bin_ids = times_to_bin_ids(t_conv, conv_mask, edges)

        bin_maps[label] = bin_ids
        stats[label] = (rate, mean_t)

    fig_roa, (axD, axE) = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True, sharey=True)
    scD = plot_roa_bins(axD, X0, bin_maps["No-FPL"], cmap, norm, "ROA (time bins) – No-FPL", x_lo, x_up, x_eq.detach().cpu().numpy(), env)
    scE = plot_roa_bins(axE, X0, bin_maps["FPL"],    cmap, norm, "ROA (time bins) – FPL",    x_lo, x_up, x_eq.detach().cpu().numpy(), env)

    cax = fig_roa.add_axes([0.92, 0.18, 0.015, 0.65])
    cb = fig_roa.colorbar(scE, cax=cax, orientation="vertical")
    K = len(labels)
    cb.set_ticks(np.arange(K))
    cb.set_ticklabels(labels)


    # stats annotations
    for ax, label in [(axD,"No-FPL"), (axE,"FPL")]:
        rate, mean_t = stats[label]
        ax.text(0.02, 0.02, f"conv. rate={rate*100:.1f}%\nmean t={mean_t:.3f}s",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
    _save_or_show(fig_roa, save_dir, f"{env}_bound{bound_level}_roa_bins", no_save)

    # ---------------- V(x) surfaces ----------------
    print("[*] Evaluating Lyapunov surface grids...")
    grid_n = 151 if env == "pendulum" else 121
    x1 = np.linspace(x_lo[0], x_up[0], grid_n)
    x2 = np.linspace(x_lo[1], x_up[1], grid_n)
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    Xflat = np.column_stack([X1.ravel(), X2.ravel()])
    Xgrid = torch.tensor(Xflat, device=device)

    V_grids = {}
    for fpl_flag, label in [(False, "No-FPL"), (True, "FPL")]:
        spec = RunSpec(env=env, bound_level=bound_level, fpl=fpl_flag)
        _, lyapunov, R = load_models(spec, device=device)

        # Allocate with the Lyapunov module's dtype
        lyap_dtype = next((p.dtype for p in lyapunov.parameters()), x_eq.dtype)
        V_vals = torch.empty(Xgrid.shape[0], device=device, dtype=lyap_dtype)

        B = 65536
        for i in tqdm(range(0, Xgrid.shape[0], B), desc=f"V-grid ({label})", leave=False):
            j = min(i + B, Xgrid.shape[0])
            V_vals[i:j] = V_batch(Xgrid[i:j], lyapunov, x_eq, 1.0, R, device)

        V_grids[label] = V_vals.detach().cpu().numpy().reshape(X1.shape)



    fig_V = plt.figure(figsize=(10.5, 8))
    gs_V = GridSpec(2, 1, figure=fig_V, hspace=0.28)
    axF = fig_V.add_subplot(gs_V[0, 0], projection="3d")
    axG = fig_V.add_subplot(gs_V[1, 0], projection="3d")
    plot_V_surface(axF, X1, X2, V_grids["No-FPL"], "V(x) Surface – No-FPL", add_floor_contours=True, env=env)
    plot_V_surface(axG, X1, X2, V_grids["FPL"],    "V(x) Surface – FPL",    add_floor_contours=True, env=env)
    _save_or_show(fig_V, save_dir, f"{env}_bound{bound_level}_V_surfaces", no_save)

    # ---------------- Phase diagrams & control effort scatter ----------------
    print("[*] Phase diagrams & control effort scatter...")

    # simulate for the same ICs under both controllers, analytical and ML
    # Pre-load controllers (GPU for ML single-step; CPU for analytical)
    ctrl_nf_cpu, _, _ = load_models(spec_nofpl, device=torch.device("cpu"))
    ctrl_f_cpu,  _, _ = load_models(spec_fpl,  device=torch.device("cpu"))
    ctrl_nf_gpu, _, _ = load_models(spec_nofpl, device=device)
    ctrl_f_gpu,  _, _ = load_models(spec_fpl,  device=device)

    # Per-IC storage for control effort plots (keep these lists! do NOT reassign later)
    ana_ts, ana_us_nf, ana_us_f = [], [], []
    ml_ts,  ml_us_nf,  ml_us_f  = [], [], []

    # ---------- Analytical phase + effort ----------
    ana_trajs_nf, ana_trajs_f = [], []
    for i, x0 in enumerate(tqdm(phase_ic_list, desc="Phase (analytical)")):
        t_nf, Xnf, Unf, *_ = simulate_single_analytical(
            env, ctrl_nf_cpu, x_eq.cpu(), u_eq.cpu(), x0, T, dt, w1, w2, eps
        )
        t_f,  Xf,  Uf,  *_ = simulate_single_analytical(
            env, ctrl_f_cpu,  x_eq.cpu(), u_eq.cpu(), x0, T, dt, w1, w2, eps
        )
        ana_trajs_nf.append(Xnf); ana_trajs_f.append(Xf)
        ana_ts.append(np.asarray(t_nf))
        ana_us_nf.append(np.asarray(Unf))
        ana_us_f.append(np.asarray(Uf))

    # ---------- ML phase + effort ----------
    ml_trajs_nf, ml_trajs_f = [], []
    for i, x0 in enumerate(tqdm(phase_ic_list, desc="Phase (ML)")):
        t_nf, Xnf, Unf = simulate_single_ml(
            env, forward_system, ctrl_nf_gpu, x_eq, u_eq, u_lo, u_up, x0, T, dt, device
        )
        t_f,  Xf,  Uf  = simulate_single_ml(
            env, forward_system, ctrl_f_gpu,  x_eq, u_eq, u_lo, u_up, x0, T, dt, device
        )
        ml_trajs_nf.append(Xnf); ml_trajs_f.append(Xf)
        ml_ts.append(np.asarray(t_nf))
        ml_us_nf.append(np.asarray(Unf))
        ml_us_f.append(np.asarray(Uf))

    # ---------- Phase diagrams ----------
    fig_phase, (axP1, axP2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for i in range(len(phase_ic_list)):
        c = ic_cols_phase[i]
        axP1.plot(ana_trajs_nf[i][:,0], ana_trajs_nf[i][:,1], linestyle="--", color=c,
                label="IC 0 No-FPL" if i == 0 else None)
        axP1.plot(ana_trajs_f[i][:,0],  ana_trajs_f[i][:,1],  linestyle="-",  color=c,
                label="IC 0 FPL" if i == 0 else None)
        axP2.plot(ml_trajs_nf[i][:,0],  ml_trajs_nf[i][:,1],  linestyle="--", color=c,
                label="IC 0 No-FPL" if i == 0 else None)
        axP2.plot(ml_trajs_f[i][:,0],   ml_trajs_f[i][:,1],   linestyle="-",  color=c,
                label="IC 0 FPL" if i == 0 else None)

    for ax in (axP1, axP2):
        draw_verification_box(ax, x_lo, x_up, color="k", linestyle="--")
        ax.plot(x_eq.detach().cpu().numpy()[0], x_eq.detach().cpu().numpy()[1],
                marker="*", color="red", markersize=8)
        if env == "pendulum":
            ax.set_xlabel("$\\theta$ (rad)"); ax.set_ylabel("$\\dot{\\theta}$ (rad/s)")
        else:
            ax.set_xlabel("$d_e$ (m)"); ax.set_ylabel("$\\theta_e$ (rad)")
    axP1.set_title("Phase diagram (Analytical)")
    axP2.set_title("Phase diagram (ML dynamics)")
    axP2.legend(ncol=2, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    _save_or_show(fig_phase, save_dir, f"{env}_bound{bound_level}_phase_diagrams", no_save)

    # ---------- Control effort panels (per-IC color; dashed=No-FPL, solid=FPL) + zoomed inset ----------
    from matplotlib.patches import Rectangle as _Rect
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
    import matplotlib.lines as mlines

    fig_u, (axU1, axU2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    INSET_T = min(0.75, 0.10 * float(T))  # first 10% of horizon, cap at 0.75 s
    INSET_Q = 0.90                        # 90th percentile for inset y-limit
    INSET_SIZE = 0.48                     # relative inset size

    def _plot_panel_with_inset(ax, times_list, us_nf_list, us_f_list, title):
        N = min(len(times_list), len(us_nf_list), len(us_f_list))
        cols = ic_cols_phase[:N]    # use the phase palette computed above
        y_nf_cache, y_f_cache = [], []

        for i in range(N):
            t_i = np.asarray(times_list[i])
            if env == "unicycle":
                y_nf = np.abs(np.asarray(us_nf_list[i]) - float(u_eq))
                y_f  = np.abs(np.asarray(us_f_list[i])  - float(u_eq))
            else:
                y_nf = np.abs(np.asarray(us_nf_list[i]))
                y_f  = np.abs(np.asarray(us_f_list[i]))
            y_nf_cache.append(y_nf); y_f_cache.append(y_f)
            ax.plot(t_i[:len(y_nf)], y_nf, linestyle="--", color=cols[i], linewidth=2.0)
            ax.plot(t_i[:len(y_f)],  y_f,  linestyle="-",  color=cols[i], linewidth=2.0)

        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$|u(t)-u_{eq}|$" if env == "unicycle" else r"$|u(t)|$")
        ax.grid(True, alpha=0.3)

        # Inset
        iax = _inset_axes(ax, width=f"{int(INSET_SIZE*100)}%", height=f"{int(INSET_SIZE*100)}%",
                        loc="upper right", borderpad=0.9)
        inset_y = []
        for i in range(N):
            t_i = np.asarray(times_list[i])
            m = t_i <= INSET_T
            if not np.any(m):
                continue
            iax.plot(t_i[m], y_nf_cache[i][:np.sum(m)], linestyle="--", color=cols[i], linewidth=2.0)
            iax.plot(t_i[m], y_f_cache[i][:np.sum(m)],  linestyle="-",  color=cols[i], linewidth=2.0)
            inset_y.append(y_nf_cache[i][:np.sum(m)])
            inset_y.append(y_f_cache[i][:np.sum(m)])
        iax.grid(True, alpha=0.3)
        iax.set_xlim(0.0, INSET_T)
        if len(inset_y) > 0:
            ymax = np.quantile(np.concatenate(inset_y), INSET_Q)
            iax.set_ylim(0.0, max(1e-6, 1.05 * float(ymax)))

        # show zoom rectangle on parent
        rect = _Rect((0.0, 0.0), INSET_T, iax.get_ylim()[1],
                    fill=False, ec="gray", ls="--", lw=1.0, alpha=0.8, zorder=3)
        ax.add_patch(rect)

    # draw both panels
    _plot_panel_with_inset(axU1, ana_ts, ana_us_nf, ana_us_f, "Control effort (Analytical)")
    _plot_panel_with_inset(axU2, ml_ts,  ml_us_nf,  ml_us_f,  "Control effort (ML dynamics)")

    # legend to explain dashed vs solid
    style_leg = [
        mlines.Line2D([], [], color="k", linestyle="--", linewidth=2.0, label="No-FPL"),
        mlines.Line2D([], [], color="k", linestyle="-",  linewidth=2.0, label="FPL"),
    ]
    axU1.legend(handles=style_leg, loc="upper left")
    axU2.legend(handles=style_leg, loc="upper left")

    fig_u.tight_layout()
    _save_or_show(fig_u, save_dir, f"{env}_bound{bound_level}_control_effort", no_save)

        # ================= Violin plot: sum |u| (no Δt) =================
    print("[*] Violin: effort sum (no Δt) ...")

    # We’ll simulate each IC in violin_ic_list under both controllers, Analytical and ML.
    ana_sum_nf, ana_sum_f = [], []
    ml_sum_nf,  ml_sum_f  = [], []

    # Reload CPU controllers for analytical (already have ctrl_nf_cpu/ctrl_f_cpu)
    for x0 in tqdm(violin_ic_list, desc="Violin (Analytical)"):
        t_nf, Xnf, Unf, *_ = simulate_single_analytical(env, ctrl_nf_cpu, x_eq.cpu(), u_eq.cpu(),
                                                        x0, T, dt, w1, w2, eps)
        t_f,  Xf,  Uf,  *_ = simulate_single_analytical(env, ctrl_f_cpu,  x_eq.cpu(), u_eq.cpu(),
                                                        x0, T, dt, w1, w2, eps)
        ana_sum_nf.append(effort_sum(np.asarray(Unf), env, u_eq))
        ana_sum_f.append( effort_sum(np.asarray(Uf),  env, u_eq))

    # ML rollouts use GPU if available (we already have ctrl_nf_gpu/ctrl_f_gpu)
    for x0 in tqdm(violin_ic_list, desc="Violin (ML)"):
        t_nf, Xnf, Unf = simulate_single_ml(env, forward_system, ctrl_nf_gpu, x_eq, u_eq, u_lo, u_up,
                                            x0, T, dt, device)
        t_f,  Xf,  Uf  = simulate_single_ml(env, forward_system, ctrl_f_gpu,  x_eq, u_eq, u_lo, u_up,
                                            x0, T, dt, device)
        ml_sum_nf.append(effort_sum(np.asarray(Unf), env, u_eq))
        ml_sum_f.append( effort_sum(np.asarray(Uf),  env, u_eq))

    fig_v, (axV1, axV2) = plt.subplots(2, 1, figsize=(7.0, 7.2), sharex=True)
    parts1 = axV1.violinplot([ana_sum_nf, ana_sum_f], showmeans=True, showextrema=False)
    axV1.set_title("Effort sum (Analytical)")
    axV1.set_ylabel(r"$\sum |u(t)|$" if env != "unicycle" else r"$\sum |u(t)-u_{eq}|$")
    axV1.set_xticks([1, 2], labels=["No-FPL", "FPL"])
    axV1.grid(True, alpha=0.3)

    parts2 = axV2.violinplot([ml_sum_nf, ml_sum_f], showmeans=True, showextrema=False)
    axV2.set_title("Effort sum (ML dynamics)")
    axV2.set_ylabel(r"$\sum |u(t)|$" if env != "unicycle" else r"$\sum |u(t)-u_{eq}|$")
    axV2.set_xticks([1, 2], labels=["No-FPL", "FPL"])
    axV2.grid(True, alpha=0.3)

    fig_v.tight_layout()
    _save_or_show(fig_v, save_dir, f"{env}_bound{bound_level}_effort_sum_violin", no_save)

        # ================= Lyapunov decrease profiles =================
    print("[*] Lyapunov decrease profiles...")

    # Load Lyapunov + R on CPU for simplicity (works for both envs)
    ly_nf_cpu, R_nf_cpu = load_models(spec_nofpl, device=torch.device("cpu"))[1:]
    ly_f_cpu,  R_f_cpu  = load_models(spec_fpl,  device=torch.device("cpu"))[1:]

    fig_lv, (axL1, axL2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    # Analytical V(t)
    for i, x0 in enumerate(tqdm(lyap_ic_list, desc="Lyap (Analytical)")):
        t_nf, Xnf, Unf, *_ = simulate_single_analytical(env, ctrl_nf_cpu, x_eq.cpu(), u_eq.cpu(),
                                                        x0, T, dt, w1, w2, eps)
        t_f,  Xf,  Uf,  *_ = simulate_single_analytical(env, ctrl_f_cpu,  x_eq.cpu(), u_eq.cpu(),
                                                        x0, T, dt, w1, w2, eps)
        c = plt.get_cmap("tab20")(i % 20)
        Vnf = lyap_on_traj(np.asarray(Xnf), ly_nf_cpu, x_eq.cpu(), R_nf_cpu, device=torch.device("cpu"))
        Vf  = lyap_on_traj(np.asarray(Xf),  ly_f_cpu,  x_eq.cpu(), R_f_cpu,  device=torch.device("cpu"))
        axL1.plot(t_nf[:len(Vnf)], Vnf, linestyle="--", color=c, linewidth=1.8)
        axL1.plot(t_f[:len(Vf)],   Vf,  linestyle="-",  color=c, linewidth=1.8)

    axL1.set_title("V(t) (Analytical)")
    axL1.set_xlabel("Time (s)"); axL1.set_ylabel("V(x)")
    axL1.grid(True, alpha=0.3)

    # ML V(t) — we can reuse CPU lyapunovs (we’re just evaluating V on arrays)
    for i, x0 in enumerate(tqdm(lyap_ic_list, desc="Lyap (ML)")):
        t_nf, Xnf, Unf = simulate_single_ml(env, forward_system, ctrl_nf_gpu, x_eq, u_eq, u_lo, u_up,
                                            x0, T, dt, device)
        t_f,  Xf,  Uf  = simulate_single_ml(env, forward_system, ctrl_f_gpu,  x_eq, u_eq, u_lo, u_up,
                                            x0, T, dt, device)
        c = plt.get_cmap("tab20")(i % 20)
        Vnf = lyap_on_traj(np.asarray(Xnf), ly_nf_cpu, x_eq.cpu(), R_nf_cpu, device=torch.device("cpu"))
        Vf  = lyap_on_traj(np.asarray(Xf),  ly_f_cpu,  x_eq.cpu(), R_f_cpu,  device=torch.device("cpu"))
        axL2.plot(t_nf[:len(Vnf)], Vnf, linestyle="--", color=c, linewidth=1.8)
        axL2.plot(t_f[:len(Vf)],   Vf,  linestyle="-",  color=c, linewidth=1.8)

    axL2.set_title("V(t) (ML dynamics)")
    axL2.set_xlabel("Time (s)")
    axL2.grid(True, alpha=0.3)

    import matplotlib.lines as mlines
    leg = [mlines.Line2D([], [], color="k", ls="--", lw=2.0, label="No-FPL"),
           mlines.Line2D([], [], color="k", ls="-",  lw=2.0, label="FPL")]
    axL1.legend(handles=leg, loc="upper right"); axL2.legend(handles=leg, loc="upper right")

    fig_lv.tight_layout()
    _save_or_show(fig_lv, save_dir, f"{env}_bound{bound_level}_lyapunov_decrease_profiles", no_save)


    # ================= Pareto: effort sum vs settling time =================
    print("[*] Pareto: effort vs settling time ...")

    # Collect trajectories (Analytical)
    ana_tr_nf, ana_u_nf, ana_t_nf = [], [], []
    ana_tr_f,  ana_u_f,  ana_t_f  = [], [], []
    for x0 in tqdm(pareto_ic_list, desc="Pareto (Analytical)"):
        t_nf, Xnf, Unf, *_ = simulate_single_analytical(env, ctrl_nf_cpu, x_eq.cpu(), u_eq.cpu(),
                                                        x0, T, dt, w1, w2, eps)
        t_f,  Xf,  Uf,  *_ = simulate_single_analytical(env, ctrl_f_cpu,  x_eq.cpu(), u_eq.cpu(),
                                                        x0, T, dt, w1, w2, eps)
        ana_tr_nf.append(np.asarray(Xnf)); ana_u_nf.append(np.asarray(Unf)); ana_t_nf.append(np.asarray(t_nf))
        ana_tr_f.append(np.asarray(Xf));   ana_u_f.append(np.asarray(Uf));   ana_t_f.append(np.asarray(t_f))

    # Collect trajectories (ML)
    ml_tr_nf, ml_u_nf, ml_t_nf = [], [], []
    ml_tr_f,  ml_u_f,  ml_t_f  = [], [], []
    for x0 in tqdm(pareto_ic_list, desc="Pareto (ML)"):
        t_nf, Xnf, Unf = simulate_single_ml(env, forward_system, ctrl_nf_gpu, x_eq, u_eq, u_lo, u_up,
                                            x0, T, dt, device)
        t_f,  Xf,  Uf  = simulate_single_ml(env, forward_system, ctrl_f_gpu,  x_eq, u_eq, u_lo, u_up,
                                            x0, T, dt, device)
        ml_tr_nf.append(np.asarray(Xnf)); ml_u_nf.append(np.asarray(Unf)); ml_t_nf.append(np.asarray(t_nf))
        ml_tr_f.append(np.asarray(Xf));   ml_u_f.append(np.asarray(Uf));   ml_t_f.append(np.asarray(t_f))

    # Compute metrics
    def pareto_arrays(tr_list, u_list, t_list):
        t_eps, e_sum = [], []
        for X, u, t in zip(tr_list, u_list, t_list):
            d = np.array([state_distance(env, x, w1, w2) for x in X])
            t_eps.append(settling_time_from_dist(t, d, eps))
            e_sum.append(effort_sum(u, env, u_eq))
        return np.asarray(t_eps), np.asarray(e_sum)

    t_nf_a, e_nf_a = pareto_arrays(ana_tr_nf, ana_u_nf, ana_t_nf)
    t_f_a,  e_f_a  = pareto_arrays(ana_tr_f,  ana_u_f,  ana_t_f)
    t_nf_m, e_nf_m = pareto_arrays(ml_tr_nf,  ml_u_nf,  ml_t_nf)
    t_f_m,  e_f_m  = pareto_arrays(ml_tr_f,   ml_u_f,   ml_t_f)

    # Plot arrows per IC (No-FPL hollow -> FPL filled)
    def scatter_arrows(ax, t_nf, e_nf, t_f, e_f):
        C = plt.get_cmap("tab20")
        for i in range(len(t_nf)):
            c = C(i % 20)
            ax.scatter(t_nf[i], e_nf[i], facecolors='none', edgecolors=c, s=50)
            ax.scatter(t_f[i],  e_f[i],  facecolors=c,     edgecolors=c, s=50, alpha=0.9)
            ax.annotate("", xy=(t_f[i], e_f[i]), xytext=(t_nf[i], e_nf[i]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.9))

    fig_p, (axA, axM) = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    scatter_arrows(axA, t_nf_a, e_nf_a, t_f_a, e_f_a)
    scatter_arrows(axM, t_nf_m, e_nf_m, t_f_m, e_f_m)
    for ax in (axA, axM):
        ax.set_xlabel(r"Settling time $t_\epsilon$ (s)")
        ax.set_ylabel(r"$\sum |u(t)|$" if env != "unicycle" else r"$\sum |u(t)-u_{eq}|$")
        ax.grid(True, alpha=0.3)
    axA.set_title("Pareto (Analytical)"); axM.set_title("Pareto (ML dynamics)")
    fig_p.tight_layout()
    _save_or_show(fig_p, save_dir, f"{env}_bound{bound_level}_pareto_effort_vs_speed", no_save)



    # ---------------- Consolidated summary (optional) ----------------
    fig_sum = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig_sum, hspace=0.32, wspace=0.25)

    # Distance (aggregate: show analytical panel)
    axS1 = fig_sum.add_subplot(gs[0, 0:2])
    # create pseudo-aggregate: plot only FPL vs No-FPL mean across ICs
    for label, curves in [("No-FPL", curves_analytical_nofpl), ("FPL", curves_analytical_fpl)]:
        grid = np.linspace(0, T, 300)
        D = np.vstack([np.interp(grid, t, d) for (t, d) in curves])
        mu = np.nanmean(D, axis=0)
        p25, p75 = np.nanpercentile(D, 25, axis=0), np.nanpercentile(D, 75, axis=0)
        ls = "--" if label == "No-FPL" else "-"
        axS1.plot(grid, mu, label=label, linestyle=ls, linewidth=2.5)
        axS1.fill_between(grid, p25, p75, alpha=0.15)
    if semilog_dist: axS1.set_yscale("log")
    axS1.set_title("Distance to $x_{eq}$ (Analytical, mean ± IQR)"); axS1.set_xlabel("Time (s)"); axS1.set_ylabel("Distance"); axS1.grid(True, alpha=0.3); axS1.legend()

    # Training wall-clock
    axS2 = fig_sum.add_subplot(gs[0, 2])
    axS3 = fig_sum.add_subplot(gs[0, 3])
    for fpl_flag, label in [(False, "No-FPL"), (True, "FPL")]:
        spec = RunSpec(env=env, bound_level=bound_level, fpl=fpl_flag)
    axS2.set_title("Pre-train Loss vs Time"); axS2.set_xlabel("Time (s)"); axS2.set_ylabel("Loss"); axS2.grid(True, alpha=0.3); axS2.legend()
    axS3.set_title("MILP Violation vs Time"); axS3.set_xlabel("Time (s)"); axS3.set_ylabel("Violation"); axS3.grid(True, alpha=0.3); axS3.legend()

    # ROA bins
    axS4 = fig_sum.add_subplot(gs[1, 0])
    axS5 = fig_sum.add_subplot(gs[1, 1])
    scS4 = plot_roa_bins(axS4, X0, bin_maps["No-FPL"], cmap, norm, "ROA – No-FPL", x_lo, x_up, x_eq.detach().cpu().numpy(), env=env)
    scS5 = plot_roa_bins(axS5, X0, bin_maps["FPL"],    cmap, norm, "ROA – FPL", x_lo, x_up, x_eq.detach().cpu().numpy(), env=env)
    # In the summary figure section
    cax = fig_sum.add_axes([0.52, 0.47, 0.015, 0.15])
    cb  = fig_sum.colorbar(scS5, cax=cax, orientation="vertical")
    K = len(labels)
    cb.set_ticks(np.arange(K))
    cb.set_ticklabels(labels)




    # V surfaces
    axS6 = fig_sum.add_subplot(gs[1:, 2], projection="3d")
    axS7 = fig_sum.add_subplot(gs[1:, 3], projection="3d")
    plot_V_surface(axS6, X1, X2, V_grids["No-FPL"], "V(x) – No-FPL", env=env)
    plot_V_surface(axS7, X1, X2, V_grids["FPL"], "V(x) – FPL", env=env)

    _save_or_show(fig_sum, save_dir, f"{env}_bound{bound_level}_summary", no_save)


def _save_or_show(fig, save_dir, name, no_save=False):
    if no_save:
        plt.show()
        return
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, name + ".png"), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="ACC unified visualizations (extended)")
    p.add_argument("--env", choices=["pendulum","unicycle"], required=True)
    p.add_argument("--bound_level", type=int, default=3)
    p.add_argument("--gpu", action="store_true", help="Use GPU where available for ROA and V grids.")
    p.add_argument("--roa_samples", type=int, default=2000, help="#random ICs for ROA bins.")
    p.add_argument("--bins", type=float, nargs="*", default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                   help="Convergence-time bin edges in seconds.")
    p.add_argument("--eps", type=float, default=1e-3, help="Convergence threshold in distance metric.")
    p.add_argument("--T", type=float, default=25.0, help="Horizon for traj/ROA.")
    p.add_argument("--dt", type=float, default=0.01, help="Step size for sim.")
    p.add_argument("--w1", type=float, default=1.0, help="Distance weight for x1.")
    p.add_argument("--w2", type=float, default=1.0, help="Distance weight for x2.")
    p.add_argument("--save_dir", type=str, default="figs", help="Where to store figures.")
    p.add_argument("--no_save", action="store_true", help="Don't save; just display with plt.show().")
    p.add_argument("--semilog_dist", action="store_true", help="Use semilog y-axis for distance plots.")
    p.add_argument("--stride", type=int, default=1, help="Check convergence every N steps in ROA (speeds up long horizons).")
    p.add_argument("--precision", choices=["float64","float32"], default="float64", help="Precision for ML rollouts/V grids.")
    p.add_argument("--analytical_workers", type=int, default=0, help="#workers to parallelize analytical ICs (0=off).")
    # Per-plot IC counts
    p.add_argument("--ic_distance", type=int, default=5,
                        help="Number of ICs for distance-to-eq curves.")
    p.add_argument("--ic_phase", type=int, default=5,
                        help="Number of ICs for phase diagrams + control-effort.")
    p.add_argument("--ic_violin", type=int, default=5,
                        help="Number of ICs for the effort-sum violin plot.")
    p.add_argument("--ic_lyap", type=int, default=5,
                        help="Number of ICs for Lyapunov decrease profiles.")
    p.add_argument("--ic_pareto", type=int, default=5,
                        help="Number of ICs for Pareto (effort vs settling time).")

    # Per-plot sampling modes
    _choices_sampler = ["random", "grid"]
    p.add_argument("--sample_mode_distance", choices=_choices_sampler, default="random",
                        help="IC sampling for distance-to-eq figure.")
    p.add_argument("--sample_mode_phase", choices=_choices_sampler, default="random",
                        help="IC sampling for phase + control-effort figures.")
    p.add_argument("--sample_mode_violin", choices=_choices_sampler, default="random",
                        help="IC sampling for violin figure.")
    p.add_argument("--sample_mode_lyap", choices=_choices_sampler, default="random",
                        help="IC sampling for Lyapunov profiles.")
    p.add_argument("--sample_mode_pareto", choices=_choices_sampler, default="random",
                        help="IC sampling for Pareto figure.")

    # Reproducibility for random sampling
    p.add_argument("--ic_seed", type=int, default=12345,
                        help="Seed for per-plot IC random sampling.")



    args = p.parse_args()
    # Precision toggle (float32 is much faster on GPU)
    if args.precision == "float32":
        torch.set_default_dtype(torch.float32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


    build_figures(
        env=args.env,
        bound_level=args.bound_level,
        use_gpu=args.gpu,
        roa_samples=args.roa_samples,
        bin_edges=args.bins,
        eps=args.eps,
        T=args.T,
        dt=args.dt,
        w1=args.w1,
        w2=args.w2,
        save_dir=args.save_dir,
        no_save=args.no_save,
        semilog_dist=args.semilog_dist,
        stride=args.stride,
        analytical_workers=args.analytical_workers,
        ic_seed=args.ic_seed,
        ic_distance=args.ic_distance,
        sample_mode_distance=args.sample_mode_distance,
        ic_phase=args.ic_phase,
        sample_mode_phase=args.sample_mode_phase,
        ic_violin=args.ic_violin,
        sample_mode_violin=args.sample_mode_violin,
        ic_lyap=args.ic_lyap,
        sample_mode_lyap=args.sample_mode_lyap,
        ic_pareto=args.ic_pareto,
        sample_mode_pareto=args.sample_mode_pareto,
    )


if __name__ == "__main__":
    main()
