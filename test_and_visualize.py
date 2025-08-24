#!/usr/bin/env python3
"""
Compare two Lyapunov-controlled path-following models (unicycle).

What it does
------------
- Loads controller, Lyapunov, and R from each model directory.
- Computes/plots the Lyapunov field for each model + their difference.
- Runs identical rollouts from the same initial states for both models.
- Compares V(t), u(t), and cumulative control energy per trajectory.

Usage
-----
# single model (saves standard plots)
python test_and_visualize.py \
  --bound_level 40 \
  --model_dir_a neural_network_lyapunov/examples/path_following_unicycle/monotonic/monotonic_bound40 \
  --output_dir viz_A

# compare two models
python test_and_visualize.py \
  --bound_level 40 \
  --model_dir_a neural_network_lyapunov/examples/path_following_unicycle/monotonic/monotonic_bound40 \
  --model_dir_b neural_network_lyapunov/examples/path_following_unicycle/monotonic_bound40_fpl \
  --label_a "Monotonic" \
  --label_b "Monotonic + FPL" \
  --output_dir viz_compare


#   Compare two models with convergence
    python test_and_visualize.py \
  --bound_level 40 \
  --model_dir_a neural_network_lyapunov/examples/path_following_unicycle/data/monotonic/monotonic_bound40 \
  --model_dir_b neural_network_lyapunov/examples/path_following_unicycle/data/monotonic/monotonic_bound40_fpl \
  --label_a "Monotonic" \
  --label_b "Monotonic + FPL" \
  --output_dir viz_compare \
  --until_converged --dt 0.01 --max_time 10 --v_eps 1e-6

"""

import argparse
import glob
import os
from typing import Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Project imports
import neural_network_lyapunov.relu_system as relu_system
import neural_network_lyapunov.examples.path_following_unicycle.path_following as path_following
import neural_network_lyapunov.feedback_system as feedback_system
import neural_network_lyapunov.monotonic_lyapunov_init.custom_lyapunov as lyapunov


# ----------------------------- Core visualizer ----------------------------- #
class PathFollowingVisualizer:
    """Visualizer for the path-following unicycle system with Lyapunov control."""

    def __init__(self, model_dir: str, bound_level: int = 40):
        self.model_dir = os.path.abspath(model_dir)
        self.bound_level = bound_level
        self.device = torch.device("cpu")
        self.dtype = torch.float64

        # Plant + bounds
        self.plant = path_following.Path_Following(self.dtype)
        self.x_lo = torch.tensor([-0.8, -0.8], dtype=self.dtype) * bound_level / 40.0
        self.x_up = torch.tensor([+0.8, +0.8], dtype=self.dtype) * bound_level / 40.0
        self.u_lo = torch.tensor([-10.0], dtype=self.dtype)
        self.u_up = torch.tensor([+10.0], dtype=self.dtype)

        # Equilibria
        self.x_equilibrium = torch.tensor([0.0, 0.0], dtype=self.dtype)
        self.u_equilibrium = torch.tensor([self.plant.v], dtype=self.dtype)

        # Forward model (.pt saved by preprocess)
        FWD_MODEL_NAME = "path_following_unicycle_forward_model.pt"

        # Let user override via CLI (passed through from main())
        override = getattr(self, "fwd_model_path", None)
        if override and os.path.isfile(override):
            fwd_path = override
        else:
            data_root = os.path.dirname(
                os.path.dirname(self.model_dir)
            )  # .../examples/path_following_unicycle
            candidates = [
                os.path.join(
                    data_root, "data", "preprocess", FWD_MODEL_NAME
                ),  # <-- your path
                os.path.join(
                    data_root, "preprocess", FWD_MODEL_NAME
                ),  # fallback for other layouts
            ]
            fwd_path = next((p for p in candidates if os.path.isfile(p)), None)

        if fwd_path is None:
            raise FileNotFoundError(
                "Could not find forward model. Tried:\n  "
                + "\n  ".join(candidates)
                + "\nOr pass --fwd_model_path explicitly."
            )

        dynamics_relu = torch.load(fwd_path, map_location=self.device)

        self.forward_system = relu_system.ReLUSystemGivenEquilibrium(
            self.dtype,
            self.x_lo,
            self.x_up,
            self.u_lo,
            self.u_up,
            dynamics_relu,
            self.x_equilibrium,
            self.u_equilibrium,
            0.01,
        )

        # Load controller, V, and R
        self._load_models()

        # Wrap systems
        self.closed_loop_system = feedback_system.FeedbackSystem(
            self.forward_system,
            self.controller_relu,
            self.x_equilibrium,
            self.u_equilibrium,
            self.u_lo.detach().cpu().numpy(),
            self.u_up.detach().cpu().numpy(),
        )
        self.lyapunov_hybrid_system = lyapunov.LyapunovDiscreteTimeHybridSystem(
            self.closed_loop_system, self.lyapunov_relu
        )

    # -- helpers --
    def _pick_first(self, patterns):
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(self.model_dir, p)))
            if hits:
                return hits[0]
        return None

    def _load_models(self):
        """Load *this* run's controller, Lyapunov, and R."""
        # Accept either plain or *_fpl_* filenames inside the chosen folder.
        ctrl_path = self._pick_first(
            [f"*bound{self.bound_level}_controller.pt", "*controller.pt"]
        )
        v_path = self._pick_first(
            [f"*bound{self.bound_level}_lyapunov.pt", "*lyapunov.pt"]
        )
        r_path = self._pick_first([f"*bound{self.bound_level}_R.pt", "*_R.pt"])

        if not (ctrl_path and v_path and r_path):
            raise FileNotFoundError(
                f"Could not find controller/V/R files under: {self.model_dir}\n"
                f"Expected patterns like: *_controller.pt, *_lyapunov.pt, *_R.pt "
                f"(bound_level={self.bound_level})."
            )

        self.controller_relu = torch.load(ctrl_path, map_location=self.device)
        self.lyapunov_relu = torch.load(v_path, map_location=self.device)
        self.R = torch.load(r_path, map_location=self.device)

        print(f"✓ Loaded from {self.model_dir}")
        print(f"  - controller: {os.path.basename(ctrl_path)}")
        print(f"  - lyapunov  : {os.path.basename(v_path)}")
        print(f"  - R         : {os.path.basename(r_path)}")

    # -- computations --
    def compute_lyapunov_value(self, x, V_lambda: float = 0.1) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype)
        return self.lyapunov_hybrid_system.lyapunov_value(
            x, self.x_equilibrium, V_lambda, R=self.R
        )

    def compute_control(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=self.dtype)
        return self.closed_loop_system.compute_u(x)

    def simulate_trajectory(
        self, x0: np.ndarray, t_final: float = 5.0, dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate using discrete step_forward inside a continuous integrator."""

        def dynamics(_t, x):
            x_t = torch.tensor(x, dtype=self.dtype)
            x_next = self.closed_loop_system.step_forward(x_t)
            return (x_next.detach().numpy() - x) / dt

        t_eval = np.arange(0.0, t_final, dt)
        sol = scipy.integrate.solve_ivp(
            dynamics, [0.0, t_final], np.asarray(x0, dtype=float), t_eval=t_eval
        )

        t = sol.t
        x_traj = sol.y.T
        u_traj = np.zeros((len(t), 1))
        V_traj = np.zeros(len(t))
        for i, xi in enumerate(x_traj):
            u_traj[i] = self.compute_control(xi).detach().numpy()
            V_traj[i] = self.compute_lyapunov_value(xi).item()
        return t, x_traj, u_traj, V_traj

    def simulate_until_convergence(
        self,
        x0: np.ndarray,
        dt: float = 0.01,
        max_time: float = 10.0,
        v_eps: float = 1e-6,
        stall_window: int = 50,
        stall_rel_change: float = 1e-4,
    ):
        """Roll discrete steps until V < v_eps or V stalls; cap by max_time.
        Returns (t, X, U, V, converged).
        Shapes: t[N], X[N,2], U[N,1], V[N].
        """
        x = torch.tensor(x0, dtype=self.dtype)
        t_hist = [0.0]
        x_hist = [x.detach().numpy()]
        u_hist = [self.compute_control(x).detach().numpy()]  # align U with times
        V_hist = [self.compute_lyapunov_value(x).item()]

        n_max = int(np.ceil(max_time / dt))
        converged = False

        for _ in range(n_max):
            # one discrete step
            x = self.closed_loop_system.step_forward(x)
            t_hist.append(t_hist[-1] + dt)
            x_hist.append(x.detach().numpy())
            u_hist.append(self.compute_control(x).detach().numpy())
            V_val = self.compute_lyapunov_value(x).item()
            V_hist.append(V_val)

            # hard threshold on V
            if V_val <= v_eps:
                converged = True
                break

            # stall detection: V isn't moving anymore (avoid asymptotic tail)
            if len(V_hist) >= stall_window:
                win = V_hist[-stall_window:]
                vmin, vmax = min(win), max(win)
                denom = max(vmin, 1e-12)
                if (vmax - vmin) / denom < stall_rel_change:
                    converged = True
                    break

        t = np.asarray(t_hist)
        X = np.stack(x_hist, axis=0)
        U = np.vstack(u_hist)  # shape [N, 1]
        V = np.asarray(V_hist)
        return t, X, U, V, converged

    # -- single-model plots (for backward-compatibility) --
    def save_all_plots(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        print("\nGenerating single-model plots...")

        # Lyapunov field + contours
        fig = self._plot_lyapunov_function()
        fig.savefig(
            os.path.join(output_dir, "lyapunov_function.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # ROA-ish multi-contour view
        fig = self._plot_region_of_attraction()
        fig.savefig(
            os.path.join(output_dir, "region_of_attraction.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Rollouts + V(t) + u(t)
        fig = self._plot_trajectories()
        fig.savefig(
            os.path.join(output_dir, "trajectories.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        fig = self._plot_control_effort()
        fig.savefig(
            os.path.join(output_dir, "control_effort.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        print(f"✓ Saved plots to {output_dir}")

    def _plot_lyapunov_function(self, n_points=100, V_lambda=0.1):
        de = np.linspace(self.x_lo[0].item(), self.x_up[0].item(), n_points)
        th = np.linspace(self.x_lo[1].item(), self.x_up[1].item(), n_points)
        DE, TH = np.meshgrid(de, th)
        V = np.zeros_like(DE)
        for i in range(n_points):
            for j in range(n_points):
                x = torch.tensor([DE[i, j], TH[i, j]], dtype=self.dtype)
                V[i, j] = self.compute_lyapunov_value(x, V_lambda).item()

        fig = plt.figure(figsize=(15, 6))
        ax2 = fig.add_subplot(111)
        levels = np.percentile(V.flatten(), np.linspace(5, 95, 20))
        c = ax2.contourf(DE, TH, V, levels=levels, cmap="viridis")
        plt.colorbar(c, ax=ax2)
        ax2.plot(0, 0, "r*", ms=12)
        rect = patches.Rectangle(
            (self.x_lo[0].item(), self.x_lo[1].item()),
            self.x_up[0].item() - self.x_lo[0].item(),
            self.x_up[1].item() - self.x_lo[1].item(),
            linewidth=2,
            edgecolor="k",
            facecolor="none",
            linestyle="--",
        )
        ax2.add_patch(rect)
        ax2.set_xlabel("$d_e$")
        ax2.set_ylabel("$\\theta_e$")
        ax2.set_aspect("equal")
        ax2.set_title("Lyapunov level sets")
        ax2.grid(True, alpha=0.3)
        return fig

    def _plot_region_of_attraction(self, n_points=120):
        de = np.linspace(self.x_lo[0].item(), self.x_up[0].item(), n_points)
        th = np.linspace(self.x_lo[1].item(), self.x_up[1].item(), n_points)
        DE, TH = np.meshgrid(de, th)
        V = np.zeros_like(DE)
        for i in range(n_points):
            for j in range(n_points):
                x = torch.tensor([DE[i, j], TH[i, j]], dtype=self.dtype)
                V[i, j] = self.compute_lyapunov_value(x).item()

        fig, ax = plt.subplots(figsize=(7, 7))
        for p in [10, 25, 50, 75, 90, 95]:
            lvl = np.percentile(V, p)
            ax.contour(DE, TH, V, levels=[lvl], linewidths=2)
        ax.plot(0, 0, "r*", ms=12)
        ax.set_xlabel("$d_e$")
        ax.set_ylabel("$\\theta_e$")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("ROA-style multi-contours")
        return fig

    def _plot_trajectories(self, n_trajectories=10, t_final=5.0):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        seeds = np.random.default_rng(0).uniform(
            [self.x_lo[0].item(), self.x_lo[1].item()],
            [self.x_up[0].item(), self.x_up[1].item()],
            size=(n_trajectories, 2),
        )
        trajs = [self.simulate_trajectory(x0, t_final) for x0 in seeds]

        ax = axes[0, 0]
        for t, x, _, _ in trajs:
            ax.plot(x[:, 0], x[:, 1], alpha=0.7)
        ax.plot(0, 0, "r*", ms=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title("State-space trajectories")
        ax.set_xlabel("$d_e$")
        ax.set_ylabel("$\\theta_e$")

        ax = axes[0, 1]
        for t, x, _, _ in trajs:
            ax.plot(t, x[:, 0], alpha=0.7)
        ax.set_title("$d_e$ vs time")
        ax.set_xlabel("t")
        ax.set_ylabel("$d_e$")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for t, x, _, _ in trajs:
            ax.plot(t, x[:, 1], alpha=0.7)
        ax.set_title("$\\theta_e$ vs time")
        ax.set_xlabel("t")
        ax.set_ylabel("$\\theta_e$")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for t, _, _, V in trajs:
            ax.semilogy(t, V, alpha=0.7)
        ax.set_title("V(x) vs time")
        ax.set_xlabel("t")
        ax.set_ylabel("V")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_control_effort(self, n_trajectories=5, t_final=5.0):
        fig, ax = plt.subplots(figsize=(10, 5))
        seeds = np.random.default_rng(1).uniform(
            [self.x_lo[0].item(), self.x_lo[1].item()],
            [self.x_up[0].item(), self.x_up[1].item()],
            size=(n_trajectories, 2),
        )
        for x0 in seeds:
            t, _, u, _ = self.simulate_trajectory(x0, t_final)
            ax.plot(t, u, alpha=0.7)
        ax.axhline(self.u_lo.item(), linestyle="--")
        ax.axhline(self.u_up.item(), linestyle="--")
        ax.set_title("Control effort u(t)")
        ax.set_xlabel("t")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)
        return fig


# --------------------------- Two-model comparison -------------------------- #
def _grid_V(visualizer: PathFollowingVisualizer, n_points=120, V_lambda=0.1):
    de = np.linspace(visualizer.x_lo[0].item(), visualizer.x_up[0].item(), n_points)
    th = np.linspace(visualizer.x_lo[1].item(), visualizer.x_up[1].item(), n_points)
    DE, TH = np.meshgrid(de, th)
    V = np.zeros_like(DE)
    for i in range(n_points):
        for j in range(n_points):
            x = torch.tensor([DE[i, j], TH[i, j]], dtype=visualizer.dtype)
            V[i, j] = visualizer.compute_lyapunov_value(x, V_lambda).item()
    return DE, TH, V


def compare_lyapunov(va, vb, outdir, V_lambda=0.1, n_points=120, labels=("A", "B")):
    DEa, THa, Va = _grid_V(va, n_points=n_points, V_lambda=V_lambda)
    DEb, THb, Vb = _grid_V(vb, n_points=n_points, V_lambda=V_lambda)
    assert np.allclose(DEa, DEb) and np.allclose(THa, THb)
    DV = Vb - Va

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    c0 = axes[0].contourf(DEa, THa, Va, levels=30)
    axes[0].set_title(f"V(x) — {labels[0]}")
    plt.colorbar(c0, ax=axes[0])

    c1 = axes[1].contourf(DEb, THb, Vb, levels=30)
    axes[1].set_title(f"V(x) — {labels[1]}")
    plt.colorbar(c1, ax=axes[1])

    c2 = axes[2].contourf(DEa, THa, DV, levels=30)
    axes[2].set_title("ΔV = V_B − V_A")
    plt.colorbar(c2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("$d_e$")
        ax.set_ylabel("$\\theta_e$")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(
        os.path.join(outdir, "compare_lyapunov.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)


def _plot_lyap_surface(ax, D, TH, Vgrid, title):
    """D, TH, Vgrid are meshgrids (shape [Ny, Nx])."""
    surf = ax.plot_surface(D, TH, Vgrid, linewidth=0, antialiased=True, cmap="viridis")
    ax.set_xlabel(r"$d_e$")
    ax.set_ylabel(r"$\theta_e$")
    ax.set_zlabel(r"$V(x)$")
    ax.set_title(title)
    ax.view_init(elev=35, azim=-135)
    return surf


def compare_lyapunov_surface(D, TH, V_A, V_B, label_a, label_b, outdir):
    """3D surfaces for V_A, V_B, and ΔV."""
    import os, matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 5.5))
    ax1 = fig.add_subplot(131, projection="3d")
    _plot_lyap_surface(ax1, D, TH, V_A, f"V(x) — {label_a}")
    ax2 = fig.add_subplot(132, projection="3d")
    _plot_lyap_surface(ax2, D, TH, V_B, f"V(x) — {label_b}")
    ax3 = fig.add_subplot(133, projection="3d")
    _plot_lyap_surface(ax3, D, TH, V_B - V_A, r"$\Delta V = V_B - V_A$")
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, "compare_lyapunov_surface.png"), dpi=200)
    plt.close(fig)


def compare_rollouts(
    va,
    vb,
    outdir,
    n_trajectories=10,
    t_final=5.0,
    labels=("A", "B"),
    until_converged=False,
    dt=0.01,
    max_time=10.0,
    v_eps=1e-6,
    stall_window=50,
    stall_rel_change=1e-4,
):
    rng = np.random.default_rng(0)
    seeds = rng.uniform(
        [va.x_lo[0].item(), va.x_lo[1].item()],
        [va.x_up[0].item(), va.x_up[1].item()],
        size=(n_trajectories, 2),
    )

    trajsA, trajsB = [], []
    for x0 in seeds:
        if until_converged:
            trajsA.append(
                va.simulate_until_convergence(
                    x0,
                    dt=dt,
                    max_time=max_time,
                    v_eps=v_eps,
                    stall_window=stall_window,
                    stall_rel_change=stall_rel_change,
                )
            )
            trajsB.append(
                vb.simulate_until_convergence(
                    x0,
                    dt=dt,
                    max_time=max_time,
                    v_eps=v_eps,
                    stall_window=stall_window,
                    stall_rel_change=stall_rel_change,
                )
            )
        else:
            trajsA.append(va.simulate_trajectory(x0, t_final=t_final, dt=dt))
            trajsB.append(vb.simulate_trajectory(x0, t_final=t_final, dt=dt))

    # State-space overlays
    fig, ax = plt.subplots(figsize=(7, 7))
    # remove the convergence flag from the tuples
    trajsA = [(t, x, u, V) for (t, x, u, V, _) in trajsA]
    trajsB = [(t, x, u, V) for (t, x, u, V, _) in trajsB]
    for (tA, xA, _, _), (tB, xB, _, _) in zip(trajsA, trajsB):
        ax.plot(
            xA[:, 0],
            xA[:, 1],
            alpha=0.8,
            label=labels[0] if not hasattr(ax, "_sa") else "",
        )
        ax.plot(
            xB[:, 0],
            xB[:, 1],
            alpha=0.8,
            linestyle="--",
            label=labels[1] if not hasattr(ax, "_sb") else "",
        )
        ax._sa, ax._sb = True, True
    ax.plot(0, 0, "r*", ms=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title("State-space trajectories")
    ax.set_xlabel("$d_e$")
    ax.set_ylabel("$\\theta_e$")
    ax.legend()
    fig.savefig(
        os.path.join(outdir, "compare_trajectories.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # V(t) overlays
    fig, ax = plt.subplots(figsize=(10, 5))
    for (tA, _, _, VA), (tB, _, _, VB) in zip(trajsA, trajsB):
        ax.semilogy(tA, VA, alpha=0.7)
        ax.semilogy(tB, VB, alpha=0.7, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("V(x)")
    ax.set_title("Lyapunov vs time (solid=A, dashed=B)")
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(outdir, "compare_V_time.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # u(t) overlays
    fig, ax = plt.subplots(figsize=(10, 5))
    for (tA, _, uA, _), (tB, _, uB, _) in zip(trajsA, trajsB):
        ax.plot(tA, uA, alpha=0.7)
        ax.plot(tB, uB, alpha=0.7, linestyle="--")
    ax.axhline(va.u_lo.item(), linestyle="--")
    ax.axhline(va.u_up.item(), linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control input")
    ax.set_title("Control effort (solid=A, dashed=B)")
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(outdir, "compare_control_traces.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    def cum_energy(u_traj, u_eq, dt):
        e = (u_traj.reshape(-1, 1) - float(u_eq)) ** 2
        return float(np.sum(e) * dt)

    EA = [cum_energy(uA, va.u_equilibrium, dt) for (_, _, uA, _) in trajsA]
    EB = [cum_energy(uB, vb.u_equilibrium, dt) for (_, _, uB, _) in trajsB]

    idx = np.arange(len(EA))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(idx - 0.2, EA, width=0.4, label=labels[0])
    ax.bar(idx + 0.2, EB, width=0.4, label=labels[1])
    ax.set_xlabel("Trajectory #")
    ax.set_ylabel(r"$\sum (u - u^*)^2\,dt$")
    ax.set_title("Cumulative control energy per trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(
        os.path.join(outdir, "compare_control_energy.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Console summary
    print("\n=== Comparison summary ===")
    print(f"Avg control energy — {labels[0]}: {np.mean(EA):.4e}")
    print(f"Avg control energy — {labels[1]}: {np.mean(EB):.4e}")


# ----------------------------------- CLI ----------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Compare two trained models (unicycle).")
    p.add_argument("--bound_level", type=int, default=40)
    p.add_argument(
        "--model_dir_a",
        type=str,
        default=None,
        help="Folder containing *controller.pt, *lyapunov.pt, *_R.pt",
    )
    p.add_argument(
        "--model_dir_b",
        type=str,
        default=None,
        help="Second folder to compare against (optional)",
    )
    p.add_argument("--label_a", type=str, default="Model A")
    p.add_argument("--label_b", type=str, default="Model B")
    p.add_argument("--output_dir", type=str, default="./visualization_results")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    p.add_argument(
        "--until_converged",
        action="store_true",
        help="Simulate each trajectory until convergence instead of fixed horizon",
    )
    p.add_argument("--dt", type=float, default=0.01, help="Simulation step (s)")
    p.add_argument(
        "--max_time", type=float, default=10.0, help="Hard cap on sim time (s)"
    )
    p.add_argument(
        "--v_eps", type=float, default=1e-6, help="Convergence threshold on V(x)"
    )
    p.add_argument(
        "--stall_window",
        type=int,
        default=50,
        help="Stop early if V stops changing over this many steps",
    )
    p.add_argument(
        "--stall_rel_change",
        type=float,
        default=1e-4,
        help="Relative change threshold in V over stall_window",
    )

    args = p.parse_args()

    # Reasonable defaults if not provided (match your screenshot)
    root = "neural_network_lyapunov/examples/path_following_unicycle"
    if args.model_dir_a is None:
        args.model_dir_a = os.path.join(
            root, "monotonic", f"monotonic_bound{args.bound_level}"
        )
    if args.model_dir_b is None:
        # Compare against FPL variant by default if present
        candidate = os.path.join(root, f"monotonic_bound{args.bound_level}_fpl")
        args.model_dir_b = candidate if os.path.isdir(candidate) else None

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_dir_b is None:
        # Single-model visualizations
        va = PathFollowingVisualizer(args.model_dir_a, args.bound_level)
        va.save_all_plots(args.output_dir)

        # Quick numeric check at eq and a random state
        x_eq = np.array([0.0, 0.0])
        u_eq = float(va.compute_control(x_eq))
        V_eq = float(va.compute_lyapunov_value(x_eq))
        print(f"\nAt equilibrium: u={u_eq:.6f}, V={V_eq:.6f} (expect u≈v, V≈0)")
        x_test = np.array([0.3, -0.2])
        print(
            f"u(x_test)={float(va.compute_control(x_test)):.6f}, "
            f"V(x_test)={float(va.compute_lyapunov_value(x_test)):.6f}"
        )

        if args.show:
            plt.show()
        return

    # Two-model comparison
    print("\n== Comparing two models ==")
    print(f"A: {args.model_dir_a}\nB: {args.model_dir_b}")
    va = PathFollowingVisualizer(args.model_dir_a, args.bound_level)
    vb = PathFollowingVisualizer(args.model_dir_b, args.bound_level)

    compare_lyapunov(
        va,
        vb,
        args.output_dir,
        V_lambda=0.1,
        n_points=120,
        labels=(args.label_a, args.label_b),
    )
    # build the grids and values for each model
    D, TH, V_A = _grid_V(va, n_points=120, V_lambda=0.1)  # A: visualizer for model A
    _, _, V_B = _grid_V(vb, n_points=120, V_lambda=0.1)  # B: visualizer for model B

    # then make the 3D surfaces
    compare_lyapunov_surface(
        D, TH, V_A, V_B, args.label_a, args.label_b, args.output_dir
    )

    compare_rollouts(
        va,
        vb,
        args.output_dir,
        n_trajectories=20,
        t_final=5.0,
        labels=(args.label_a, args.label_b),
        until_converged=args.until_converged,
        dt=args.dt,
        max_time=args.max_time,
        v_eps=args.v_eps,
        stall_window=args.stall_window,
        stall_rel_change=args.stall_rel_change,
    )

    print(f"\n✓ Comparison plots saved to: {args.output_dir}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
