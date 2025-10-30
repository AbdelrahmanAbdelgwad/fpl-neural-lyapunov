#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper figures (Phase, ROA-continuous, Effort map, Lyapunov surfaces+GIF)
with parallel/parallelized implementations:

- Phase (analytical): optional CPU multiprocessing (--phase_workers)
- Phase (ML): batched GPU rollouts (one loop over time for all ICs)
- ROA: batched (unchanged)
- Effort map: batched (unchanged)
- Lyapunov surfaces: batched evaluation (unchanged)
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import matplotlib as mpl

# ---------- paper font sizing ----------
TITLE_FT  = 22
LABEL_FT  = 22
TICK_FT   = 20
LEGEND_FT = 22
CBLAB_FT  = 22
CBTICK_FT = 20

mpl.rcParams.update({
    "font.size": TITLE_FT,       # base
    "axes.titlesize": TITLE_FT,
    "axes.labelsize": LABEL_FT,
    "xtick.labelsize": TICK_FT,
    "ytick.labelsize": TICK_FT,
    "legend.fontsize": LEGEND_FT,
    "figure.titlesize": TITLE_FT,
})

def style_axes(ax, title=None):
    """Force big labels, ticks, and (existing or new) title on a 2D axis."""
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FT, pad=6)
    else:
        # If a title already exists, bump its fontsize
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=TITLE_FT, pad=6)
    ax.xaxis.label.set_size(LABEL_FT)
    ax.yaxis.label.set_size(LABEL_FT)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FT)


def style_legend(ax):
    """Force legend font size if a legend exists."""
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(LEGEND_FT)

def style_colorbar(cb, label=None):
    """Force colorbar label/ticks sizes."""
    if label is not None:
        cb.set_label(label, fontsize=CBLAB_FT)
    cb.ax.tick_params(labelsize=CBTICK_FT)
    cb.ax.yaxis.label.set_size(CBLAB_FT)

def style_axes_3d(ax, title=None):
    """Same idea for 3D axes."""
    if title is not None:
        ax.set_title(title, fontsize=TITLE_FT, pad=6)
    ax.set_xlabel(ax.get_xlabel(), fontsize=LABEL_FT, labelpad=8)
    ax.set_ylabel(ax.get_ylabel(), fontsize=LABEL_FT, labelpad=8)
    if hasattr(ax, "set_zlabel"):
        ax.set_zlabel(getattr(ax, "get_zlabel")(), fontsize=LABEL_FT, labelpad=10)
    ax.tick_params(axis="both", which="both", labelsize=TICK_FT)
    if hasattr(ax, "zaxis"):
        ax.zaxis.set_tick_params(labelsize=TICK_FT)


# Your existing helper module
import acc_viz as viz


# -----------------------------
# Utilities
# -----------------------------
def setup_device(use_gpu: bool) -> torch.device:
    return torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")


def set_default_dtype(precision: str):
    torch.set_default_dtype(torch.float64 if precision == "float64" else torch.float32)


def save_or_show(fig, save_dir: str, stem: str, no_save: bool):
    if no_save:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def axis_labels(env: str):
    if env == "pendulum":
        return r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"
    return r"$d_e$ (m)", r"$\theta_e$ (rad)"  # Path_Following (unicycle)


def sample_ic_list(n, mode, x_lo, x_up, rng):
    x_lo = np.asarray(x_lo, dtype=float)
    x_up = np.asarray(x_up, dtype=float)
    if n <= 0:
        return []
    if mode == "random":
        X = rng.uniform(low=x_lo, high=x_up, size=(n, 2))
        return [X[i] for i in range(n)]
    nx = int(np.ceil(np.sqrt(n)))
    ny = int(np.ceil(n / nx))
    g0 = np.linspace(x_lo[0], x_up[0], nx)
    g1 = np.linspace(x_lo[1], x_up[1], ny)
    G0, G1 = np.meshgrid(g0, g1)
    X = np.stack([G0.ravel(), G1.ravel()], axis=1)
    return [X[i] for i in range(n)]


# -----------------------------
# Parallel worker (analytical phase) — must be top-level for pickling
# -----------------------------
def _analytical_phase_worker(pack):
    """Returns (X_traj) for one IC (No-FPL or FPL), analytical dynamics."""
    env, controller_cpu, x_eq_cpu, u_eq_cpu, x0, T, dt, w1, w2, eps = pack
    t, X, U, _, _ = viz.simulate_single_analytical(
        env, controller_cpu, x_eq_cpu, u_eq_cpu, x0, T, dt, w1, w2, eps
    )
    return X  # [steps, 2]


# -----------------------------
# Batched ML phase rollouts
# -----------------------------
@torch.no_grad()
def _phase_batch_ml(forward_system, controller, x_eq_t, u_eq_t, u_lo, u_up,
                    X0_list, T, dt, device, desc="Phase (ML, batched)"):
    """Roll B ICs together on one device; returns list of [steps,2] arrays."""
    B = len(X0_list)
    if B == 0:
        return []

    X = torch.as_tensor(np.array(X0_list), device=device, dtype=torch.float64)  # [B,2]
    xeq_X = x_eq_t.to(device=device, dtype=torch.float64)

    # controller dtype/device
    c_dtype = next(controller.parameters()).dtype
    xeq_C = x_eq_t.to(device=device, dtype=c_dtype)
    ueq_C = u_eq_t.to(device=device, dtype=c_dtype)

    # bounds for clamping
    if np.isscalar(u_lo):
        ulo_t = torch.tensor(float(u_lo), device=device, dtype=c_dtype)
        uup_t = torch.tensor(float(u_up), device=device, dtype=c_dtype)
        clamp = lambda u: torch.clamp(u, ulo_t, uup_t)
    else:
        ulo_t = torch.as_tensor(u_lo, device=device, dtype=c_dtype)
        uup_t = torch.as_tensor(u_up, device=device, dtype=c_dtype)
        clamp = lambda u: torch.max(torch.min(u, uup_t), ulo_t)

    steps = int(np.ceil(T / dt))
    traj = torch.empty((steps + 1, B, 2), device=device, dtype=torch.float64)
    traj[0] = X

    for k in tqdm(range(steps), desc=desc, leave=False):
        Xc = X.to(dtype=c_dtype)
        u = controller(Xc) - controller(xeq_C) + ueq_C  # [B,?]
        u = clamp(u).to(dtype=torch.float64)
        X = forward_system.step_forward(X, u)  # keeps float64
        traj[k + 1] = X

    out = traj.detach().cpu().numpy()  # [T+1,B,2]
    return [out[:, i, :] for i in range(B)]


# -----------------------------
# Phase diagrams (parallel analytical + batched ML)
# -----------------------------
def plot_phase_diagrams(args, forward_system, x_eq_t, u_eq_t,
                        ctrl_nf_cpu, ctrl_f_cpu, ctrl_nf_gpu, ctrl_f_gpu,
                        x_lo, x_up, u_lo, u_up,
                        phase_ic_list, save_dir, stem):
    colors = viz.ic_colors(len(phase_ic_list))
    xl, yl = axis_labels(args.env)

    figA, (axA1, axA2) = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    # after ax1.legend(...) and ax2.legend(...)
    for ax in (axA1, axA2):
        style_axes(ax)  # (legend is added later; we'll upsize it then)


    # ---- Analytical (optional multiprocessing) ----
    if args.phase_workers > 1 and len(phase_ic_list) > 0:
        jobs = []
        with ProcessPoolExecutor(max_workers=args.phase_workers) as ex:
            # No-FPL jobs
            for x0 in phase_ic_list:
                pack = (args.env, ctrl_nf_cpu, x_eq_t.cpu(), u_eq_t.cpu(),
                        x0, args.T, args.dt, args.w1, args.w2, args.eps)
                jobs.append(("nf", ex.submit(_analytical_phase_worker, pack)))
            # FPL jobs
            for x0 in phase_ic_list:
                pack = (args.env, ctrl_f_cpu, x_eq_t.cpu(), u_eq_t.cpu(),
                        x0, args.T, args.dt, args.w1, args.w2, args.eps)
                jobs.append(("f", ex.submit(_analytical_phase_worker, pack)))

            for tag, fut in tqdm(jobs, desc="Phase (Analytical, parallel)", leave=False):
                X = fut.result()
                i = len([1 for t, _ in jobs[:jobs.index((tag, fut))] if t == tag])  # approximate order
                i = min(i, len(colors)-1)
                c = colors[i]
                axA1.plot(X[:, 0], X[:, 1], ls="--" if tag == "nf" else "-", color=c, lw=2.0)
    else:
        for i, x0 in enumerate(tqdm(phase_ic_list, desc="Phase (Analytical, seq.)", leave=False)):
            c = colors[i]
            t, X, U, _, _ = viz.simulate_single_analytical(
                args.env, ctrl_nf_cpu, x_eq_t.cpu(), u_eq_t.cpu(),
                x0, args.T, args.dt, args.w1, args.w2, args.eps
            )
            axA1.plot(X[:, 0], X[:, 1], linestyle="--", color=c, linewidth=2.0)
            t, X, U, _, _ = viz.simulate_single_analytical(
                args.env, ctrl_f_cpu, x_eq_t.cpu(), u_eq_t.cpu(),
                x0, args.T, args.dt, args.w1, args.w2, args.eps
            )
            axA1.plot(X[:, 0], X[:, 1], linestyle="-", color=c, linewidth=2.0)

    # ---- ML (batched on GPU/CPU) ----
    dev_nf = next(ctrl_nf_gpu.parameters()).device
    dev_f  = next(ctrl_f_gpu.parameters()).device
    if len(phase_ic_list) > 0:
        Xs_nf = _phase_batch_ml(forward_system, ctrl_nf_gpu, x_eq_t.to(dev_nf), u_eq_t.to(dev_nf),
                                u_lo, u_up, phase_ic_list, args.T, args.dt, dev_nf,
                                desc="Phase (ML No-FPL, batched)")
        Xs_f  = _phase_batch_ml(forward_system, ctrl_f_gpu,  x_eq_t.to(dev_f),  u_eq_t.to(dev_f),
                                u_lo, u_up, phase_ic_list, args.T, args.dt, dev_f,
                                desc="Phase (ML FPL, batched)")
        for i, (Xnf, Xf) in enumerate(zip(Xs_nf, Xs_f)):
            c = colors[i]
            axA2.plot(Xnf[:, 0], Xnf[:, 1], ls="--", color=c, lw=2.0)
            axA2.plot(Xf[:, 0],  Xf[:, 1],  ls="-",  color=c, lw=2.0)

    # Cosmetics
    for ax in (axA1, axA2):
        viz.draw_verification_box(ax, x_lo, x_up, color="k", linestyle="--", linewidth=1.0)
        ax.scatter([float(x_eq_t[0])], [float(x_eq_t[1])], c="k", marker="x", s=60, label="$x_{eq}$")
        ax.set_xlabel(xl, fontsize=LABEL_FT); 
        if ax is axA1:
            ax.set_ylabel(yl, fontsize=LABEL_FT)
        ax.grid(True, alpha=0.3)
    axA1.set_title("Phase (Analytical)", fontsize=TITLE_FT)
    axA2.set_title("Phase (RELU dynamics)", fontsize=TITLE_FT)
    import matplotlib.lines as mlines
    leg = [mlines.Line2D([], [], color="k", ls="--", lw=2.0, label="No-FPL"),
           mlines.Line2D([], [], color="k", ls="-",  lw=2.0, label="FPL")]
    axA1.legend(handles=leg, loc="best", fontsize=LEGEND_FT)
    axA2.legend(handles=leg, loc="best", fontsize=LEGEND_FT)
    style_legend(axA1); style_legend(axA2)  # force in case Matplotlib overrides

    figA.tight_layout()
    save_or_show(figA, save_dir, f"{stem}_phase_diagrams", args.no_save)


# -----------------------------
# ROA (continuous colors, shared scale) — unchanged
# -----------------------------
def plot_roa_continuous(args, forward_system, x_eq_t, u_eq_t,
                        ctrl_nf_gpu, ctrl_f_gpu, x_lo, x_up, u_lo, u_up,
                        save_dir, stem):
    rng = np.random.default_rng(args.ic_seed)
    X0 = np.column_stack([
        rng.uniform(float(x_lo[0]), float(x_up[0]), size=args.roa_samples),
        rng.uniform(float(x_lo[1]), float(x_up[1]), size=args.roa_samples)
    ]).astype(np.float64)
    X0_t = torch.from_numpy(X0).to(torch.float64)

    dev_nf = next(ctrl_nf_gpu.parameters()).device
    dev_f  = next(ctrl_f_gpu.parameters()).device

    def run_one(controller, label, dev):
        conv, tconv = viz.simulate_batch_ml(
            args.env, forward_system, controller, x_eq_t.to(dev), u_eq_t.to(dev),
            u_lo, u_up, X0_t.to(dev), args.T, args.dt, args.w1, args.w2, args.eps, dev,
            stride=args.stride, progress=True, progress_desc=f"ROA steps ({label})"
        )
        return conv, tconv

    conv_nf, t_nf = run_one(ctrl_nf_gpu, "No-FPL", dev_nf)
    conv_f,  t_f  = run_one(ctrl_f_gpu,  "FPL",   dev_f)

    t_all = np.concatenate([t_nf[conv_nf], t_f[conv_f]]) if (np.any(conv_nf) or np.any(conv_f)) else np.array([0.0, 1.0])
    vmin, vmax = float(np.min(t_all)), float(np.max(t_all))

    base = cm.get_cmap('summer')  # green: fast
    cmap = ListedColormap(base(np.linspace(0.30, 1.00, 256)))

    xl, yl = axis_labels(args.env)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    for ax in (ax1, ax2):              # whatever you call them
        style_axes(ax)

    def panel(ax, conv, tconv, title):
        im = ax.scatter(X0[conv, 0], X0[conv, 1],
                        c=tconv[conv], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=14, edgecolor='none')
        ax.scatter(X0[~conv, 0], X0[~conv, 1], c='red', s=16, marker='x', label='fail')
        viz.draw_verification_box(ax, x_lo, x_up, color="k", linestyle="--", linewidth=1.0)
        ax.scatter([float(x_eq_t[0])], [float(x_eq_t[1])], c="k", marker="x", s=60)
        ax.set_title(title, fontsize=TITLE_FT)
        ax.set_xlabel(xl, fontsize=LABEL_FT)
        if ax is ax1:
            ax.set_ylabel(yl, fontsize=LABEL_FT)
        ax.grid(True, alpha=0.3)
        style_axes(ax)  # ensures labels/ticks stay large too

        return im

    im1 = panel(ax1, conv_nf, t_nf, "Settling Time (No-FPL)")
    im2 = panel(ax2, conv_f,  t_f,  "Settling Time (FPL)")

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(im2, cax=cax)
    style_colorbar(cb)  # enforce label/tick sizes
    cb.set_label(r"Settling time $t_\epsilon$ (s)")
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.label.set_size(22)


    for ax, conv, tconv in ((ax1, conv_nf, t_nf), (ax2, conv_f, t_f)):
        rate = float(np.mean(conv))
        mean_t = float(np.mean(tconv[conv])) if np.any(conv) else float("nan")
        ax.text(0.02, 0.98, f"rate={rate*100:.1f}%,  mean t={mean_t:.2f}s",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=20, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    fig.tight_layout()
    save_or_show(fig, save_dir, f"{stem}_roa_continuous", args.no_save)


# -----------------------------
# Effort map (batched) — unchanged
# -----------------------------
@torch.no_grad()
def plot_effort_map(args, forward_system, x_eq_t, u_eq_t,
                    ctrl_nf_gpu, ctrl_f_gpu, x_lo, x_up, u_lo, u_up,
                    save_dir, stem):
    rng = np.random.default_rng(args.ic_seed)
    X0 = np.column_stack([
        rng.uniform(float(x_lo[0]), float(x_up[0]), size=args.ic_effort),
        rng.uniform(float(x_lo[1]), float(x_up[1]), size=args.ic_effort)
    ]).astype(np.float64)

    def run_eff(controller):
        dev = next(controller.parameters()).device
        steps = int(np.ceil(args.T / args.dt))
        X = torch.from_numpy(X0).to(device=dev, dtype=torch.float64)
        xeq_X = x_eq_t.to(device=dev, dtype=torch.float64)
        c_dtype = next(controller.parameters()).dtype
        xeq_C = x_eq_t.to(device=dev, dtype=c_dtype)
        ueq_C = u_eq_t.to(device=dev, dtype=c_dtype)

        if np.isscalar(u_lo):
            ulo_t = torch.tensor(float(u_lo), device=dev, dtype=c_dtype)
            uup_t = torch.tensor(float(u_up), device=dev, dtype=c_dtype)
            clamp = lambda u: torch.clamp(u, ulo_t, uup_t)
        else:
            ulo_t = torch.as_tensor(u_lo, device=dev, dtype=c_dtype)
            uup_t = torch.as_tensor(u_up, device=dev, dtype=c_dtype)
            clamp = lambda u: torch.max(torch.min(u, uup_t), ulo_t)

        B = X.shape[0]
        effort = torch.zeros(B, device=dev, dtype=torch.float64)
        done = torch.zeros(B, device=dev, dtype=torch.bool)
        w = torch.tensor([args.w1, args.w2], device=dev, dtype=torch.float64)

        for _ in tqdm(range(steps), desc="Effort steps", leave=False):
            Xc = X.to(dtype=c_dtype)
            u = clamp(controller(Xc) - controller(xeq_C) + ueq_C).to(dtype=torch.float64)
            inc = (u - ueq_C.to(dtype=torch.float64)).abs()
            if inc.ndim > 1:
                inc = inc.sum(dim=-1)
            effort = effort + torch.where(done, torch.zeros_like(inc), inc)
            X = forward_system.step_forward(X, u)
            dist = torch.sum(w * torch.abs(X - xeq_X), dim=-1)
            done = done | (dist <= args.eps)
            if bool(done.all()):
                break

        return effort.cpu().numpy(), (~done).cpu().numpy()

    eff_nf, fail_nf = run_eff(ctrl_nf_gpu)
    eff_f,  fail_f  = run_eff(ctrl_f_gpu)

    good_nf, good_f = ~fail_nf, ~fail_f
    eff_all = np.concatenate([eff_nf[good_nf], eff_f[good_f]]) if (good_nf.any() or good_f.any()) else np.array([0.0, 1.0])
    vmin, vmax = float(np.min(eff_all)), float(np.max(eff_all))

    base = cm.get_cmap('summer')
    cmap = ListedColormap(base(np.linspace(0.30, 1.00, 256)))

    xl, yl = axis_labels(args.env)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)
    for ax in (ax1, ax2):
        style_axes(ax)

    def panel(ax, good, eff, title):
        im = ax.scatter(X0[good, 0], X0[good, 1], c=eff[good], cmap=cmap, vmin=vmin, vmax=vmax,
                        s=14, edgecolor='none')
        ax.scatter(X0[~good, 0], X0[~good, 1], c='red', s=16, marker='x')
        viz.draw_verification_box(ax, x_lo, x_up, color="k", linestyle="--", linewidth=1.0)
        ax.scatter([float(x_eq_t[0])], [float(x_eq_t[1])], c="k", marker="x", s=60)
        ax.set_title(title, fontsize=TITLE_FT)
        ax.set_xlabel(xl, fontsize=LABEL_FT)
        # set y label only on left plot for symmetry
        if ax is ax1:
            ax.set_ylabel(yl, fontsize=LABEL_FT)
        ax.grid(True, alpha=0.3)
        style_axes(ax)

        return im

    im1 = panel(ax1, good_nf, eff_nf, "Control effort map (No-FPL)")
    im2 = panel(ax2, good_f,  eff_f,  "Control effort map (FPL)")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cb = fig.colorbar(im2, cax=cax)
    style_colorbar(cb)

    cb.set_label(r"Total effort $\sum_t \|u(t)-u_{eq}\|_1$")
    cb.ax.tick_params(labelsize=20)
    cb.ax.yaxis.label.set_size(22)

    for ax, good, eff in ((ax1, good_nf, eff_nf), (ax2, good_f, eff_f)):
        mean_e = float(np.mean(eff[good])) if good.any() else float("nan")
        ax.text(0.02, 0.98, f"mean effort={mean_e:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=20, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    fig.tight_layout()
    save_or_show(fig, save_dir, f"{stem}_effort_map", args.no_save)


# -----------------------------
# Lyapunov surfaces (+ optional GIF) — unchanged
# -----------------------------
@torch.no_grad()
def plot_lyapunov_surfaces(args, env, x_eq_t, ly_nf, R_nf, ly_f, R_f, save_dir, stem, grid_n=75):
    if env == "pendulum":
        x1 = np.linspace(float(x_eq_t[0]) - np.pi, float(x_eq_t[0]) + np.pi, grid_n)
        x2 = np.linspace(-5.0, 5.0, grid_n)
    else:
        x1 = np.linspace(-0.8, 0.8, grid_n)
        x2 = np.linspace(-0.8, 0.8, grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    Xgrid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    device0 = next((p.device for p in ly_nf.parameters()), torch.device("cpu"))

    V_nf = viz.V_batch(torch.from_numpy(Xgrid).to(device0), ly_nf, x_eq_t.to(device0), 1.0, R_nf, device0).cpu().numpy()
    V_f  = viz.V_batch(torch.from_numpy(Xgrid).to(device0), ly_f,  x_eq_t.to(device0), 1.0, R_f,  device0).cpu().numpy()
    V_nf = V_nf.reshape(X1.shape); V_f = V_f.reshape(X1.shape)

    xl, yl = axis_labels(env)
    fig = plt.figure(figsize=(12.5, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for ax, V, title in [(ax1, V_nf, "Lyapunov surface (No-FPL)"),
                         (ax2, V_f,  "Lyapunov surface (FPL)")]:
        surf = ax.plot_surface(X1, X2, V, rstride=2, cstride=2, cmap="viridis",
                               linewidth=0, antialiased=True, alpha=0.95)
        z0 = float(np.percentile(V, 5.0))
        ax.contour(X1, X2, V, zdir='z', offset=z0, cmap="viridis", levels=18, linewidths=0.8)
        ax.set_zlim(z0, float(np.max(V)))
        ax.set_xlabel(xl, fontsize=LABEL_FT)
        if ax is ax1:
            ax.set_ylabel(yl, fontsize=LABEL_FT)
        ax.set_zlabel("$V(x)$", fontsize=LABEL_FT)
        ax.set_title(title, fontsize=TITLE_FT)
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    fig.tight_layout()
    save_or_show(fig, save_dir, f"{stem}_V_surfaces", args.no_save)

    if args.gif and not args.no_save:
        fig_gif = plt.figure(figsize=(12.5, 5.4))
        axg1 = fig_gif.add_subplot(1, 2, 1, projection="3d")
        axg2 = fig_gif.add_subplot(1, 2, 2, projection="3d")
        surf1 = axg1.plot_surface(X1, X2, V_nf, rstride=2, cstride=2, cmap="viridis", linewidth=0, alpha=0.95)
        z01 = float(np.percentile(V_nf, 5.0)); axg1.contour(X1, X2, V_nf, zdir='z', offset=z01, cmap="viridis", levels=18, linewidths=0.8)
        axg1.set_zlim(z01, float(np.max(V_nf))); axg1.set_title("No-FPL", fontsize=TITLE_FT)
        axg1.set_xlabel(xl, fontsize=LABEL_FT); axg1.set_ylabel(yl, fontsize=LABEL_FT); axg1.set_zlabel("$V(x)$", fontsize=LABEL_FT)
        surf2 = axg2.plot_surface(X1, X2, V_f, rstride=2, cstride=2, cmap="viridis", linewidth=0, alpha=0.95)
        z02 = float(np.percentile(V_f, 5.0)); axg2.contour(X1, X2, V_f, zdir='z', offset=z02, cmap="viridis", levels=18, linewidths=0.8)
        axg2.set_zlim(z02, float(np.max(V_f))); axg2.set_title("FPL", fontsize=TITLE_FT)
        axg2.set_xlabel(xl, fontsize=LABEL_FT); axg2.set_ylabel(yl, fontsize=LABEL_FT); axg2.set_zlabel("$V(x)$", fontsize=LABEL_FT)

        def update(frame):
            az = 30 + frame
            axg1.view_init(elev=30, azim=az)
            axg2.view_init(elev=30, azim=az)
            return []

        anim = FuncAnimation(fig_gif, update, frames=120, interval=50, blit=False)
        gif_path = os.path.join(save_dir, f"{stem}_V_surfaces.gif")
        anim.save(gif_path, writer=PillowWriter(fps=20))
        plt.close(fig_gif)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["pendulum", "unicycle"], required=True)
    p.add_argument("--bound_level", type=int, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")

    p.add_argument("--T", type=float, default=25.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=1.0)

    p.add_argument("--ic_phase", type=int, default=8)
    p.add_argument("--sample_mode_phase", choices=["random", "grid"], default="random")
    p.add_argument("--phase_workers", type=int, default=0, help=">0 to enable CPU multiprocessing for analytical phase.")
    p.add_argument("--ic_seed", type=int, default=0)

    p.add_argument("--roa_samples", type=int, default=2000)

    p.add_argument("--ic_effort", type=int, default=2000)
    p.add_argument("--sample_mode_effort", choices=["random", "grid"], default="random")

    p.add_argument("--save_dir", type=str, default="paper_figs/minimal/")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--gif", action="store_true")

    args = p.parse_args()

    set_default_dtype(args.precision)
    device = setup_device(args.gpu)
    device_cpu = torch.device("cpu")

    # Load forward model + bounds
    dynamics_model, forward_system, x_lo_t, x_up_t, u_lo, u_up, x_eq_t, u_eq_t = viz.load_forward_system(
        args.env, device, args.dt
    )
    x_lo = x_lo_t.cpu().numpy(); x_up = x_up_t.cpu().numpy()

    # Load models
    spec_nf = viz.RunSpec(env=args.env, bound_level=args.bound_level, fpl=False)
    spec_f  = viz.RunSpec(env=args.env, bound_level=args.bound_level, fpl=True)
    ctrl_nf_gpu, ly_nf_gpu, R_nf_gpu = viz.load_models(spec_nf, device=device)
    ctrl_f_gpu,  ly_f_gpu,  R_f_gpu  = viz.load_models(spec_f,  device=device)
    ctrl_nf_cpu, ly_nf_cpu, R_nf_cpu = viz.load_models(spec_nf, device=device_cpu)
    ctrl_f_cpu,  ly_f_cpu,  R_f_cpu  = viz.load_models(spec_f,  device=device_cpu)

    rng = np.random.default_rng(args.ic_seed)
    phase_ic_list = sample_ic_list(args.ic_phase, args.sample_mode_phase, x_lo, x_up, rng)
    stem = f"{args.env}_bound{args.bound_level}"

    # 1) Phase diagrams (parallel/batched)
    plot_phase_diagrams(args, forward_system, x_eq_t, u_eq_t,
                        ctrl_nf_cpu, ctrl_f_cpu, ctrl_nf_gpu, ctrl_f_gpu,
                        x_lo, x_up, u_lo, u_up, phase_ic_list, args.save_dir, stem)

    # 2) ROA (continuous, shared colorbar)
    plot_roa_continuous(args, forward_system, x_eq_t, u_eq_t,
                        ctrl_nf_gpu, ctrl_f_gpu, x_lo, x_up, u_lo, u_up,
                        args.save_dir, stem)

    # 3) Effort map (batched)
    plot_effort_map(args, forward_system, x_eq_t, u_eq_t,
                    ctrl_nf_gpu, ctrl_f_gpu, x_lo, x_up, u_lo, u_up,
                    args.save_dir, stem)

    # 4) Lyapunov surfaces (+ GIF)
    plot_lyapunov_surfaces(args, args.env, x_eq_t,
                           ly_nf_cpu, R_nf_cpu, ly_f_cpu, R_f_cpu,
                           args.save_dir, stem, grid_n=75)
