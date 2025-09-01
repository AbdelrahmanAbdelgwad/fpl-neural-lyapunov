#!/usr/bin/env python3
# neural_network_lyapunov/examples/point_navigation/preprocess/build_pwl_unicycle.py
# Build a TorchScript model φ([x,y,θ,v,ω]) -> x_next using PWL sin/cos.
# No training. Saved .pt loads with torch.load(...) (it auto-dispatches to jit).

import math
import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


def _wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    return (
        theta + math.pi - torch.floor((theta + math.pi) / two_pi) * two_pi
    ) - math.pi


class PWL1DPeriodic(nn.Module):
    """Uniform periodic linear spline on [-pi, pi], extended periodically."""

    def __init__(self, y_knots: torch.Tensor):
        super().__init__()
        assert y_knots.dim() == 1 and y_knots.numel() >= 2
        K = y_knots.numel() - 1
        thetas = torch.linspace(-math.pi, math.pi, steps=K + 1, dtype=y_knots.dtype)
        if not torch.isclose(y_knots[0], y_knots[-1]):
            raise ValueError("y_knots[-1] must equal y_knots[0] for periodicity.")
        self.K = K
        self.register_buffer("theta_knots", thetas)
        self.register_buffer("y_knots", y_knots)
        slopes = (y_knots[1:] - y_knots[:-1]) / (thetas[1:] - thetas[:-1])
        self.register_buffer("slopes", slopes)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        # Works for scalar, 1D or ND inputs; returns same shape
        dtype = self.theta_knots.dtype
        device = self.theta_knots.device
        x = theta.to(dtype=dtype, device=device)
        x = _wrap_to_pi(x)
        idx = torch.bucketize(x, self.theta_knots[1:])  # 0..K-1, shape(x)
        theta_left = self.theta_knots[idx]
        y_left = self.y_knots[idx]
        slope = self.slopes[idx]
        return y_left + slope * (x - theta_left)


class TrigPWL(nn.Module):
    """(sin, cos) via PWL interpolants."""

    def __init__(self, K: int = 128, dtype: torch.dtype = torch.float64):
        super().__init__()
        th = torch.linspace(-math.pi, math.pi, steps=K + 1, dtype=dtype)
        s = torch.sin(th.clone())
        s[-1] = s[0]
        c = torch.cos(th.clone())
        c[-1] = c[0]
        self.sin_pwl = PWL1DPeriodic(s)
        self.cos_pwl = PWL1DPeriodic(c)

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sin_pwl(theta), self.cos_pwl(theta)


class PhiConcatUnicyclePWL(nn.Module):
    """
    φ([x,y,θ,v,ω]) -> [x^+, y^+, θ^+], using PWL sin/cos. No training.
    Returns shape (3,) for 1D input (5,), or (N,3) for (N,5).
    """

    def __init__(
        self, dt: float = 0.1, K: int = 128, dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        self.dt = float(dt)
        self.trig = TrigPWL(K=K, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 1:
            # 1D input -> 1D output
            x0, y0, th, v, w = z[0], z[1], z[2], z[3], z[4]
            s, c = self.trig(th)
            px_next = x0 + v * self.dt * c
            py_next = y0 + v * self.dt * s
            th_next = th + w * self.dt  # teleop wraps downstream
            return torch.stack((px_next, py_next, th_next))
        else:
            # Batched -> batched
            x = z[..., :3]
            u = z[..., 3:5]
            th = x[..., 2]
            v = u[..., 0]
            w = u[..., 1]
            s, c = self.trig(th)
            px_next = x[..., 0] + v * self.dt * c
            py_next = x[..., 1] + v * self.dt * s
            th_next = th + w * self.dt
            return torch.stack((px_next, py_next, th_next), dim=-1)


def eval_trig_err(K=128, dtype=torch.float64, num=20001):
    t = torch.linspace(-math.pi, math.pi, steps=num, dtype=dtype)
    s_true, c_true = torch.sin(t), torch.cos(t)
    trig = TrigPWL(K=K, dtype=dtype)
    s_hat, c_hat = trig(t)
    return (s_true - s_hat).abs().max().item(), (c_true - c_hat).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", "-K", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument(
        "--dtype", type=str, default="float64", choices=["float32", "float64"]
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .pt (TorchScript). Your teleop's torch.load will auto-load it.",
    )
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    model = PhiConcatUnicyclePWL(dt=args.dt, K=args.segments, dtype=dtype).eval()

    s_err, c_err = eval_trig_err(K=args.segments, dtype=dtype)
    print(
        f"[Diag] K={args.segments} → max|sin err|≈{s_err:.3e}, max|cos err|≈{c_err:.3e}"
    )

    scripted = torch.jit.script(model)  # no pickle issues
    scripted.save(str(args.out))
    print(f"[OK] TorchScript saved to: {args.out.resolve()}")

    # 1D sanity (should be ~[+0.0707, +0.0707, +0.7854] for θ=π/4, v=1, ω=0, dt=0.1)
    with torch.no_grad():
        z = torch.tensor([0.0, 0.0, 0.25 * math.pi, 1.0, 0.0], dtype=dtype)
        print("[Demo] x_next:", scripted(z))


if __name__ == "__main__":
    main()
