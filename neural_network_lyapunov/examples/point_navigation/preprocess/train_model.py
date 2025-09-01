#!/usr/bin/env python3
# Train a ReLU (piecewise-linear) next-state predictor φ for a Unicycle env.
# Loss uses the equilibrium-shifted prediction: x_hat^+ = φ(z) - φ(z_eq) + x_eq
# Saves ONLY φ (plain nn.Sequential of Linear/ReLU/.../Linear) as a .pt file.

import argparse, math, random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.spaces import Box

# --------------------- Utils ---------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_to_pi(th: torch.Tensor) -> torch.Tensor:
    return (th + math.pi) % (2.0 * math.pi) - math.pi


def to_t(x):
    return torch.as_tensor(x, dtype=torch.float64)


# --------------------- Unicycle Env ---------------------


class UnicycleEnv(gym.Env):
    """
    State:  s = [x, y, theta]
    Action: a = [v, omega]
    Dynamics (Euler):
      x_{k+1} = x_k + v*dt*cos(theta)
      y_{k+1} = y_k + v*dt*sin(theta)
      th_{k+1} = wrap(theta + omega*dt)
    """

    metadata = {"render_modes": []}

    def __init__(
        self, dt=0.1, xlim=1.0, v_max=10.0, w_max=math.pi, seed: Optional[int] = None
    ):
        super().__init__()
        self.dt = float(dt)
        self.rng = np.random.default_rng(seed)
        self.observation_space = Box(
            low=np.array([-xlim, -xlim, -math.pi], dtype=np.float64),
            high=np.array([+xlim, +xlim, +math.pi], dtype=np.float64),
            dtype=np.float64,
        )
        self.action_space = Box(
            low=np.array([-v_max, -w_max], dtype=np.float64),
            high=np.array([+v_max, +w_max], dtype=np.float64),
            dtype=np.float64,
        )
        self.state = None

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        x = self.rng.uniform(-1.0, 1.0)
        y = self.rng.uniform(-1.0, 1.0)
        th = self.rng.uniform(-math.pi, math.pi)
        self.state = np.array([x, y, th], dtype=np.float64)
        return self.state.copy(), {}

    def step(self, action):
        assert self.state is not None
        a = np.clip(action, self.action_space.low, self.action_space.high)
        x, y, th = self.state
        v, w = a
        x_next = x + v * self.dt * math.cos(th)
        y_next = y + v * self.dt * math.sin(th)
        th_next = (th + w * self.dt + math.pi) % (2.0 * math.pi) - math.pi
        self.state = np.array([x_next, y_next, th_next], dtype=np.float64)
        # No terminal condition; return dummy reward/done flags for completeness
        return self.state.copy(), 0.0, False, False, {}


# --------------------- Model (ReLU = PWL) ---------------------


def make_relu_mlp(in_dim: int, hidden: Tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        d = h
    layers += [nn.Linear(d, out_dim)]
    m = nn.Sequential(*layers).double()
    for mod in m:
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            nn.init.zeros_(mod.bias)
    return m  # piecewise-linear by construction


# --------------------- Data stream ---------------------


def random_policy(state: np.ndarray, action_space: gym.spaces.Space):
    # Uniform in action bounds
    return action_space.sample()


def rollout_stream(env: gym.Env, episode_len: int, policy=random_policy):
    obs, _ = env.reset()
    t = 0
    while True:
        a = policy(obs, env.action_space)
        s = obs
        obs, _, done, trunc, _ = env.step(a)
        yield {"state": s, "action": a, "next_state": obs}
        t += 1
        if done or trunc or t >= episode_len:
            obs, _ = env.reset()
            t = 0


def batch_stream(env: gym.Env, batch_size: int, steps: int, episode_len: int):
    gen = rollout_stream(env, episode_len)
    S = env.observation_space.shape[0]
    A = env.action_space.shape[0]
    while True:
        states = np.zeros((batch_size * steps, S), dtype=np.float64)
        actions = np.zeros((batch_size * steps, A), dtype=np.float64)
        nexts = np.zeros((batch_size * steps, S), dtype=np.float64)
        for i in range(batch_size * steps):
            item = next(gen)
            states[i] = item["state"]
            actions[i] = item["action"]
            nexts[i] = item["next_state"]
        yield {
            "state": to_t(states),
            "action": to_t(actions),
            "next_state": to_t(nexts),
        }


# --------------------- Training (equilibrium-shifted loss) ---------------------


def predict_with_equilibrium(
    phi: nn.Module,
    state: torch.Tensor,
    action: torch.Tensor,
    x_eq: torch.Tensor,
    u_eq: torch.Tensor,
    latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if latent is None:
        z = torch.cat([state, action], dim=-1)  # [N, 5]
        z_eq = torch.cat([x_eq.expand_as(state), u_eq.expand_as(action)], dim=-1)
    else:
        z = torch.cat([state, action, latent], dim=-1)  # [N, 5+L]
        z_eq = torch.cat(
            [x_eq.expand_as(state), u_eq.expand_as(action), torch.zeros_like(latent)],
            dim=-1,
        )
    return phi(z) - phi(z_eq) + x_eq  # shifted prediction


def train(args):
    set_seed(args.seed)

    # 1) Environment (included here; self-contained)
    env = UnicycleEnv(
        dt=args.dt, xlim=args.xlim, v_max=args.vmax, w_max=args.wmax, seed=args.seed
    )

    # 2) Generator φ: ReLU (PWL)
    hidden = tuple(int(s) for s in args.hidden.split(",")) if args.hidden else (64, 64)
    in_dim = 3 + 2 + (args.latent if args.latent > 0 else 0)  # [x,y,θ] + [v,ω] + latent
    phi = make_relu_mlp(in_dim, hidden, out_dim=3)

    # 3) Equilibrium (paper uses shifted prediction in the loss)
    x_eq = (
        torch.zeros(1, 3, dtype=torch.float64)
        if args.eq_zero
        else to_t([0.0, 0.0, 0.0]).unsqueeze(0)
    )
    u_eq = (
        torch.zeros(1, 2, dtype=torch.float64)
        if args.eq_zero
        else to_t([0.0, 0.0]).unsqueeze(0)
    )

    # 4) Optimizer
    # We add a scheduled lr based on loss update
    opt = torch.optim.Adam(phi.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    # 5) Data
    stream = batch_stream(env, args.batch, args.steps_per_iter, args.episodesize)

    # 6) Train
    for ep in range(1, args.epochs + 1):
        running = 0.0
        for _ in range(args.iters_per_epoch):
            batch = next(stream)
            s = batch["state"]  # [B*steps, 3]
            a = batch["action"]  # [B*steps, 2]
            ns = batch["next_state"]  # [B*steps, 3]
            latent = (
                torch.randn(s.size(0), args.latent, dtype=torch.float64)
                if args.latent > 0
                else None
            )

            pred = predict_with_equilibrium(phi, s, a, x_eq, u_eq, latent)
            loss = F.mse_loss(pred, ns)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(phi.parameters(), args.clip)
            opt.step()
            running += loss.item()
        print(f"epoch {ep:03d} | loss={running/args.iters_per_epoch:.4e}")

    # 7) Save ONLY φ (plain nn.Sequential of Linear/ReLU/.../Linear)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(phi, str(args.out))
    print(f"[OK] Saved generator φ to: {args.out.resolve()}")


# --------------------- CLI ---------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Env / data
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--xlim", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=10.0)
    ap.add_argument("--wmax", type=float, default=math.pi)
    ap.add_argument("--episodesize", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument(
        "--steps_per_iter",
        type=int,
        default=1,
        help="#steps each sample advances between env calls",
    )
    # Model / training
    ap.add_argument(
        "--hidden",
        type=str,
        default="8,8",
        help="comma-separated widths (e.g., 8,8 for 5-8-8-3)",
    )
    ap.add_argument(
        "--latent", type=int, default=0, help="extra latent dims appended to [x,u]"
    )
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters_per_epoch", type=int, default=300)
    ap.add_argument(
        "--eq_zero", action="store_true", help="use x*=0,u*=0 as equilibrium"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "neural_network_lyapunov/examples/point_navigation/data/preprocess/point_nav_forward_model.pt"
        ),
    )
    args = ap.parse_args()
    train(args)
