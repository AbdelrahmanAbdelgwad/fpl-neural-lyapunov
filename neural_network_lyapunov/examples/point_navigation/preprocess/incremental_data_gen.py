#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental data generation + training (fast version)

Key improvements vs your original:
  • Vectorized closed-form unicycle step (no solve_ivp).
  • MultiFileDataset lookup uses binary search (O(log F)).
  • DataLoader avoids cross-file shuffling by default; uses persistent workers + prefetch.
  • Single cached equilibrium forward per epoch (no redundant per-batch forward).
  • Paths unified via --data_dir (used by both generation and training).

You can reproduce your runs:

  # Generate 1,000 × 10,000 (but consider fewer, larger files for speed)
  python3 incremental_data_gen.py --generate_chunks --num_chunks 1000 --samples_per_chunk 10000

  # Train on multi-file dataset
  python3 incremental_data_gen.py --train_multifile

Recommended (same total samples, far fewer files):
  python3 incremental_data_gen.py --generate_chunks --num_chunks 100 --samples_per_chunk 100000
"""
import os
import glob
import time
import math
import argparse
import bisect
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Unicycle closed-form dynamics
# ---------------------------
@torch.no_grad()
def unicycle_step(
    x: torch.Tensor, u: torch.Tensor, dt: float, eps: float = 1e-8
) -> torch.Tensor:
    """
    Closed-form one-step integration of the unicycle model under constant control over dt.

    State x = [x, y, theta]
    Control u = [v, omega]

    Args:
      x: (..., 3) tensor (float64 recommended)
      u: (..., 2) tensor
      dt: step size (float)
      eps: small threshold to switch straight-line approximation

    Returns:
      x_next: (..., 3) tensor
    """
    assert x.shape[-1] == 3 and u.shape[-1] == 2, "x must be (...,3), u must be (...,2)"
    x = x.clone()
    u = u.clone()
    dtype = x.dtype

    X, Y, TH = x[..., 0], x[..., 1], x[..., 2]
    V, OM = u[..., 0], u[..., 1]

    TH_next = TH + OM * dt

    # Use piecewise formula to avoid division by zero at small |omega|
    straight = OM.abs() <= eps
    curved = ~straight

    # Allocate outputs
    X_next = torch.empty_like(X, dtype=dtype)
    Y_next = torch.empty_like(Y, dtype=dtype)

    if straight.any():
        # Straight-line motion
        Vdt = V[straight] * dt
        c = torch.cos(TH[straight])
        s = torch.sin(TH[straight])
        X_next[straight] = X[straight] + Vdt * c
        Y_next[straight] = Y[straight] + Vdt * s

    if curved.any():
        # Circular arc
        omg = OM[curved]
        v_over_omg = V[curved] / omg
        THp = TH_next[curved]
        TH0 = TH[curved]
        X_next[curved] = X[curved] + v_over_omg * (torch.sin(THp) - torch.sin(TH0))
        Y_next[curved] = Y[curved] - v_over_omg * (torch.cos(THp) - torch.cos(TH0))

    return torch.stack((X_next, Y_next, TH_next), dim=-1)


# ---------------------------
# Data generation
# ---------------------------
def generate_dynamics_data_chunk(
    dt: float,
    num_samples: int = 10000,
    chunk_id: int = 0,
    seed: int = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one chunk of (x,u)->x' samples using vectorized closed-form step.
    Returns:
      dataset_input:  (N, 5)  [x, y, theta, v, omega]
      dataset_output: (N, 3)  [x', y', theta']
    """
    if seed is not None:
        # Make chunks different but still reproducible
        torch.manual_seed(seed + chunk_id)
        np.random.seed(seed + chunk_id)

    # Ranges (match your prior script)
    x_range = (-2.0, 2.0)
    y_range = (-2.0, 2.0)
    theta_range = (-math.pi, math.pi)
    v_range = (-10.0, 10.0)
    omega_range = (-10.0, 10.0)

    # Sample states and controls (vectorized)
    # Use utils.uniform_sample_in_box if available to keep parity; otherwise fall back.
    try:
        import neural_network_lyapunov.utils as utils

        def _to_lastdim(t, d):
            # ensure the last dimension is the feature dimension d
            return t if t.shape[-1] == d else t.T

        x_samples = _to_lastdim(
            utils.uniform_sample_in_box(
                torch.tensor([x_range[0], y_range[0], theta_range[0]], dtype=dtype),
                torch.tensor([x_range[1], y_range[1], theta_range[1]], dtype=dtype),
                num_samples,
            ),
            3,
        )

        u_samples = _to_lastdim(
            utils.uniform_sample_in_box(
                torch.tensor([v_range[0], omega_range[0]], dtype=dtype),
                torch.tensor([v_range[1], omega_range[1]], dtype=dtype),
                num_samples,
            ),
            2,
        )

    except Exception:
        # Fallback (pure torch)
        x_low = torch.tensor([x_range[0], y_range[0], theta_range[0]], dtype=dtype)
        x_high = torch.tensor([x_range[1], y_range[1], theta_range[1]], dtype=dtype)
        u_low = torch.tensor([v_range[0], omega_range[0]], dtype=dtype)
        u_high = torch.tensor([v_range[1], omega_range[1]], dtype=dtype)
        x_samples = x_low + (x_high - x_low) * torch.rand((num_samples, 3), dtype=dtype)
        u_samples = u_low + (u_high - u_low) * torch.rand((num_samples, 2), dtype=dtype)

    # Advance one step
    x_next = unicycle_step(x_samples, u_samples, dt)

    # Build dataset tensors
    dataset_input = torch.cat((x_samples, u_samples), dim=1)  # (N,5)
    dataset_output = x_next  # (N,3)
    return dataset_input, dataset_output


def generate_multiple_data_files(
    save_dir: str,
    dt: float,
    num_chunks: int,
    samples_per_chunk: int,
    seed: int = 42,
    prefix: str = "chunk",
    dtype: torch.dtype = torch.float64,
) -> None:
    """Generate many .pt files, each containing a dict {'input':(N,5), 'output':(N,3)}."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating {num_chunks} chunks × {samples_per_chunk} samples to {save_dir}")

    for i in range(num_chunks):
        t0 = time.time()
        inputs, outputs = generate_dynamics_data_chunk(
            dt=dt, num_samples=samples_per_chunk, chunk_id=i, seed=seed, dtype=dtype
        )
        # Ensure dtype is consistent
        inputs = inputs.to(dtype)
        outputs = outputs.to(dtype)

        path = os.path.join(save_dir, f"{prefix}_{i:04d}.pt")
        torch.save({"input": inputs, "output": outputs}, path)
        dt_ms = (time.time() - t0) * 1e3
        print(f"  wrote {path}  [{inputs.shape[0]} samples]  ({dt_ms:.0f} ms)")


# ---------------------------
# Dataset that streams multi-file sets
# ---------------------------
class MultiFileDataset(Dataset):
    """Dataset that loads from multiple .pt files on-the-fly and caches the current file."""

    def __init__(
        self,
        data_dir: str,
        pattern: str = "dynamics_chunk_*.pt",
        load_all: bool = False,
        dtype=torch.float64,
    ):
        super().__init__()
        self.dtype = dtype

        self.files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if not self.files:
            raise ValueError(
                f"No files found in '{data_dir}' matching pattern '{pattern}'"
            )
        print(f"Found {len(self.files)} data files in {data_dir}")

        self.load_all = load_all
        self.file_sizes = []
        self.cumulative_sizes = [0]  # prefix sums

        if load_all:
            # Load everything into memory (fastest at train time if RAM allows)
            print("Loading all data into memory...")
            all_inputs, all_outputs = [], []
            for f in self.files:
                d = torch.load(f)
                all_inputs.append(d["input"].to(self.dtype))
                all_outputs.append(d["output"].to(self.dtype))
            self.all_inputs = torch.cat(all_inputs, dim=0)
            self.all_outputs = torch.cat(all_outputs, dim=0)
            self.total_size = self.all_inputs.shape[0]
        else:
            # Only stat sizes to prepare an index map
            for f in self.files:
                d = torch.load(f)
                n = d["input"].shape[0]
                self.file_sizes.append(n)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + n)
            self.total_size = self.cumulative_sizes[-1]
            self.current_file_idx = -1
            self.current_data = None

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx: int):
        if self.load_all:
            return self.all_inputs[idx].to(self.dtype), self.all_outputs[idx].to(
                self.dtype
            )

        # O(log F) file lookup
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        if file_idx < 0 or file_idx >= len(self.files):
            raise IndexError(idx)

        if file_idx != getattr(self, "current_file_idx", -1):
            # swap cache
            self.current_data = torch.load(self.files[file_idx])
            self.current_file_idx = file_idx

        local_idx = idx - self.cumulative_sizes[file_idx]
        x = self.current_data["input"][local_idx].to(self.dtype)
        y = self.current_data["output"][local_idx].to(self.dtype)
        return x, y


# ---------------------------
# Training on the multi-file dataset
# ---------------------------
def train_forward_model_multifile(
    dynamics_model: torch.nn.Module,
    data_dir: str,
    *,
    num_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    load_all_data: bool = False,
    shuffle_across_files: bool = False,
    num_workers: int = 2,
    prefetch_factor: int = 4,
    dtype: torch.dtype = torch.float64,
    model_save_path: str = None,  # Add this parameter
) -> torch.nn.Module:
    """
    Train the forward model f(x,u) to predict x' on a dataset that spans many .pt files.
    """
    import datetime as _dt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dynamics_model = dynamics_model.to(device=device, dtype=dtype)

    dataset = MultiFileDataset(
        data_dir=data_dir,
        pattern="dynamics_chunk_*.pt",
        load_all=load_all_data,
        dtype=dtype,
    )

    # DataLoader tuned to minimize cross-file thrashing
    if load_all_data:
        num_workers = 0  # unnecessary when everything is in RAM

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(shuffle_across_files),  # default False to keep locality
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    # Equilibrium input and its forward pass (cached per epoch)
    x_eq = torch.zeros(3, dtype=dtype, device=device)
    u_eq = torch.zeros(2, dtype=dtype, device=device)
    eq_input = torch.cat((x_eq, u_eq), dim=0).unsqueeze(0)  # (1,5)

    print("\n" + "=" * 72)
    print("Training configuration")
    print("=" * 72)
    print(f"  device             : {device}")
    print(f"  epochs             : {num_epochs}")
    print(f"  batch_size         : {batch_size}")
    print(f"  lr                 : {lr}")
    print(f"  load_all_data      : {load_all_data}")
    print(f"  shuffle_across_files: {shuffle_across_files}")
    print(f"  num_workers        : {num_workers}")
    print(f"  prefetch_factor    : {prefetch_factor if num_workers>0 else 'N/A'}")
    print("=" * 72)

    best_epoch_loss = float("inf")
    best_model_state = None
    t_train0 = time.time()

    for epoch in range(num_epochs):
        dynamics_model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        # REMOVE the cached eq_out here

        for b, (xin, xnext) in enumerate(dataloader):
            xin = xin.to(device=device, dtype=dtype)
            xnext = xnext.to(device=device, dtype=dtype)

            # Compute eq_out fresh for current weights
            with torch.no_grad():
                eq_out = dynamics_model(eq_input).squeeze(0)

            pred = dynamics_model(xin) - eq_out + x_eq
            loss = mse(pred, xnext)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

            if (b + 1) % 100 == 0:
                print(
                    f"  epoch {epoch+1:3d} | batch {b+1:5d}/{len(dataloader):5d} | loss {loss.item():.6e}",
                    end="\r",
                )

        epoch_loss = running_loss / max(n_batches, 1)

        # Save checkpoint every epoch
        if model_save_path:
            # checkpoint_path = model_save_path.replace(".pt", f"_epoch_{epoch+1:03d}.pt")
            checkpoint_path = model_save_path
            torch.save(dynamics_model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Track best model
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_model_state = dynamics_model.state_dict().copy()
            if model_save_path:
                best_path = model_save_path.replace(".pt", "_best.pt")
                torch.save(best_model_state, best_path)
                print(f"  New best model saved: {best_path}")

        t1 = time.time()
        print(
            f"Epoch {epoch+1:3d}/{num_epochs}  loss={epoch_loss:.6e}  ({t1 - t0:.1f}s)"
        )

    # Load best model state before returning
    if best_model_state is not None:
        dynamics_model.load_state_dict(best_model_state)

    print(
        f"Done. best_epoch_loss={best_epoch_loss:.6e}  total_time={time.time() - t_train0:.1f}s"
    )
    return dynamics_model


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Incremental unicycle data gen + training"
    )

    # Common / paths
    parser.add_argument(
        "--data_dir",
        type=str,
        default="neural_network_lyapunov/examples/point_navigation/data/preprocess/dynamics_chunks",
        help="Directory to write/read dynamics_chunk_*.pt files.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="neural_network_lyapunov/examples/point_navigation/data/preprocess/point_nav_forward_model.pt",
        help="Where to save the trained forward model.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float64", choices=["float32", "float64"]
    )

    # Generation options
    parser.add_argument(
        "--generate_chunks",
        action="store_true",
        help="Generate dynamics_chunk_*.pt files.",
    )
    parser.add_argument("--num_chunks", type=int, default=1000)
    parser.add_argument("--samples_per_chunk", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix",
        type=str,
        default="dynamics_chunk",
        help="Filename prefix, e.g., dynamics_chunk_0000.pt",
    )

    # Training options
    parser.add_argument(
        "--train_multifile",
        action="store_true",
        help="Train the forward model on multi-file dataset.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--load_all",
        action="store_true",
        help="Load all data into RAM before training.",
    )
    parser.add_argument(
        "--shuffle_across_files",
        action="store_true",
        help="Shuffle indices across files (slower).",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--prefetch_factor", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.generate_chunks:
        generate_multiple_data_files(
            save_dir=args.data_dir,
            dt=args.dt,
            num_chunks=args.num_chunks,
            samples_per_chunk=args.samples_per_chunk,
            seed=args.seed,
            prefix=args.prefix,
            dtype=dtype,
        )

    if args.train_multifile:
        # Build the same network topology as before using project utils (preferred)
        import neural_network_lyapunov.utils as utils

        dynamics_model = utils.setup_relu(
            (5, 16, 16, 3),
            params=None,
            negative_slope=0.1,
            bias=True,
            dtype=dtype,
        )

        trained = train_forward_model_multifile(
            dynamics_model,
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            load_all_data=args.load_all,
            shuffle_across_files=args.shuffle_across_files,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            dtype=dtype,
        )

        os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
        torch.save(trained, args.model_out)
        print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
