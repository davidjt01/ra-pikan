"""
Generic trainer for physics-informed models (MLP-PINN, fixed-grid PIKAN,
RAD-PINN, uniform-extension PIKAN, RA-PIKAN).

Training uses:
  1. Adam for the main training phase.
  2. Optional L-BFGS fine-tuning at the end of each phase.

Loss = lambda_r * MSE(residual) + lambda_bc * MSE(bc) + lambda_ic * MSE(ic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TrainConfig:
    """Hyper-parameters for one training phase."""
    n_adam: int = 3000
    n_lbfgs: int = 500
    lr_adam: float = 1e-3
    lr_lbfgs: float = 1.0
    lambda_r: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0
    log_every: int = 500
    device: str = "cpu"


def _pde_loss(
    model: nn.Module,
    pde_fn: Callable,
    t_or_x: torch.Tensor,
    x_or_y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute mean-square PDE residual loss."""
    if x_or_y is not None:
        res = pde_fn(model, t_or_x, x_or_y)
    else:
        res = pde_fn(model, t_or_x)
    return (res ** 2).mean()


def compute_total_loss(
    model: nn.Module,
    collocator: Callable,
    ic_sampler: Optional[Callable],
    bc_sampler: Optional[Callable],
    pde_fn: Callable,
    cfg: TrainConfig,
    is_2d: bool = False,
) -> torch.Tensor:
    """Compute the composite PINN loss for one optimiser step."""
    if is_2d:
        x_c, y_c = collocator()
        loss = cfg.lambda_r * _pde_loss(model, pde_fn, x_c, y_c)
    else:
        t_c, x_c = collocator()
        loss = cfg.lambda_r * _pde_loss(model, pde_fn, t_c, x_c)

    if bc_sampler is not None:
        pts_bc, u_bc = bc_sampler()
        pred_bc = model(pts_bc)
        loss = loss + cfg.lambda_bc * ((pred_bc - u_bc) ** 2).mean()

    if ic_sampler is not None:
        pts_ic, u_ic = ic_sampler()
        pred_ic = model(pts_ic)
        loss = loss + cfg.lambda_ic * ((pred_ic - u_ic) ** 2).mean()

    return loss


def train_model(
    model: nn.Module,
    collocator: Callable,
    ic_sampler: Optional[Callable],
    bc_sampler: Optional[Callable],
    pde_fn: Callable,
    cfg: TrainConfig,
    is_2d: bool = False,
    loss_history: Optional[List[float]] = None,
) -> Dict:
    """
    Train *model* for cfg.n_adam Adam steps + cfg.n_lbfgs L-BFGS steps.

    Returns a dict with: {loss_history, elapsed_seconds}.
    """
    if loss_history is None:
        loss_history = []

    model.train()
    adam = optim.Adam(model.parameters(), lr=cfg.lr_adam)

    t0 = time.perf_counter()

    for step in range(cfg.n_adam):
        adam.zero_grad()
        loss = compute_total_loss(
            model, collocator, ic_sampler, bc_sampler, pde_fn, cfg, is_2d
        )
        loss.backward()
        adam.step()
        loss_history.append(loss.item())
        if (step + 1) % cfg.log_every == 0:
            print(f"  Adam {step+1}/{cfg.n_adam}  loss={loss.item():.4e}")

    if cfg.n_lbfgs > 0:
        lbfgs = optim.LBFGS(
            model.parameters(),
            lr=cfg.lr_lbfgs,
            max_iter=cfg.n_lbfgs,
            history_size=30,
            line_search_fn="strong_wolfe",
        )

        # Fix data for L-BFGS; a consistent loss function is required.
        fixed_coll = collocator()
        fixed_ic = ic_sampler() if ic_sampler else None
        fixed_bc = bc_sampler() if bc_sampler else None
        _last_loss = [float("inf")]

        def closure():
            lbfgs.zero_grad()
            l = compute_total_loss(
                model,
                lambda: fixed_coll,
                (lambda: fixed_ic) if fixed_ic is not None else None,
                (lambda: fixed_bc) if fixed_bc is not None else None,
                pde_fn, cfg, is_2d,
            )
            l.backward()
            _last_loss[0] = l.item()
            return l

        lbfgs.step(closure)
        loss_history.append(_last_loss[0])
        print(f"  L-BFGS done  loss={_last_loss[0]:.4e}")

    elapsed = time.perf_counter() - t0
    steps_per_second = cfg.n_adam / elapsed if elapsed > 0 else 0.0
    return {
        "loss_history": loss_history,
        "elapsed_seconds": elapsed,
        "steps_per_second": steps_per_second,
    }


def rad_resample(
    current_pts: Tuple[torch.Tensor, ...],
    residual_vals: torch.Tensor,
    n_add: int,
    rng: np.random.Generator,
    device: str = "cpu",
) -> Tuple[torch.Tensor, ...]:
    """
    Residual-Adaptive Data (RAD) resampling: draw *n_add* new collocation
    points proportionally to |residual|.

    Args:
        current_pts: tuple of 1-D tensors (e.g. (t, x) or (x, y))
        residual_vals: (N,) pointwise residual magnitudes
        n_add: number of new points to draw

    Returns:
        New pts tuple drawn proportional to residual magnitude.
    """
    with torch.no_grad():
        probs = residual_vals.abs().cpu().numpy()
        probs = probs + 1e-8  # Avoid zero probability.
        probs /= probs.sum()
        idx = rng.choice(len(probs), size=n_add, replace=True, p=probs)

    new_pts = []
    for dim_pts in current_pts:
        arr = dim_pts.detach().cpu().numpy()
        new_arr = arr[idx].astype(np.float32)
        new_pts.append(
            torch.tensor(new_arr, requires_grad=True, device=device)
        )
    return tuple(new_pts)
