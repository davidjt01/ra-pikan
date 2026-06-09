"""
KAN layer implementations for PIKAN experiments.

Two layer types:
  - ChebyKANLayer: Chebyshev-polynomial activations (fixed degree, used for
      fixed-grid PIKAN baseline).
  - BSplineKANLayer: B-spline activations with a learnable grid that can be
      extended globally (uniform-extension PIKAN) or in targeted sub-domains
      (RA-PIKAN).

Each layer maps R^{in_features} -> R^{out_features} following the KAN
paradigm: each edge (i -> j) has its own univariate activation function.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChebyKANLayer(nn.Module):
    """
    KAN layer using Chebyshev polynomial activations (degree *k*).

    Activation on edge (i -> j):
        phi_{j,i}(x) = sum_{d=0}^{k} c_{j,i,d} * T_d(tanh(x))

    where T_d is the d-th Chebyshev polynomial of the first kind.
    Input is mapped through tanh to [-1, 1] before polynomial evaluation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree

        self.coef = nn.Parameter(
            torch.empty(out_features, in_features, degree + 1)
        )
        nn.init.normal_(self.coef, std=0.1)

    def _chebyshev(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate T_0, ..., T_k at each element of x (mapped via tanh).

        Args:
            x: (batch, in_features)
        Returns:
            T: (batch, in_features, degree+1)
        """
        z = torch.tanh(x)
        T = [torch.ones_like(z), z]
        for d in range(2, self.degree + 1):
            T.append(2.0 * z * T[-1] - T[-2])
        return torch.stack(T[:self.degree + 1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            y: (batch, out_features)
        """
        T = self._chebyshev(x)
        return torch.einsum("bid,jid->bj", T, self.coef)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" degree={self.degree}"
        )


def _extend_grid(grid: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Extend a base grid with repeated boundary knots for clamped B-splines.

    Args:
        grid: (G+1,) - base knot positions (G intervals)
        spline_order: k
    Returns:
        extended: (G + 2*k - 1,)  [k-1 extra on each side, clamped]
    """
    h = grid[1] - grid[0]
    left_ext = grid[0] - h * torch.arange(spline_order - 1, 0, -1, device=grid.device, dtype=grid.dtype)
    right_ext = grid[-1] + h * torch.arange(1, spline_order, device=grid.device, dtype=grid.dtype)
    return torch.cat([left_ext, grid, right_ext])


def b_spline_basis(
    x: torch.Tensor,
    grid: torch.Tensor,
    spline_order: int,
) -> torch.Tensor:
    """Evaluate B-spline basis using Cox-de Boor recursion.

    Differentiable w.r.t. *x* via torch autograd.

    Args:
        x: (batch,) - 1-D input points
        grid: (n_knots,) - full knot sequence (already extended at boundaries)
        spline_order: k - order of the B-spline (degree = k-1)

    Returns:
        B: (batch, n_knots - spline_order) - basis values at each x
    """
    n = grid.shape[0]
    x_exp = x.unsqueeze(1)

    g_l = grid[:-1].unsqueeze(0)
    g_r = grid[1:].unsqueeze(0)
    B = ((x_exp >= g_l) & (x_exp < g_r)).to(x.dtype)
    last = (x == grid[-1]).to(x.dtype).unsqueeze(1)
    B = torch.cat([B[:, :-1], B[:, -1:] + last], dim=1)

    for k in range(2, spline_order + 1):
        m = n - k
        t_lo = grid[:m].unsqueeze(0)
        t_hi = grid[k - 1: m + k - 1].unsqueeze(0)
        d1 = (t_hi - t_lo).clamp(min=1e-8)
        alpha = (x_exp - t_lo) / d1

        t_lo2 = grid[1: m + 1].unsqueeze(0)
        t_hi2 = grid[k: m + k].unsqueeze(0)
        d2 = (t_hi2 - t_lo2).clamp(min=1e-8)
        beta = (t_hi2 - x_exp) / d2

        B = alpha * B[:, :m] + beta * B[:, 1: m + 1]

    return B


def b_spline_basis_batch_dims(
    x: torch.Tensor,
    grids: List[torch.Tensor],
    spline_order: int,
) -> List[torch.Tensor]:
    """Evaluate B-spline basis for multiple input dimensions.

    Args:
        x: (batch, in_features)
        grids: list of length in_features, each (n_knots_i,)
        spline_order: k

    Returns:
        list of (batch, n_basis_i) tensors
    """
    return [b_spline_basis(x[:, i], grids[i], spline_order)
            for i in range(x.shape[1])]


class BSplineKANLayer(nn.Module):
    """
    KAN layer using cubic B-spline activations with a per-input-dimension
    knot grid that can be refined.

    Activation on edge (i -> j):
        phi_{j,i}(x_i) = sum_k coef[j, i, k] * B_k(x_i)

    where {B_k} are the B-spline basis functions for the current grid on
    dimension i.

    The layer stores a separate grid per input dimension, allowing spatially
    targeted refinement (different grid densities per dimension).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.spline_order = spline_order

        k = spline_order
        base = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        ext = _extend_grid(base, k)

        # Register them as buffers so they follow .to(device) calls.
        self._n_dims = in_features
        for i in range(in_features):
            self.register_buffer(f"grid_{i}", ext.clone())

        n_basis = grid_size + k - 1
        self.coef = nn.Parameter(
            torch.empty(out_features, in_features, n_basis)
        )
        nn.init.normal_(self.coef, std=1.0 / math.sqrt(in_features * n_basis))

    def _grid(self, i: int) -> torch.Tensor:
        return getattr(self, f"grid_{i}")

    def _n_basis(self, i: int) -> int:
        return self._grid(i).shape[0] - self.spline_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)
        Returns:
            y: (batch, out_features)
        """
        batch = x.shape[0]
        y = torch.zeros(batch, self.out_features, device=x.device, dtype=x.dtype)
        for i in range(self.in_features):
            xi = x[:, i]
            grid_i = self._grid(i)
            B = b_spline_basis(xi, grid_i, self.spline_order)
            nb = B.shape[1]
            y = y + B @ self.coef[:, i, :nb].T
        return y


    def n_basis_per_dim(self) -> List[int]:
        return [self._n_basis(i) for i in range(self.in_features)]

    def extend_grid_uniform(
        self,
        new_grid_size: int,
        fit_pts: Optional[torch.Tensor] = None,
    ) -> None:
        """Globally extend all grids to *new_grid_size* intervals.

        If *fit_pts* (shape: batch × in_features) is provided, the new
        coefficients are initialised by least-squares fitting the current
        function values.
        """
        for i in range(self.in_features):
            old_grid = self._grid(i)
            if fit_pts is not None and fit_pts.shape[0] > 0:
                # Adapt grid range to cover actual activation distribution.
                # Without this, deeper-layer grids stay at their init range (e.g. [-1,1])
                # even when activations drift after targeted input-layer extension,
                # causing B-spline extrapolation and catastrophic test-error regression.
                xi = fit_pts[:, i].detach()
                xi_min, xi_max = xi.min().item(), xi.max().item()
                margin = max(0.01, 0.1 * (xi_max - xi_min))
                a, b = xi_min - margin, xi_max + margin
            else:
                a = old_grid[self.spline_order - 1].item()
                b = old_grid[-(self.spline_order)].item()
            new_base = torch.linspace(a, b, new_grid_size + 1, device=old_grid.device, dtype=old_grid.dtype)
            new_ext = _extend_grid(new_base, self.spline_order)
            self._extend_dim(i, new_ext, fit_pts)

    def extend_grid_targeted(
        self,
        dim: int,
        sub_low: float,
        sub_high: float,
        n_new_points: int = 5,
        fit_pts: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert *n_new_points* extra knots in [sub_low, sub_high] for dimension *dim*.

        This is the spatially targeted grid extension unique to RA-PIKAN.
        """
        old_grid = self._grid(dim)
        k = self.spline_order
        inner = old_grid[k - 1: -(k - 1)]
        new_knots = torch.linspace(
            sub_low, sub_high, n_new_points + 2,
            device=old_grid.device, dtype=old_grid.dtype,
        )[1:-1]
        merged_inner = torch.sort(torch.cat([inner, new_knots]))[0]
        new_ext = _extend_grid(
            torch.linspace(merged_inner[0], merged_inner[-1], merged_inner.shape[0],
                           device=old_grid.device, dtype=old_grid.dtype),
            k,
        )
        new_ext_actual = torch.cat([
            old_grid[:k - 1],
            merged_inner,
            old_grid[-(k - 1):],
        ])
        self._extend_dim(dim, new_ext_actual, fit_pts)

    def _extend_dim(
        self,
        dim: int,
        new_grid: torch.Tensor,
        fit_pts: Optional[torch.Tensor] = None,
    ) -> None:
        """Replace the grid for dimension *dim* with *new_grid* and re-project coefficients."""
        k = self.spline_order
        old_grid = self._grid(dim)
        old_n_basis = old_grid.shape[0] - k
        new_n_basis = new_grid.shape[0] - k

        if fit_pts is not None and fit_pts.shape[0] > 0:
            xi = fit_pts[:, dim]
            with torch.no_grad():
                B_old = b_spline_basis(xi, old_grid, k)
                old_coef_dim = self.coef[:, dim, :old_n_basis]
                old_vals = old_coef_dim @ B_old.T
                B_new = b_spline_basis(xi, new_grid, k)
                B_new_np = B_new.cpu().numpy()
                old_vals_np = old_vals.T.cpu().numpy()
                w_np = np.linalg.lstsq(B_new_np, old_vals_np, rcond=None)[0]
                new_coef_dim = torch.tensor(
                    w_np.T, dtype=self.coef.dtype, device=self.coef.device
                )
        else:
            # No fit points: pad or truncate with zeros.
            new_coef_dim = torch.zeros(
                self.out_features, new_n_basis,
                dtype=self.coef.dtype, device=self.coef.device,
            )
            copy_n = min(old_n_basis, new_n_basis)
            new_coef_dim[:, :copy_n] = self.coef[:, dim, :copy_n].detach()

        setattr(self, f"grid_{dim}", new_grid)

        all_dims = []
        for j in range(self.in_features):
            if j == dim:
                all_dims.append(new_coef_dim.unsqueeze(1))
            else:
                all_dims.append(self.coef[:, j, :].unsqueeze(1).detach())
        max_n = max(t.shape[2] for t in all_dims)
        padded = []
        for t in all_dims:
            pad_size = max_n - t.shape[2]
            if pad_size > 0:
                t = F.pad(t, (0, pad_size))
            padded.append(t)
        new_coef = nn.Parameter(torch.cat(padded, dim=1))
        self.coef = new_coef

    def extra_repr(self) -> str:
        bases = self.n_basis_per_dim()
        return (
            f"in_features={self.in_features}, out_features={self.out_features},"
            f" spline_order={self.spline_order}, n_basis_per_dim={bases}"
        )
