"""
Neural network models for PIKAN experiments.

Models:
  - ChebyPIKAN  : stacked ChebyKANLayers (fixed-grid PIKAN baseline)
  - BSplinePIKAN: stacked BSplineKANLayers (RA-PIKAN and uniform-extension)
  - MLPPINN     : tanh MLP (MLP-PINN and RAD-PINN baselines)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .kan_layers import BSplineKANLayer, ChebyKANLayer


# ---------------------------------------------------------------------------
# ChebyPIKAN
# ---------------------------------------------------------------------------

class ChebyPIKAN(nn.Module):
    """
    PIKAN built from ChebyKANLayers.  Used as the *fixed-grid PIKAN* baseline.

    Args:
        layer_widths: [input_dim, h1, h2, ..., output_dim]
        degree: Chebyshev degree for all layers
    """

    def __init__(self, layer_widths: List[int], degree: int = 5) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ChebyKANLayer(layer_widths[i], layer_widths[i + 1], degree=degree)
            for i in range(len(layer_widths) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# BSplinePIKAN
# ---------------------------------------------------------------------------

class BSplinePIKAN(nn.Module):
    """
    PIKAN built from BSplineKANLayers.

    Used for:
      - RA-PIKAN (targeted grid extension per refinement cycle)
      - Uniform-extension PIKAN (global grid extension per cycle)

    Args:
        layer_widths: [input_dim, h1, h2, ..., output_dim]
        grid_size: initial number of B-spline intervals
        spline_order: order of B-spline (3 = cubic)
        grid_range: (a, b) initial knot range
    """

    def __init__(
        self,
        layer_widths: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.layer_widths = layer_widths
        self.layers = nn.ModuleList([
            BSplineKANLayer(
                layer_widths[i],
                layer_widths[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            )
            for i in range(len(layer_widths) - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extend_uniform(
        self,
        new_grid_size: int,
        fit_pts: Optional[torch.Tensor] = None,
    ) -> None:
        """Extend all B-spline grids uniformly (uniform-extension PIKAN).

        *fit_pts* are model inputs (first layer).  They are propagated through
        each layer so that deeper layers receive the correct activations for
        coefficient re-projection.
        """
        with torch.no_grad():
            layer_input = fit_pts
            for layer in self.layers:
                layer.extend_grid_uniform(new_grid_size, fit_pts=layer_input)
                if layer_input is not None:
                    layer_input = layer(layer_input).detach()

    def extend_targeted(
        self,
        dim: int,
        sub_low: float,
        sub_high: float,
        n_new_points: int = 5,
        fit_pts: Optional[torch.Tensor] = None,
    ) -> None:
        """Insert extra knots in [sub_low, sub_high] on dimension *dim*
        of the input layer only.  Deeper layers are left unchanged here;
        call extend_deeper_uniform once per cycle after all regions are
        processed.
        """
        with torch.no_grad():
            layer_input = fit_pts
            for layer_idx, layer in enumerate(self.layers):
                if layer_idx == 0:
                    layer.extend_grid_targeted(
                        dim=dim,
                        sub_low=sub_low,
                        sub_high=sub_high,
                        n_new_points=n_new_points,
                        fit_pts=layer_input,
                    )
                if layer_input is not None:
                    layer_input = layer(layer_input).detach()

    def extend_deeper_uniform(
        self,
        n_new_points: int = 5,
        fit_pts: Optional[torch.Tensor] = None,
    ) -> None:
        """Extend all layers after the first by *n_new_points* uniformly.

        Called once per refinement cycle in ra_pikan_train, after all
        targeted region extensions on the input layer are complete.
        fit_pts are original model inputs; they are propagated through
        the already-updated layer 0 to produce correct activations for
        deeper-layer re-projection.
        """
        with torch.no_grad():
            layer_input = fit_pts
            for layer_idx, layer in enumerate(self.layers):
                if layer_idx > 0:
                    nb_per = layer.n_basis_per_dim()
                    new_gs = max(nb_per) - layer.spline_order + 1 + n_new_points
                    layer.extend_grid_uniform(new_gs, fit_pts=layer_input)
                if layer_input is not None:
                    layer_input = layer(layer_input).detach()


# ---------------------------------------------------------------------------
# MLP-PINN
# ---------------------------------------------------------------------------

class MLPPINN(nn.Module):
    """
    Tanh-activated multi-layer perceptron for MLP-PINN and RAD-PINN baselines.

    Args:
        layer_widths: [input_dim, h1, h2, ..., output_dim]
    """

    def __init__(self, layer_widths: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(layer_widths) - 1):
            layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))
            if i < len(layer_widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
