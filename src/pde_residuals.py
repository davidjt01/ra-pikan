"""
PDE residual functions for each benchmark.

All functions accept a model and collocation points (with grad enabled) and
return the pointwise residual tensor (shape: (N,)).  They use torch.autograd
to compute spatial/temporal derivatives.

Conventions:
  - Burgers/Allen-Cahn: model input (t, x), output u(t, x)
  - Advection-diffusion: model input (x, y), output u(x, y)
  - Duffing: model input (t,), output x(t)
"""

from __future__ import annotations

import torch


def _grad(u: torch.Tensor, v: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute du/dv via autograd."""
    (g,) = torch.autograd.grad(
        u, v, grad_outputs=torch.ones_like(u),
        create_graph=create_graph, retain_graph=True,
    )
    return g


# ---------------------------------------------------------------------------
# 1D Burgers
# ---------------------------------------------------------------------------

def burgers_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    nu: float,
) -> torch.Tensor:
    """
    PDE: u_t + u * u_x - nu * u_xx = 0

    Args:
        t, x: (N, 1) tensors with requires_grad=True
        nu:   kinematic viscosity
    Returns:
        residual: (N,)
    """
    inp = torch.cat([t, x], dim=1)  # (N, 2)
    u = model(inp)                  # (N, 1)
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return (u_t + u * u_x - nu * u_xx).squeeze(1)


# ---------------------------------------------------------------------------
# 1D Allen–Cahn
# ---------------------------------------------------------------------------

def allen_cahn_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """
    PDE: u_t - eps^2 * u_xx - u + u^3 = 0

    Args:
        t, x: (N, 1) with requires_grad=True
        eps:  interface width parameter
    Returns:
        residual: (N,)
    """
    inp = torch.cat([t, x], dim=1)
    u = model(inp)
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return (u_t - eps ** 2 * u_xx - u + u ** 3).squeeze(1)


# ---------------------------------------------------------------------------
# 2D Steady Advection–Diffusion
# ---------------------------------------------------------------------------

def advection_diffusion_residual(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    Pe: float,
    bx: float = 1.0,
    by: float = 0.0,
) -> torch.Tensor:
    """
    PDE: -eps * (u_xx + u_yy) + bx * u_x + by * u_y = f
    where eps = 1/Pe and f is chosen so that the exact solution is known.

    With f = 0 and b = (1, 0):  -eps * Delta u + u_x = 0  (pure advection-diffusion)

    For Pe in {100, 500} with b = (1, 0), the solution has an exponential
    boundary layer at x=1:
        u(x, y) = (exp(Pe * x) - 1) / (exp(Pe) - 1)  (1D profile, uniform in y)

    We use f = 0 and the exact solution to define boundary conditions.

    Args:
        x, y: (N, 1) with requires_grad=True
        Pe:   Péclet number
        bx, by: advection velocity components
    Returns:
        residual: (N,)
    """
    eps = 1.0 / Pe
    inp = torch.cat([x, y], dim=1)
    u = model(inp)
    u_x = _grad(u, x)
    u_y = _grad(u, y)
    u_xx = _grad(u_x, x)
    u_yy = _grad(u_y, y)
    return (-eps * (u_xx + u_yy) + bx * u_x + by * u_y).squeeze(1)


# ---------------------------------------------------------------------------
# 1D Duffing Oscillator
# ---------------------------------------------------------------------------

def duffing_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    delta: float,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
) -> torch.Tensor:
    """
    ODE: x'' + delta * x' + alpha * x + beta * x^3 = gamma * cos(omega * t)

    Args:
        t: (N, 1) with requires_grad=True
        delta, alpha, beta, gamma, omega: Duffing parameters
    Returns:
        residual: (N,)
    """
    x = model(t)           # (N, 1)
    x_t = _grad(x, t)
    x_tt = _grad(x_t, t)
    forcing = gamma * torch.cos(omega * t)
    return (x_tt + delta * x_t + alpha * x + beta * x ** 3 - forcing).squeeze(1)


# ---------------------------------------------------------------------------
# Boundary / initial condition helpers
# ---------------------------------------------------------------------------

def ic_loss(
    model: torch.nn.Module,
    t0_pts: torch.Tensor,
    u_ic: torch.Tensor,
) -> torch.Tensor:
    """Mean-square IC loss.  t0_pts and u_ic have shape (N, input_dim) and (N, 1)."""
    pred = model(t0_pts)
    return ((pred - u_ic) ** 2).mean()


def bc_loss_dirichlet(
    model: torch.nn.Module,
    bc_pts: torch.Tensor,
    u_bc: torch.Tensor,
) -> torch.Tensor:
    """Mean-square Dirichlet BC loss."""
    pred = model(bc_pts)
    return ((pred - u_bc) ** 2).mean()
