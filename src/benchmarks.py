"""
Benchmark problem definitions: data generators, exact solutions, collocation
samplers, and boundary/initial condition constructors.

Each benchmark class exposes:
  - sample_collocation(N) -> (pts, t_comp, x_comp, ...) with requires_grad
  - sample_ic(N) -> (pts, u_vals)
  - sample_bc(N) -> (pts, u_vals)
  - exact_solution(pts) -> u_ref
  - test_grid() -> (pts, u_ref)
  - pde_residual(model, pts) -> residual tensor
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .pde_residuals import (
    allen_cahn_residual,
    advection_diffusion_residual,
    burgers_residual,
    duffing_residual,
)


class BurgersBenchmark:
    """
    u_t + u * u_x = nu * u_xx,  (t,x) in [0,1] x [-1,1]
    IC: u(0,x) = -sin(pi*x)
    BC: u(t,-1) = u(t,1) = 0   (periodic / homogeneous Dirichlet)

    Exact solution: approximated by pseudo-spectral solver (stored in data/).
    For error evaluation we use the reference computed below.
    """

    def __init__(self, nu: float = 0.01, device: str = "cpu", seed: int = 42) -> None:
        self.nu = nu
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._ref_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _compute_reference(self, n_t: int = 100, n_x: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pseudo-spectral reference solution.

        Uses LSODA (auto-switching Adams/BDF) which handles the stiffness of
        small-nu Burgers.  A 2/3 de-aliasing mask eliminates Gibbs oscillations
        from the nonlinear term.  Falls back gracefully if the integrator
        returns fewer than n_t time steps.
        """
        from scipy.integrate import odeint

        nu = self.nu
        x = np.linspace(-1, 1, n_x, endpoint=False)
        t_eval = np.linspace(0.0, 1.0, n_t)
        u0 = -np.sin(np.pi * x)

        k = np.fft.fftfreq(n_x, d=(x[1] - x[0])) * 2 * np.pi

        # 2/3 de-aliasing mask to avoid aliasing errors in the nonlinear term.
        dealias = np.zeros(n_x, dtype=bool)
        dealias[: n_x // 3] = True
        dealias[-n_x // 3 :] = True

        def rhs_flat(u_hat_flat, t):
            u_hat = u_hat_flat.view(complex).reshape(-1)
            u = np.fft.ifft(u_hat).real
            # Compute u*u_x in physical space (de-aliased product).
            u_hat_da = u_hat.copy()
            u_hat_da[~dealias] = 0.0
            u_x = np.fft.ifft(1j * k * u_hat_da).real
            prod = u * u_x
            nonlinear = np.fft.fft(-prod)
            nonlinear[~dealias] = 0.0
            rhs_hat = nonlinear - nu * k ** 2 * u_hat
            return np.concatenate([rhs_hat.view(float)])

        u0_hat = np.fft.fft(u0)
        y0 = np.concatenate([u0_hat.view(float)])

        # odeint uses LSODA, which auto-detects stiffness.
        sol_y, info = odeint(
            rhs_flat, y0, t_eval,
            rtol=1e-6, atol=1e-8,
            full_output=True,
        )

        n_saved = sol_y.shape[0]
        u_ref = np.zeros((n_saved, n_x))
        for i in range(n_saved):
            u_hat = sol_y[i].view(complex)
            u_ref[i] = np.fft.ifft(u_hat).real

        t_saved = t_eval[:n_saved]
        t_grid, x_grid = np.meshgrid(t_saved, x, indexing="ij")
        return t_grid, x_grid, u_ref

    def test_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pts, u_ref) on a 100x256 test grid."""
        if self._ref_cache is None:
            t_g, x_g, u_g = self._compute_reference(100, 256)
            pts = torch.tensor(
                np.stack([t_g.ravel(), x_g.ravel()], axis=1), dtype=torch.float32
            ).to(self.device)
            u_ref = torch.tensor(u_g.ravel(), dtype=torch.float32).to(self.device)
            self._ref_cache = (pts, u_ref)
        return self._ref_cache

    def sample_collocation(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Latin-hypercube-like random sample in (0,1] x [-1,1]."""
        t = self.rng.uniform(0.0, 1.0, (N, 1)).astype(np.float32)
        x = self.rng.uniform(-1.0, 1.0, (N, 1)).astype(np.float32)
        t_t = torch.tensor(t, requires_grad=True, device=self.device)
        x_t = torch.tensor(x, requires_grad=True, device=self.device)
        return t_t, x_t

    def sample_ic(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """IC: u(0, x) = -sin(pi*x)."""
        x = self.rng.uniform(-1.0, 1.0, (N, 1)).astype(np.float32)
        u = -np.sin(np.pi * x)
        pts = torch.tensor(
            np.hstack([np.zeros((N, 1), dtype=np.float32), x]), device=self.device
        )
        u_t = torch.tensor(u, device=self.device)
        return pts, u_t

    def sample_bc(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dirichlet BC: u(t, ±1) = 0."""
        t = self.rng.uniform(0.0, 1.0, (N, 1)).astype(np.float32)
        x_left = np.full((N, 1), -1.0, dtype=np.float32)
        x_right = np.full((N, 1), 1.0, dtype=np.float32)
        pts_l = torch.tensor(np.hstack([t, x_left]), device=self.device)
        pts_r = torch.tensor(np.hstack([t, x_right]), device=self.device)
        pts = torch.cat([pts_l, pts_r], dim=0)
        u_bc = torch.zeros(2 * N, 1, device=self.device)
        return pts, u_bc

    def pde_residual(
        self, model: torch.nn.Module,
        t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return burgers_residual(model, t, x, self.nu)


class AllenCahnBenchmark:
    """
    u_t = eps^2 * u_xx + u - u^3,  (t,x) in [0,1] x [-1,1]
    IC: u(0,x) = x^2 * cos(pi*x)
    BC: u(t,-1) = u(t,1) = -1

    Reference solution from pseudo-spectral integration.
    """

    def __init__(self, eps: float = 0.1, device: str = "cpu", seed: int = 42) -> None:
        self.eps = eps
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._ref_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _compute_reference(self, n_t: int = 100, n_x: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.integrate import solve_ivp

        eps = self.eps
        x = np.linspace(-1, 1, n_x)
        t_eval = np.linspace(0.0, 1.0, n_t)
        u0 = x ** 2 * np.cos(np.pi * x)

        dx = x[1] - x[0]

        def rhs(t, u):
            u_xx = np.zeros_like(u)
            u_xx[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2
            u_xx[0] = 0.0
            u_xx[-1] = 0.0
            return eps ** 2 * u_xx + u - u ** 3

        sol = solve_ivp(
            rhs, (0.0, 1.0), u0, t_eval=t_eval,
            method="RK45", rtol=1e-6, atol=1e-8,
        )
        u_ref = sol.y.T
        t_grid, x_grid = np.meshgrid(t_eval, x, indexing="ij")
        return t_grid, x_grid, u_ref

    def test_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ref_cache is None:
            t_g, x_g, u_g = self._compute_reference(100, 256)
            pts = torch.tensor(
                np.stack([t_g.ravel(), x_g.ravel()], axis=1), dtype=torch.float32
            ).to(self.device)
            u_ref = torch.tensor(u_g.ravel(), dtype=torch.float32).to(self.device)
            self._ref_cache = (pts, u_ref)
        return self._ref_cache

    def sample_collocation(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.rng.uniform(0.0, 1.0, (N, 1)).astype(np.float32)
        x = self.rng.uniform(-1.0, 1.0, (N, 1)).astype(np.float32)
        t_t = torch.tensor(t, requires_grad=True, device=self.device)
        x_t = torch.tensor(x, requires_grad=True, device=self.device)
        return t_t, x_t

    def sample_ic(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.rng.uniform(-1.0, 1.0, (N, 1)).astype(np.float32)
        u = (x ** 2 * np.cos(np.pi * x)).astype(np.float32)
        pts = torch.tensor(np.hstack([np.zeros((N, 1), dtype=np.float32), x]), device=self.device)
        return pts, torch.tensor(u, device=self.device)

    def sample_bc(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.rng.uniform(0.0, 1.0, (N, 1)).astype(np.float32)
        x_left = np.full((N, 1), -1.0, dtype=np.float32)
        x_right = np.full((N, 1), 1.0, dtype=np.float32)
        pts = torch.tensor(
            np.vstack([np.hstack([t, x_left]), np.hstack([t, x_right])]),
            device=self.device,
        )
        u_bc = torch.full((2 * N, 1), -1.0, device=self.device)
        return pts, u_bc

    def pde_residual(self, model, t, x):
        return allen_cahn_residual(model, t, x, self.eps)


class AdvectionDiffusionBenchmark:
    """
    -eps * Delta u + b . grad(u) = 0, (x,y) in [0,1]^2
    b = (1, 0), eps = 1/Pe

    Exact solution (1D exponential layer in x):
        u_exact(x, y) = (exp(Pe * x) - 1) / (exp(Pe) - 1)

    BC:
        u(0, y) = 0,  u(1, y) = 1   (Dirichlet on left/right)
        du/dn = 0 on top/bottom (Neumann, automatically satisfied by 1D exact)
    """

    def __init__(self, Pe: float = 100.0, device: str = "cpu", seed: int = 42) -> None:
        self.Pe = Pe
        self.eps = 1.0 / Pe
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._ref_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def exact(self, x_np: np.ndarray) -> np.ndarray:
        """Compute exact solution at x-coordinates (ignoring y due to separability).

        Reformulated to avoid exp(Pe) overflow at large Pe:
            u = (exp(Pe*(x-1)) - exp(-Pe)) / (1 - exp(-Pe))
        """
        Pe = self.Pe
        # exp(-Pe) underflows to 0 for Pe >= ~745, which is fine.
        exp_neg_Pe = np.exp(-float(Pe))
        return (np.exp(Pe * (x_np.astype(np.float64) - 1.0)) - exp_neg_Pe) / (1.0 - exp_neg_Pe)

    def test_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-uniform test grid: 64 uniform + 64 concentrated near x=1.

        Using a non-uniform x-grid ensures the boundary layer (width ~1/Pe)
        is well-represented in the evaluation, making relative L2 meaningful.
        """
        if self._ref_cache is None:
            n = 64
            eps = self.eps
            # Uniform part.
            x_uniform = np.linspace(0, 1, n, dtype=np.float32)
            # Concentrated part near x=1 (covers ~5 layer widths with n points).
            x_layer = (1.0 - 5.0 * eps * np.linspace(1, 0, n, dtype=np.float32))
            x_layer = np.clip(x_layer, 0.0, 1.0)
            x_1d = np.sort(np.unique(np.concatenate([x_uniform, x_layer])))
            y_1d = np.linspace(0, 1, 64, dtype=np.float32)
            xg, yg = np.meshgrid(x_1d, y_1d, indexing="ij")
            pts = torch.tensor(
                np.stack([xg.ravel().astype(np.float32),
                          yg.ravel().astype(np.float32)], axis=1),
                device=self.device,
            )
            u_ref = torch.tensor(
                self.exact(xg.ravel()).astype(np.float32), device=self.device
            )
            self._ref_cache = (pts, u_ref)
        return self._ref_cache

    def sample_collocation(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-uniform sampling: 50% uniform + 50% concentrated near x=1.

        The exact solution has an exponential boundary layer of width O(1/Pe)
        near x=1.  Biased sampling ensures the layer is represented in the
        collocation set even at high Pe.
        """
        eps = self.eps
        n_uniform = N // 2
        n_layer = N - n_uniform

        # Uniform half.
        x_u = self.rng.uniform(0.0, 1.0, (n_uniform, 1)).astype(np.float32)
        y_u = self.rng.uniform(0.0, 1.0, (n_uniform, 1)).astype(np.float32)

        # Concentrated half: x in [1 - 10*eps, 1] (covers ~10 layer widths).
        x_lo = max(0.0, 1.0 - 10.0 * eps)
        x_l = self.rng.uniform(x_lo, 1.0, (n_layer, 1)).astype(np.float32)
        y_l = self.rng.uniform(0.0, 1.0, (n_layer, 1)).astype(np.float32)

        x = np.vstack([x_u, x_l])
        y = np.vstack([y_u, y_l])
        x_t = torch.tensor(x, requires_grad=True, device=self.device)
        y_t = torch.tensor(y, requires_grad=True, device=self.device)
        return x_t, y_t

    def sample_bc(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dirichlet on x=0 (u=0) and x=1 (u=1)."""
        y = self.rng.uniform(0.0, 1.0, (N, 1)).astype(np.float32)
        pts_l = torch.tensor(np.hstack([np.zeros((N, 1), dtype=np.float32), y]), device=self.device)
        pts_r = torch.tensor(np.hstack([np.ones((N, 1), dtype=np.float32), y]), device=self.device)
        u_l = torch.zeros(N, 1, device=self.device)
        u_r = torch.ones(N, 1, device=self.device)
        pts = torch.cat([pts_l, pts_r], dim=0)
        u_bc = torch.cat([u_l, u_r], dim=0)
        return pts, u_bc

    def pde_residual(self, model, x, y):
        return advection_diffusion_residual(model, x, y, self.Pe)


class DuffingBenchmark:
    """
    x'' + delta * x' + alpha * x + beta * x^3 = gamma * cos(omega * t)
    t in [0, T]

    Default parameters (moderately nonlinear):
        delta=0.5, alpha=1.0, beta=0.1, gamma=0.3, omega=1.2, T=10.0
    IC: x(0) = 1.0, x'(0) = 0.0

    Reference from scipy ODE solver.
    """

    def __init__(
        self,
        delta: float = 0.5,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.3,
        omega: float = 1.2,
        T: float = 10.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.T = T
        self.device = device
        self.rng = np.random.default_rng(seed)
        self._ref_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _compute_reference(self, n_pts: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.integrate import solve_ivp

        def rhs(t, y):
            x, xd = y
            xdd = (
                self.gamma * np.cos(self.omega * t)
                - self.delta * xd
                - self.alpha * x
                - self.beta * x ** 3
            )
            return [xd, xdd]

        t_eval = np.linspace(0.0, self.T, n_pts)
        sol = solve_ivp(rhs, (0.0, self.T), [1.0, 0.0], t_eval=t_eval,
                        method="RK45", rtol=1e-10, atol=1e-12)
        return sol.t, sol.y[0]

    def test_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ref_cache is None:
            t_np, x_np = self._compute_reference(1000)
            pts = torch.tensor(t_np[:, None], dtype=torch.float32, device=self.device)
            u_ref = torch.tensor(x_np, dtype=torch.float32, device=self.device)
            self._ref_cache = (pts, u_ref)
        return self._ref_cache

    def sample_collocation(self, N: int) -> torch.Tensor:
        t = self.rng.uniform(0.0, self.T, (N, 1)).astype(np.float32)
        return torch.tensor(t, requires_grad=True, device=self.device)

    def sample_ic(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return IC points for x(0)=1 and x'(0)=0."""
        t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=True, device=self.device)
        return t0

    def pde_residual(self, model, t):
        return duffing_residual(model, t, self.delta, self.alpha, self.beta, self.gamma, self.omega)
