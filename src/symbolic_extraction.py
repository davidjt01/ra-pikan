"""
Symbolic extraction for trained KAN models.

After training, each activation function (a 1D spline or Chebyshev curve) can
be approximated by a symbolic expression from a candidate library.  We fit
each activation's input-output curve, compose the expressions through the
network layers, and produce a human-readable formula.

Candidate library: polynomials (degree 1-4), sin, cos, exp, log, sqrt, x^2,
                   x^3.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Symbolic function library
# ---------------------------------------------------------------------------

LIBRARY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "1":     lambda x: np.ones_like(x),
    "x":     lambda x: x,
    "x^2":   lambda x: x ** 2,
    "x^3":   lambda x: x ** 3,
    "x^4":   lambda x: x ** 4,
    "sin":   lambda x: np.sin(x),
    "cos":   lambda x: np.cos(x),
    "exp":   lambda x: np.exp(np.clip(x, -10, 10)),
    "log":   lambda x: np.log(np.abs(x) + 1e-8),
    "sqrt":  lambda x: np.sqrt(np.abs(x)),
    "tanh":  lambda x: np.tanh(x),
}


def fit_activation(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    library: Optional[Dict[str, Callable]] = None,
    max_terms: int = 3,
) -> Tuple[str, float]:
    """
    Fit the 1D curve (x_vals -> y_vals) using a subset of the library via
    forward-selection least squares.

    Returns:
        (formula_string, r2_score)
    """
    if library is None:
        library = LIBRARY

    best_r2 = -np.inf
    best_formula = "unknown"

    for n_terms in range(1, max_terms + 1):
        for names in itertools.combinations(library.keys(), n_terms):
            X = np.column_stack([library[n](x_vals) for n in names])
            if not np.all(np.isfinite(X)):
                continue
            # Least squares
            coeffs, _, _, _ = np.linalg.lstsq(X, y_vals, rcond=None)
            pred = X @ coeffs
            ss_res = np.sum((y_vals - pred) ** 2)
            ss_tot = np.sum((y_vals - y_vals.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-16)
            if r2 > best_r2:
                best_r2 = r2
                # Build formula string
                terms = []
                for c, n in zip(coeffs, names):
                    if abs(c) > 1e-6:
                        terms.append(f"{c:.4f}*{n}")
                best_formula = " + ".join(terms) if terms else "0"

    return best_formula, float(best_r2)


def extract_edge_activation(
    layer: nn.Module,
    out_idx: int,
    in_idx: int,
    n_pts: int = 200,
    x_range: Tuple[float, float] = (-2.0, 2.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the activation function of edge (in_idx -> out_idx) in *layer* by
    passing points through it.

    Returns:
        (x_vals, y_vals) arrays
    """
    x_np = np.linspace(x_range[0], x_range[1], n_pts, dtype=np.float32)
    x_t = torch.tensor(x_np[:, None], requires_grad=False)

    # Build a dummy full input: set all input dims to zero except in_idx
    in_features = layer.in_features if hasattr(layer, "in_features") else layer.layers[0].in_features
    dummy = torch.zeros(n_pts, in_features)
    dummy[:, in_idx] = torch.tensor(x_np)
    device = next(layer.parameters()).device
    dummy = dummy.to(device)

    with torch.no_grad():
        # Output contribution from in_idx only: isolate by zeroing other dims
        full_out = layer(dummy)  # (n_pts, out_features)
        y_vals = full_out[:, out_idx].cpu().numpy()

    return x_np, y_vals


def symbolic_extraction_report(
    model: nn.Module,
    r2_threshold: float = 0.95,
    n_pts: int = 300,
) -> List[Dict]:
    """
    Run symbolic extraction for all edges in the first layer of *model*.

    Returns a list of dicts with keys:
        layer_idx, in_idx, out_idx, formula, r2
    """
    results = []
    layers = model.layers if hasattr(model, "layers") else list(model.modules())
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "in_features"):
            continue
        for i in range(layer.in_features):
            for j in range(layer.out_features):
                x_v, y_v = extract_edge_activation(layer, j, i, n_pts=n_pts)
                formula, r2 = fit_activation(x_v, y_v)
                entry = {
                    "layer_idx": layer_idx,
                    "in_idx": i,
                    "out_idx": j,
                    "formula": formula,
                    "r2": r2,
                }
                results.append(entry)
                if r2 >= r2_threshold:
                    print(f"  Layer {layer_idx} edge ({i}->{j}): {formula}  R²={r2:.4f}")
    return results


def formula_l2_error(
    formula_fn: Callable[[np.ndarray], np.ndarray],
    test_pts: torch.Tensor,
    u_ref: torch.Tensor,
) -> float:
    """Evaluate L2 error of a symbolic formula on the test grid."""
    x_np = test_pts.cpu().numpy()
    pred_np = formula_fn(x_np)
    ref_np = u_ref.cpu().numpy()
    num = np.linalg.norm(pred_np - ref_np)
    den = np.linalg.norm(ref_np) + 1e-16
    return float(num / den)
