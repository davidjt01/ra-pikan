"""
Evaluation metrics for PIKAN experiments.
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


def relative_l2_error(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative L2 error: ||pred - ref||_2 / ||ref||_2."""
    with torch.no_grad():
        num = torch.norm(pred.squeeze() - ref.squeeze())
        den = torch.norm(ref.squeeze())
        return (num / (den + 1e-16)).item()


def gini_coefficient(residuals: torch.Tensor) -> float:
    """
    Gini coefficient of the pointwise residual magnitudes.

    G = 0 means uniform distribution (residuals equal everywhere).
    G = 1 means maximally concentrated (all residual at one point).

    Used to confirm that adaptive refinement concentrates and then reduces
    residuals in targeted regions.
    """
    with torch.no_grad():
        vals = residuals.abs().squeeze().cpu().numpy()
        vals = np.sort(vals)
        n = len(vals)
        if n == 0 or vals.sum() < 1e-30:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * (idx * vals).sum() - (n + 1) * vals.sum()) / (n * vals.sum()))


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@contextmanager
def timer():
    """Context manager that measures wall-clock time (seconds)."""
    start = time.perf_counter()
    result = {"elapsed": 0.0}
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


@contextmanager
def memory_tracker():
    """Context manager that measures peak RAM usage (bytes)."""
    tracemalloc.start()
    result = {"peak_bytes": 0}
    try:
        yield result
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_bytes"] = peak
