"""Dense QUBO construction for induced MWIS objectives."""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np


def build_qubo_matrix(
    weights: Sequence[float],
    edges: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Build the exact dense Q for minimise x^T Q x on an MWIS instance.

    Convention: Q_ii = -w_i; Q_ij = Q_ji = P/2 on each conflict edge,
    with P = 1 + 2 * max_i |w_i|.
    """
    local = np.asarray(weights, dtype=float)
    if local.ndim != 1 or not np.all(np.isfinite(local)):
        raise ValueError("weights must be one finite vector")
    if any(left < 0 or right >= len(local) or left >= right for left, right in edges):
        raise ValueError("edges must be sorted pairs inside the weight vector")
    penalty = 1.0 + 2.0 * float(np.max(np.abs(local), initial=0.0))
    q = np.diag(-local)
    for left, right in edges:
        q[left, right] = penalty / 2.0
        q[right, left] = penalty / 2.0
    return q


def qubo_sha256(q: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(q, dtype="<f8").tobytes()).hexdigest()
