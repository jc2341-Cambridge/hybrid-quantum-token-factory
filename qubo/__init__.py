"""Quantum / QUBO layer for the hybrid token-factory study.

Public builders live in ``qubo.builders``. Coupling matrices are under
``qubo/matrices/`` (safe to commit; raw market data are not).
"""
from __future__ import annotations

from .builders import (  # noqa: F401
    Qubo,
    add_nogood_cut,
    add_optimality_cut_qubo,
    build_full_qubo,
    build_reduced_qubo,
    build_storage_master_qubo,
    decode_full,
    decode_storage_master,
    greedy_storage_masks,
)

__all__ = [
    "Qubo",
    "build_storage_master_qubo",
    "build_reduced_qubo",
    "build_full_qubo",
    "decode_full",
    "decode_storage_master",
    "greedy_storage_masks",
    "add_nogood_cut",
    "add_optimality_cut_qubo",
]
