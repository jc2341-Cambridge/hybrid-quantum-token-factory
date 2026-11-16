"""Move-WMIS pricing for Iteration7.

Quantum/surrogate sampling acts on local edits around a feasible 240-bit
reference window. This package enforces physical independent-set validity;
the canonical campaign then audits selected bits with the Iteration5
no-repair forward model and exact reduced-cost gate. Bits are never repaired.
"""
from __future__ import annotations

from .apply import apply_moves_to_bits
from .conflict import build_conflict_adjacency, moves_conflict, solve_exact_mwis
from .diversity import diversity_prune
from .generators import generate_window_moves
from .move_space_sa import MoveSpaceSA, compare_convergence
from .pipeline import run_move_wmis_pipeline
from .samplers import GraphSampleBatch, sample_ahs_surrogate
from .scoring import score_moves
from .types import LocalMove, MoveKind

__all__ = [
    "LocalMove",
    "MoveKind",
    "apply_moves_to_bits",
    "build_conflict_adjacency",
    "compare_convergence",
    "diversity_prune",
    "generate_window_moves",
    "GraphSampleBatch",
    "moves_conflict",
    "MoveSpaceSA",
    "run_move_wmis_pipeline",
    "sample_ahs_surrogate",
    "score_moves",
    "solve_exact_mwis",
]
