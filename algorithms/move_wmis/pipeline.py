"""Warm-start move-WMIS pipeline around a feasible reference window."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .apply import apply_moves_to_bits
from .conflict import build_conflict_adjacency, selected_moves, solve_exact_mwis
from .diversity import diversity_prune
from .generators import generate_window_moves
from .move_space_sa import MoveSpaceSA, compare_convergence
from .samplers import sample_ahs_surrogate
from .scoring import score_moves
from .types import LocalMove, MoveKind


def run_move_wmis_pipeline(
    reference_bits: Sequence[int],
    *,
    price_usd_kwh: Sequence[float] | None = None,
    carbon_g_kwh: Sequence[float] | None = None,
    dual_slot: Mapping[int, float] | None = None,
    site_weights: Sequence[float] | None = None,
    top_k: int = 64,
    refine_with_sa: bool = True,
    sa_seed: int = 42,
    ahs_shots: int = 128,
    total_evaluation_budget: int = 1000,
    include_block_swaps: bool = True,
    allowed_kinds: Sequence[MoveKind] | None = None,
    exact_mwis: bool = True,
) -> dict:
    """Generate, score, prune, solve, optionally SA-refine, and apply moves.

    Returns a report plus the resulting 240-bit trajectory. No bit repair.
    """
    generated_moves = generate_window_moves(
        reference_bits, include_block_swaps=include_block_swaps
    )
    if allowed_kinds is not None:
        allowed = set(allowed_kinds)
        generated_moves = [
            move for move in generated_moves if move.kind in allowed
        ]
    raw_moves: list[LocalMove] = []
    invalid_move_ids: list[str] = []
    for move in generated_moves:
        try:
            apply_moves_to_bits(reference_bits, [move])
        except ValueError:
            invalid_move_ids.append(move.move_id)
        else:
            raw_moves.append(move)
    scored = score_moves(
        raw_moves,
        reference_bits=reference_bits,
        site_weights=site_weights,
        price_usd_kwh=price_usd_kwh,
        carbon_g_kwh=carbon_g_kwh,
        dual_slot=dual_slot,
    )
    pruned, diversity_report = diversity_prune(scored, k=top_k)
    adjacency = build_conflict_adjacency(pruned, reference_bits=reference_bits)

    exact = (
        solve_exact_mwis(pruned, adjacency=adjacency)
        if exact_mwis
        else {"status": "skipped", "selected_ids": [], "objective": 0.0, "n_selected": 0}
    )
    ahs = sample_ahs_surrogate(
        pruned,
        adjacency,
        outcomes=ahs_shots,
        detuning_beta=1.5,
        seed=sa_seed,
    )
    warm_ids = sorted(ahs.best_state)

    sa_report = None
    compare = None
    if refine_with_sa and pruned:
        positive_weights = [move.weight for move in pruned if move.weight > 0.0]
        temperature = max(
            float(np.std(positive_weights)) if positive_weights else 0.0,
            1e-3,
        )
        warm_iterations = max(1, int(total_evaluation_budget) - int(ahs_shots))
        warm = MoveSpaceSA(
            pruned,
            adjacency,
            initial_state=warm_ids,
            initial_temp=temperature,
            cooling_rate=0.95,
            min_temp=1e-6,
            steps_per_temp=10,
            random_seed=sa_seed,
            max_iterations=warm_iterations,
        ).run()
        cold = MoveSpaceSA(
            pruned,
            adjacency,
            initial_state=None,
            initial_temp=temperature,
            cooling_rate=0.95,
            min_temp=1e-6,
            steps_per_temp=10,
            random_seed=sa_seed + 1,
            max_iterations=int(total_evaluation_budget),
        ).run()
        compare = compare_convergence(warm, cold)
        chosen_ids = warm.best_state
        sa_report = {
            "warm": {
                "best_state": warm.best_state,
                "best_weight_sum": warm.best_weight_sum,
                "n_iterations": warm.n_iterations,
                "iterations_to_95pct": warm.iterations_to_95pct,
            },
            "cold": {
                "best_state": cold.best_state,
                "best_weight_sum": cold.best_weight_sum,
                "n_iterations": cold.n_iterations,
                "iterations_to_95pct": cold.iterations_to_95pct,
            },
            "compare": compare,
            "budget": {
                "ahs_objective_evaluations": ahs.objective_evaluations,
                "warm_sa_iterations": warm_iterations,
                "cold_sa_iterations": int(total_evaluation_budget),
                "hybrid_total_objective_evaluations": (
                    ahs.objective_evaluations + warm.n_iterations
                ),
            },
        }
    else:
        chosen_ids = warm_ids

    chosen: list[LocalMove] = selected_moves(pruned, chosen_ids)
    result_bits = apply_moves_to_bits(reference_bits, chosen)

    return {
        "n_raw_moves": len(generated_moves),
        "n_individually_valid_moves": len(raw_moves),
        "n_individually_invalid_moves_removed": len(invalid_move_ids),
        "individually_invalid_move_ids": invalid_move_ids,
        "n_pruned": len(pruned),
        "diversity": diversity_report,
        "exact_mwis": exact,
        "ahs_surrogate": {
            "shots": ahs.objective_evaluations,
            "best_state": sorted(ahs.best_state),
            "best_weight_sum": ahs.best_value,
            "hardware_counts_present": False,
        },
        "selected_ids": chosen_ids,
        "n_selected": len(chosen_ids),
        "sa": sa_report,
        "result_bits": result_bits,
        "reference_bits": tuple(int(v) for v in reference_bits),
        "changed_sites": [
            i
            for i, (a, b) in enumerate(
                zip(tuple(int(v) for v in reference_bits), result_bits)
            )
            if a != b
        ],
    }
