"""Conflict graph utilities for move-WMIS (no NetworkX dependency)."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from algorithms import encoding

from .types import LocalMove


def moves_conflict(
    a: LocalMove,
    b: LocalMove,
    *,
    reference_bits: Sequence[int] | None = None,
) -> bool:
    """True when two moves cannot be applied together.

    Semantic key overlap is always a conflict. When ``reference_bits`` is
    supplied, the pair also conflicts if joint application would destroy the
    physical independent set (extra UDG edges), even if keys are disjoint.
    """
    if a.touched_keys & b.touched_keys:
        return True
    if reference_bits is None:
        return False
    values = [int(v) for v in reference_bits]
    for move in (a, b):
        for index, target in move.flips:
            values[index] = int(target)
    try:
        encoding.decode_independent_set(values, verify_independent=True)
    except ValueError:
        return True
    return False


def build_conflict_adjacency(
    moves: Sequence[LocalMove],
    *,
    reference_bits: Sequence[int] | None = None,
) -> dict[str, set[str]]:
    """Symmetric adjacency: move_id -> conflicting move_ids."""
    move_list = list(moves)
    adj: dict[str, set[str]] = {m.move_id: set() for m in move_list}
    for i, left in enumerate(move_list):
        for right in move_list[i + 1 :]:
            if moves_conflict(left, right, reference_bits=reference_bits):
                adj[left.move_id].add(right.move_id)
                adj[right.move_id].add(left.move_id)
    return adj


def solve_exact_mwis(
    moves: Sequence[LocalMove],
    *,
    adjacency: dict[str, set[str]] | None = None,
    time_limit_s: float = 30.0,
) -> dict:
    """Exact MWIS via SciPy HiGHS MILP. Objective: maximize sum of weights."""
    move_list = list(moves)
    if not move_list:
        return {
            "status": "empty",
            "objective": 0.0,
            "selected_ids": [],
            "n_selected": 0,
            "success": True,
            "mip_gap": 0.0,
            "certified_exact": True,
        }

    ids = [m.move_id for m in move_list]
    idx = {mid: i for i, mid in enumerate(ids)}
    weights = np.asarray([float(m.weight) for m in move_list], dtype=float)
    adj = adjacency if adjacency is not None else build_conflict_adjacency(move_list)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    ub: list[float] = []
    row = 0
    seen: set[tuple[str, str]] = set()
    for u, neighbours in adj.items():
        for v in neighbours:
            edge = (u, v) if u < v else (v, u)
            if edge in seen:
                continue
            seen.add(edge)
            rows.extend([row, row])
            cols.extend([idx[u], idx[v]])
            data.extend([1.0, 1.0])
            ub.append(1.0)
            row += 1

    constraints = []
    if ub:
        matrix = coo_matrix((data, (rows, cols)), shape=(len(ub), len(ids)))
        constraints.append(
            LinearConstraint(matrix, lb=np.zeros(len(ub)), ub=np.asarray(ub, dtype=float))
        )

    result = milp(
        c=-weights,
        integrality=np.ones(len(ids), dtype=int),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options={"time_limit": float(time_limit_s)},
    )
    if result.x is None:
        return {
            "status": str(result.message),
            "objective": 0.0,
            "selected_ids": [],
            "n_selected": 0,
            "success": False,
            "mip_gap": (
                None
                if getattr(result, "mip_gap", None) is None
                else float(result.mip_gap)
            ),
            "certified_exact": False,
        }
    selected = [ids[i] for i, value in enumerate(result.x) if value >= 0.5]
    objective = float(sum(weights[idx[mid]] for mid in selected))
    gap = (
        None
        if getattr(result, "mip_gap", None) is None
        else float(result.mip_gap)
    )
    certified = bool(
        result.success
        and int(result.status) == 0
        and gap is not None
        and abs(gap) <= 1e-10
    )
    return {
        "status": "optimal" if result.success else str(result.message),
        "objective": objective,
        "selected_ids": selected,
        "n_selected": len(selected),
        "success": bool(result.success),
        "mip_gap": gap,
        "certified_exact": certified,
    }


def selected_moves(
    moves: Iterable[LocalMove], selected_ids: Sequence[str]
) -> list[LocalMove]:
    by_id = {m.move_id: m for m in moves}
    return [by_id[mid] for mid in selected_ids if mid in by_id]
