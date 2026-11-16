"""Sampling helpers on a move-conflict graph.

The AHS-labelled routine is an ideal local-detuning hard-core surrogate. It
does not represent Aquila hardware counts or timing.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np

from .types import LocalMove


@dataclass(frozen=True)
class GraphSampleBatch:
    states: tuple[frozenset[str], ...]
    values: tuple[float, ...]
    objective_evaluations: int
    wall_seconds: float
    method: str

    @property
    def best_state(self) -> frozenset[str]:
        if not self.states:
            return frozenset()
        return self.states[int(np.argmax(self.values))]

    @property
    def best_value(self) -> float:
        return 0.0 if not self.values else float(max(self.values))


def _construct(
    ids: Sequence[str],
    priorities: np.ndarray,
    weights: dict[str, float],
    adjacency: dict[str, set[str]],
) -> frozenset[str]:
    selected: set[str] = set()
    blocked: set[str] = set()
    for raw_index in np.argsort(-priorities, kind="stable"):
        move_id = ids[int(raw_index)]
        if weights[move_id] <= 0.0 or move_id in blocked:
            continue
        selected.add(move_id)
        blocked.add(move_id)
        blocked.update(adjacency.get(move_id, set()))
    return frozenset(selected)


def sample_ahs_surrogate(
    moves: Sequence[LocalMove],
    adjacency: dict[str, set[str]],
    *,
    outcomes: int = 128,
    detuning_beta: float = 1.5,
    seed: int = 42,
) -> GraphSampleBatch:
    """Emit direct independent sets under ideal randomized local detunings."""
    if outcomes <= 0:
        raise ValueError("outcomes must be positive")
    started = time.perf_counter()
    ids = [move.move_id for move in moves]
    weights = {move.move_id: float(move.weight) for move in moves}
    positive = np.asarray([max(weights[mid], 0.0) for mid in ids], dtype=float)
    mean_positive = float(np.mean(positive[positive > 0.0])) if np.any(positive > 0.0) else 1.0
    normalized = positive / max(mean_positive, 1e-12)
    rng = np.random.default_rng(seed)
    states: list[frozenset[str]] = []
    values: list[float] = []
    for _ in range(outcomes):
        priorities = detuning_beta * normalized + rng.gumbel(size=len(ids))
        state = _construct(ids, priorities, weights, adjacency)
        states.append(state)
        values.append(
            float(sum(weights[mid] for mid in sorted(state)))
        )
    return GraphSampleBatch(
        states=tuple(states),
        values=tuple(values),
        objective_evaluations=outcomes,
        wall_seconds=time.perf_counter() - started,
        method="ideal_local_detuning_ahs_surrogate",
    )
