"""Move-space simulated annealing on the conflict graph."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .types import LocalMove


@dataclass
class ConvergenceMetrics:
    best_state: list[str]
    best_energy: float
    best_weight_sum: float
    n_iterations: int
    final_temperature: float
    history: list[tuple[int, float, float]] = field(default_factory=list)
    n_feasible_states: int = 0
    warm_start_used: bool = False
    iterations_to_95pct: int | None = None
    iterations_to_99pct: int | None = None


class MoveSpaceSA:
    """Metropolis SA over independent sets of the move conflict graph."""

    def __init__(
        self,
        moves: Sequence[LocalMove],
        conflict_graph: dict[str, set[str]],
        *,
        initial_state: list[str] | None = None,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.995,
        min_temp: float = 0.01,
        steps_per_temp: int = 50,
        max_iterations: int = 10_000,
        random_seed: int = 42,
    ) -> None:
        self.moves = list(moves)
        self.weights = {m.move_id: float(m.weight) for m in self.moves}
        self.conflict_graph = {
            mid: set(neighbours) for mid, neighbours in conflict_graph.items()
        }
        for mid in self.weights:
            self.conflict_graph.setdefault(mid, set())
        self.initial_state = list(initial_state or [])
        self.initial_temp = float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.min_temp = float(min_temp)
        self.steps_per_temp = int(steps_per_temp)
        self.max_iterations = int(max_iterations)
        self.rng = random.Random(random_seed)
        self._ids = [m.move_id for m in self.moves]

    def _is_feasible(self, state: set[str]) -> bool:
        for mid in state:
            if self.conflict_graph.get(mid, set()) & state:
                return False
        return True

    def _energy(self, state: set[str]) -> float:
        return -sum(
            self.weights.get(mid, 0.0) for mid in sorted(state)
        )

    def _propose(self, state: set[str]) -> set[str]:
        if not self._ids:
            return set(state)
        candidate = set(state)
        op = self.rng.random()
        mid = self.rng.choice(self._ids)
        if op < 0.4:
            # flip membership
            if mid in candidate:
                candidate.remove(mid)
            else:
                candidate -= self.conflict_graph.get(mid, set())
                candidate.add(mid)
        elif op < 0.7 and candidate:
            # replace one selected move
            old = self.rng.choice(sorted(candidate))
            candidate.remove(old)
            candidate -= self.conflict_graph.get(mid, set())
            candidate.add(mid)
        else:
            # add if possible else drop random
            if mid not in candidate and not (self.conflict_graph.get(mid, set()) & candidate):
                candidate.add(mid)
            elif candidate:
                candidate.remove(self.rng.choice(sorted(candidate)))
        return candidate

    def run(self) -> ConvergenceMetrics:
        state = {mid for mid in self.initial_state if mid in self.weights}
        if not self._is_feasible(state):
            # Drop conflicting warm-start members greedily by weight.
            ordered = sorted(
                state,
                key=lambda mid: (-self.weights.get(mid, 0.0), mid),
            )
            state = set()
            for mid in ordered:
                if not (self.conflict_graph.get(mid, set()) & state):
                    state.add(mid)

        best = set(state)
        best_energy = self._energy(best)
        temperature = self.initial_temp
        history: list[tuple[int, float, float]] = []
        feasible = 0
        iteration = 0

        while temperature > self.min_temp and iteration < self.max_iterations:
            for _ in range(self.steps_per_temp):
                iteration += 1
                proposal = self._propose(state)
                if not self._is_feasible(proposal):
                    continue
                feasible += 1
                current_e = self._energy(state)
                proposal_e = self._energy(proposal)
                delta = proposal_e - current_e
                if delta <= 0.0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                    state = proposal
                    if proposal_e < best_energy:
                        best = set(proposal)
                        best_energy = proposal_e
                if iteration >= self.max_iterations:
                    break
            history.append((iteration, best_energy, temperature))
            temperature *= self.cooling_rate

        weight_sum = -best_energy
        to_95 = to_99 = None
        if weight_sum > 0:
            for step, energy, _temp in history:
                achieved = -energy
                if to_95 is None and achieved >= 0.95 * weight_sum:
                    to_95 = step
                if to_99 is None and achieved >= 0.99 * weight_sum:
                    to_99 = step

        return ConvergenceMetrics(
            best_state=sorted(best),
            best_energy=float(best_energy),
            best_weight_sum=float(weight_sum),
            n_iterations=iteration,
            final_temperature=float(temperature),
            history=history,
            n_feasible_states=feasible,
            warm_start_used=bool(self.initial_state),
            iterations_to_95pct=to_95,
            iterations_to_99pct=to_99,
        )


def compare_convergence(warm: ConvergenceMetrics, cold: ConvergenceMetrics) -> dict:
    return {
        "warm_weight_sum": warm.best_weight_sum,
        "cold_weight_sum": cold.best_weight_sum,
        "warm_better": warm.best_weight_sum > cold.best_weight_sum,
        "warm_iterations_to_95pct": warm.iterations_to_95pct,
        "cold_iterations_to_95pct": cold.iterations_to_95pct,
        "warm_n_selected": len(warm.best_state),
        "cold_n_selected": len(cold.best_state),
    }
