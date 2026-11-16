"""Local edit types for the token-factory 240-site window."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class MoveKind(StrEnum):
    ISLAND_LEVEL = "island_level"
    BATTERY_ACTION = "battery_action"
    BATCH_ACTION = "batch_action"
    SLOT_BLOCK_SWAP = "slot_block_swap"
    PEAK_RELIEF = "peak_relief"


@dataclass(frozen=True)
class LocalMove:
    """One conservative local edit relative to a feasible reference sample."""

    move_id: str
    kind: MoveKind
    touched_keys: frozenset[str]
    flips: tuple[tuple[int, int], ...]
    """Pairs ``(site_index, target_bit)`` applied atomically to the reference."""

    weight: float = 0.0
    meta: Mapping[str, object] | None = None

    def with_weight(self, weight: float) -> LocalMove:
        return LocalMove(
            move_id=self.move_id,
            kind=self.kind,
            touched_keys=self.touched_keys,
            flips=self.flips,
            weight=float(weight),
            meta=self.meta,
        )
