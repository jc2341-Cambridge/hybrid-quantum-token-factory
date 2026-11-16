"""Generate local edits around a feasible 240-bit reference window."""
from __future__ import annotations

from typing import Sequence

from algorithms import encoding

from .types import LocalMove, MoveKind

_ISLAND_ROWS = ((0, 1), (2, 3), (4, 5))
_BATTERY_ROWS = ((6, 7), (8, 9))
_LEVEL_BITS = {
    0: (0, 0),  # low / vacuum
    1: (1, 0),  # medium
    2: (0, 1),  # high
}
_BATTERY_BITS = {
    "idle": (0, 0),
    "charge": (1, 0),
    "discharge": (0, 1),
}
_BATCH_BITS = {
    "work_conserving": (0, 0),
    "deadline_only": (1, 0),
    "defer": (0, 1),
}


def _bits_tuple(bits: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(v) for v in bits)
    if len(values) != encoding.N_SITES:
        raise ValueError(f"expected {encoding.N_SITES} bits, got {len(values)}")
    if any(v not in (0, 1) for v in values):
        raise ValueError("bits must be binary")
    return values


def _level_at(bits: tuple[int, ...], slot: int, island: int) -> int:
    medium_row, high_row = _ISLAND_ROWS[island]
    medium = bits[encoding.atom_index(slot, medium_row)]
    high = bits[encoding.atom_index(slot, high_row)]
    if medium and high:
        raise ValueError("invalid medium/high double selection in reference")
    return 2 if high else 1 if medium else 0


def _battery_at(bits: tuple[int, ...], slot: int, battery: int) -> str:
    charge_row, discharge_row = _BATTERY_ROWS[battery]
    charge = bits[encoding.atom_index(slot, charge_row)]
    discharge = bits[encoding.atom_index(slot, discharge_row)]
    if charge and discharge:
        raise ValueError("invalid charge/discharge double selection in reference")
    return "charge" if charge else "discharge" if discharge else "idle"


def _batch_at(bits: tuple[int, ...], slot: int) -> str:
    deadline = bits[encoding.atom_index(slot, 10)]
    defer = bits[encoding.atom_index(slot, 11)]
    if deadline and defer:
        raise ValueError("invalid deadline/defer double selection in reference")
    return "deadline_only" if deadline else "defer" if defer else "work_conserving"


def _set_pair_flips(
    bits: tuple[int, ...],
    slot: int,
    rows: tuple[int, int],
    target: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    flips: list[tuple[int, int]] = []
    for row, value in zip(rows, target):
        index = encoding.atom_index(slot, row)
        if bits[index] != value:
            flips.append((index, value))
    return tuple(flips)


def generate_window_moves(
    bits: Sequence[int],
    *,
    include_block_swaps: bool = True,
) -> list[LocalMove]:
    """Enumerate local edits that differ from the reference independent set.

    Guard/marker atoms are never flipped: they carry no feasibility certificate.
    """
    reference = _bits_tuple(bits)
    if not encoding.is_independent(reference):
        raise ValueError("reference must be a physical independent set")

    moves: list[LocalMove] = []

    for slot in range(encoding.WINDOW_SLOTS):
        for island in range(3):
            current = _level_at(reference, slot, island)
            for target in (0, 1, 2):
                if target == current:
                    continue
                flips = _set_pair_flips(
                    reference, slot, _ISLAND_ROWS[island], _LEVEL_BITS[target]
                )
                if not flips:
                    continue
                moves.append(
                    LocalMove(
                        move_id=f"isl_i{island}_s{slot}_{current}to{target}",
                        kind=MoveKind.ISLAND_LEVEL,
                        touched_keys=frozenset({f"compute:{island}:{slot}"}),
                        flips=flips,
                        meta={"island": island, "slot": slot, "from": current, "to": target},
                    )
                )

        for battery in range(2):
            current = _battery_at(reference, slot, battery)
            for target in ("idle", "charge", "discharge"):
                if target == current:
                    continue
                flips = _set_pair_flips(
                    reference, slot, _BATTERY_ROWS[battery], _BATTERY_BITS[target]
                )
                if not flips:
                    continue
                moves.append(
                    LocalMove(
                        move_id=f"bat_b{battery}_s{slot}_{current}_to_{target}",
                        kind=MoveKind.BATTERY_ACTION,
                        touched_keys=frozenset({f"battery:{battery}:{slot}"}),
                        flips=flips,
                        meta={"battery": battery, "slot": slot, "from": current, "to": target},
                    )
                )

        current_batch = _batch_at(reference, slot)
        for target in ("work_conserving", "deadline_only", "defer"):
            if target == current_batch:
                continue
            flips = _set_pair_flips(reference, slot, (10, 11), _BATCH_BITS[target])
            if not flips:
                continue
            moves.append(
                LocalMove(
                    move_id=f"batch_s{slot}_{current_batch}_to_{target}",
                    kind=MoveKind.BATCH_ACTION,
                    touched_keys=frozenset({f"batch:{slot}"}),
                    flips=flips,
                    meta={"slot": slot, "from": current_batch, "to": target},
                )
            )

    if include_block_swaps:
        for slot in range(encoding.WINDOW_SLOTS - 1):
            other = slot + 1
            for island in range(3):
                level_a = _level_at(reference, slot, island)
                level_b = _level_at(reference, other, island)
                if level_a == level_b:
                    continue
                flips = (
                    _set_pair_flips(
                        reference, slot, _ISLAND_ROWS[island], _LEVEL_BITS[level_b]
                    )
                    + _set_pair_flips(
                        reference, other, _ISLAND_ROWS[island], _LEVEL_BITS[level_a]
                    )
                )
                moves.append(
                    LocalMove(
                        move_id=f"swap_isl_i{island}_s{slot}_{other}",
                        kind=MoveKind.SLOT_BLOCK_SWAP,
                        touched_keys=frozenset(
                            {
                                f"compute:{island}:{slot}",
                                f"compute:{island}:{other}",
                            }
                        ),
                        flips=flips,
                        meta={
                            "island": island,
                            "slots": (slot, other),
                            "levels": (level_a, level_b),
                        },
                    )
                )

    return moves
