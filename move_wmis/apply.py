"""Apply selected moves to a reference bitstring without repair."""
from __future__ import annotations

from typing import Sequence

from integrated_model import encoding

from .conflict import moves_conflict
from .types import LocalMove


def apply_moves_to_bits(
    bits: Sequence[int],
    moves: Sequence[LocalMove],
    *,
    verify_independent: bool = True,
    verify_conflict_free: bool = True,
) -> tuple[int, ...]:
    """Apply a conflict-free move set to the reference independent set.

    Guard atoms are left unchanged by construction of the generators. This
    function never repairs conflicts; it raises if the result is invalid.
    """
    values = [int(v) for v in bits]
    if len(values) != encoding.N_SITES:
        raise ValueError(f"expected {encoding.N_SITES} bits")

    move_list = list(moves)
    if verify_conflict_free:
        for i, left in enumerate(move_list):
            for right in move_list[i + 1 :]:
                if moves_conflict(left, right, reference_bits=bits):
                    raise ValueError(
                        f"selected moves conflict: {left.move_id} vs {right.move_id}"
                    )

    for move in move_list:
        for index, target in move.flips:
            if not 0 <= index < encoding.N_SITES:
                raise ValueError(f"flip index out of range: {index}")
            values[index] = int(target)

    result = tuple(values)
    if verify_independent and not encoding.is_independent(result):
        raise ValueError("applied moves destroyed physical independence")
    # Decode also checks logical pair exclusions.
    if verify_independent:
        encoding.decode_independent_set(result, verify_independent=True)
    return result
