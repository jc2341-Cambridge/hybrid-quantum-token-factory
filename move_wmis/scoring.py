"""Dual / multi-channel scoring for token-factory moves."""
from __future__ import annotations

from typing import Mapping, Sequence

from .types import LocalMove, MoveKind


def _slot_from_meta(move: LocalMove) -> int | None:
    meta = move.meta or {}
    if "slot" in meta:
        return int(meta["slot"])  # type: ignore[arg-type]
    if "slots" in meta:
        slots = meta["slots"]
        if isinstance(slots, (list, tuple)) and slots:
            return int(slots[0])
    return None


def score_move(
    move: LocalMove,
    *,
    reference_bits: Sequence[int] | None = None,
    site_weights: Sequence[float] | None = None,
    price_usd_kwh: Sequence[float] | None = None,
    carbon_g_kwh: Sequence[float] | None = None,
    dual_slot: Mapping[int, float] | None = None,
    price_weight: float = 1.0,
    carbon_weight: float = 0.0,
    dual_weight: float = 1.0,
) -> float:
    """Heuristic reduced-cost proxy for a local edit.

    Positive scores are preferred by MWIS. This is proposer-only: exact
    admission remains the Iteration5 unsmoothed reduced-cost gate.
    """
    slot = _slot_from_meta(move)
    score = 0.0

    using_site_weights = site_weights is not None
    if site_weights is not None:
        if reference_bits is None:
            raise ValueError("reference_bits are required with site_weights")
        if len(reference_bits) != len(site_weights):
            raise ValueError("reference_bits and site_weights must have equal length")
        # dual_pricing_weights assigns each site the benefit of its included
        # state. A move's additive benefit relative to the reference is
        # therefore (target - reference) * site_weight.
        score += sum(
            (int(target) - int(reference_bits[index])) * float(site_weights[index])
            for index, target in move.flips
        )

    # The signed site weights already include direct and RMP-dual channels.
    # Do not mix their scale with the fallback price/carbon heuristics.
    if (
        not using_site_weights
        and dual_slot is not None
        and slot is not None
        and dual_weight
    ):
        meta = move.meta or {}
        if move.kind == MoveKind.SLOT_BLOCK_SWAP:
            slots = meta.get("slots", ())
            levels = meta.get("levels", ())
            if (
                isinstance(slots, (list, tuple))
                and isinstance(levels, (list, tuple))
                and len(slots) == 2
                and len(levels) == 2
            ):
                s0, s1 = int(slots[0]), int(slots[1])
                l0, l1 = int(levels[0]), int(levels[1])
                score += dual_weight * (
                    (l1 - l0) * float(dual_slot.get(s0, 0.0))
                    + (l0 - l1) * float(dual_slot.get(s1, 0.0))
                )
        else:
            score += dual_weight * float(dual_slot.get(slot, 0.0))

    if (
        not using_site_weights
        and price_usd_kwh is not None
        and slot is not None
        and price_weight
    ):
        price = float(price_usd_kwh[slot])
        meta = move.meta or {}
        if move.kind == MoveKind.ISLAND_LEVEL:
            # Prefer lowering compute when price is high.
            delta = int(meta.get("from", 0)) - int(meta.get("to", 0))
            score += price_weight * price * float(delta)
        elif move.kind == MoveKind.BATTERY_ACTION:
            # Prefer charge at low price, discharge at high price.
            target = str(meta.get("to", "idle"))
            if target == "charge":
                score += price_weight * (1.0 - price)
            elif target == "discharge":
                score += price_weight * price
        elif move.kind == MoveKind.BATCH_ACTION:
            target = str(meta.get("to", "work_conserving"))
            if target == "defer":
                score += price_weight * price
            elif target == "deadline_only":
                score += 0.25 * price_weight * price
        elif move.kind == MoveKind.SLOT_BLOCK_SWAP and price_usd_kwh is not None:
            slots = meta.get("slots", ())
            levels = meta.get("levels", ())
            if (
                isinstance(slots, (list, tuple))
                and isinstance(levels, (list, tuple))
                and len(slots) == 2
                and len(levels) == 2
            ):
                s0, s1 = int(slots[0]), int(slots[1])
                l0, l1 = int(levels[0]), int(levels[1])
                before = l0 * float(price_usd_kwh[s0]) + l1 * float(price_usd_kwh[s1])
                after = l1 * float(price_usd_kwh[s0]) + l0 * float(price_usd_kwh[s1])
                score += price_weight * (before - after)

    if (
        not using_site_weights
        and carbon_g_kwh is not None
        and slot is not None
        and carbon_weight
    ):
        carbon = float(carbon_g_kwh[slot])
        meta = move.meta or {}
        if move.kind == MoveKind.ISLAND_LEVEL:
            delta = int(meta.get("from", 0)) - int(meta.get("to", 0))
            score += carbon_weight * carbon * float(delta)

    return float(score)


def score_moves(
    moves: Sequence[LocalMove],
    *,
    reference_bits: Sequence[int] | None = None,
    site_weights: Sequence[float] | None = None,
    price_usd_kwh: Sequence[float] | None = None,
    carbon_g_kwh: Sequence[float] | None = None,
    dual_slot: Mapping[int, float] | None = None,
    price_weight: float = 1.0,
    carbon_weight: float = 0.0,
    dual_weight: float = 1.0,
) -> list[LocalMove]:
    return [
        move.with_weight(
            score_move(
                move,
                reference_bits=reference_bits,
                site_weights=site_weights,
                price_usd_kwh=price_usd_kwh,
                carbon_g_kwh=carbon_g_kwh,
                dual_slot=dual_slot,
                price_weight=price_weight,
                carbon_weight=carbon_weight,
                dual_weight=dual_weight,
            )
        )
        for move in moves
    ]
