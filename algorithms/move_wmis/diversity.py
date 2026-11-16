"""Diversity-preserving top-K pruning for token-factory moves."""
from __future__ import annotations

from typing import Any, Sequence

from .types import LocalMove, MoveKind


def _slot(move: LocalMove) -> int:
    meta = move.meta or {}
    if "slot" in meta:
        return int(meta["slot"])  # type: ignore[arg-type]
    slots = meta.get("slots")
    if isinstance(slots, (list, tuple)) and slots:
        return int(slots[0])
    return -1


def _segment(slot: int) -> str:
    if slot < 0:
        return "unknown"
    if slot < 5:
        return "early"
    if slot < 11:
        return "mid"
    return "late"


def _asset_key(move: LocalMove) -> str:
    meta = move.meta or {}
    if move.kind == MoveKind.ISLAND_LEVEL:
        return f"island:{meta.get('island')}"
    if move.kind == MoveKind.BATTERY_ACTION:
        return f"battery:{meta.get('battery')}"
    if move.kind == MoveKind.BATCH_ACTION:
        return "batch"
    if move.kind == MoveKind.SLOT_BLOCK_SWAP:
        return f"swap_island:{meta.get('island')}"
    if move.kind == MoveKind.PEAK_RELIEF:
        return f"peak:{meta.get('family')}"
    return "other"


def diversity_prune(
    moves: Sequence[LocalMove],
    *,
    k: int = 64,
    diversity_first: bool = True,
) -> tuple[list[LocalMove], dict[str, Any]]:
    """Select at most *k* moves with structural diversity then weight fill."""
    candidates = sorted(moves, key=lambda m: float(m.weight), reverse=True)
    if len(candidates) <= k:
        return list(candidates), {
            "requested_k": k,
            "selected": len(candidates),
            "diversity_first": diversity_first,
            "input": len(candidates),
        }

    selected: list[LocalMove] = []
    selected_ids: set[str] = set()

    def _take(move: LocalMove) -> None:
        if move.move_id in selected_ids or len(selected) >= k:
            return
        selected.append(move)
        selected_ids.add(move.move_id)

    if diversity_first:
        # Per asset family
        assets = {_asset_key(m) for m in candidates}
        for asset in sorted(assets):
            for move in candidates:
                if _asset_key(move) == asset:
                    _take(move)
                    break

        # Per time segment
        for segment in ("early", "mid", "late"):
            for move in candidates:
                if _segment(_slot(move)) == segment:
                    _take(move)
                    break

        # Per move kind present
        kinds = {m.kind for m in candidates}
        for kind in sorted(kinds, key=str):
            for move in candidates:
                if move.kind == kind:
                    _take(move)
                    break

    for move in candidates:
        _take(move)
        if len(selected) >= k:
            break

    report = {
        "requested_k": k,
        "selected": len(selected),
        "diversity_first": diversity_first,
        "input": len(candidates),
        "kinds": sorted({str(m.kind) for m in selected}),
        "segments": sorted({_segment(_slot(m)) for m in selected}),
    }
    return selected, report
