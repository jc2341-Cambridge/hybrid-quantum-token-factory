"""Per-window Mode-B proposal harness.

This small harness does not own or propagate ``BoundaryState``. The canonical
30-day boundary-chained implementation is ``run_mode_b_stress.py``, which
calls ``columns.evaluate_sample`` and threads every accepted exit boundary.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .pipeline import run_move_wmis_pipeline


def run_mode_b_windows(
    windows: Sequence[Mapping[str, object]],
    *,
    top_k: int = 64,
    sa_seed: int = 42,
    evaluate_fn: Callable[[Sequence[int], Mapping[str, object]], bool] | None = None,
) -> dict:
    """Roll across windows; on evaluation failure, retry move-WMIS once.

    Each window mapping must provide ``reference_bits`` and may provide
    ``price_usd_kwh``, ``carbon_g_kwh``, and ``dual_slot``.
    """
    outcomes = []
    n_pass = 0
    n_recourse = 0
    n_fail = 0

    for index, window in enumerate(windows):
        reference = window["reference_bits"]  # type: ignore[index]
        report = run_move_wmis_pipeline(
            reference,  # type: ignore[arg-type]
            price_usd_kwh=window.get("price_usd_kwh"),  # type: ignore[arg-type]
            carbon_g_kwh=window.get("carbon_g_kwh"),  # type: ignore[arg-type]
            dual_slot=window.get("dual_slot"),  # type: ignore[arg-type]
            top_k=top_k,
            sa_seed=sa_seed + index,
        )
        bits = report["result_bits"]
        ok = True
        if evaluate_fn is not None:
            ok = bool(evaluate_fn(bits, window))
        used_recourse = False
        if not ok:
            used_recourse = True
            n_recourse += 1
            # Recourse: regenerate with a shifted seed and no block swaps to
            # shrink the neighbourhood toward conservative edits.
            report = run_move_wmis_pipeline(
                reference,  # type: ignore[arg-type]
                price_usd_kwh=window.get("price_usd_kwh"),  # type: ignore[arg-type]
                carbon_g_kwh=window.get("carbon_g_kwh"),  # type: ignore[arg-type]
                dual_slot=window.get("dual_slot"),  # type: ignore[arg-type]
                top_k=top_k,
                sa_seed=sa_seed + 10_000 + index,
                include_block_swaps=False,
            )
            bits = report["result_bits"]
            ok = True if evaluate_fn is None else bool(evaluate_fn(bits, window))

        if ok:
            n_pass += 1
        else:
            n_fail += 1
        outcomes.append(
            {
                "window_index": index,
                "accepted": ok,
                "recourse": used_recourse,
                "n_selected": report["n_selected"],
                "n_changed_sites": len(report["changed_sites"]),
            }
        )

    return {
        "n_windows": len(windows),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_recourse": n_recourse,
        "pass_rate": (n_pass / len(windows)) if windows else 0.0,
        "outcomes": outcomes,
    }
