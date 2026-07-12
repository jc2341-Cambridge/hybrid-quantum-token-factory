"""Surrogate-fidelity bridge between block QUBO and revised MILP."""
from __future__ import annotations

import json
from pathlib import Path
from paths import ensure_output

import numpy as np
import pandas as pd

import model as m
import qubo as qb
from revised_dispatch import solve_revised_dispatch, workload_profiles
from solvers import solve_exact

HERE = Path(__file__).parent
OUT = ensure_output()
BLOCK = 3
N_BLOCKS = m.T_SLOTS // BLOCK


def block_high_flags(levels: np.ndarray) -> np.ndarray:
    lv = levels.reshape(N_BLOCKS, BLOCK)
    return (lv.mean(axis=1) >= 1.5).astype(int)


def expand_qubo_levels(x_bits: np.ndarray) -> np.ndarray:
    levels = np.zeros(m.T_SLOTS, dtype=int)
    for b in range(N_BLOCKS):
        levels[b * BLOCK:(b + 1) * BLOCK] = 2 if x_bits[2 * b] else 0
    return levels


def repair_for_interactive(levels: np.ndarray) -> tuple[np.ndarray, int]:
    _, interactive, _ = workload_profiles()
    r = m.throughput_mtok_h(m.U_LEVELS) * m.DT_H
    repaired = levels.copy()
    lifts = 0
    for t in range(m.T_SLOTS):
        need = interactive[t]
        lv = int(repaired[t])
        while lv < 2 and r[lv] + 1e-9 < need:
            lv += 1
            lifts += 1
        repaired[t] = lv
    return repaired, lifts


def block_mean_tariff() -> np.ndarray:
    price, carbon, _, _ = m.profiles()
    tariff = m.effective_tariff(price, carbon)
    return tariff.reshape(N_BLOCKS, BLOCK).mean(axis=1)


def main() -> None:
    free = solve_revised_dispatch()
    free2 = solve_revised_dispatch()
    n_const_blocks = int(sum(
        len(set(free["levels"][b * BLOCK:(b + 1) * BLOCK].tolist())) == 1
        for b in range(N_BLOCKS)
    ))

    high_low_ok = True
    high_low_msg = ""
    high_low = None
    try:
        high_low = solve_revised_dispatch(high_low_only=True)
    except RuntimeError as exc:
        high_low_ok = False
        high_low_msg = str(exc)

    q, meta = qb.build_reduced_qubo()
    exact = solve_exact(q)
    x = exact["x"]
    qubo_high = np.array([x[2 * b] for b in range(N_BLOCKS)], dtype=int)
    qubo_dis = np.array([x[2 * b + 1] for b in range(N_BLOCKS)], dtype=int)
    qubo_levels = expand_qubo_levels(x)
    repaired, n_lifts = repair_for_interactive(qubo_levels)

    milp_high = block_high_flags(free["levels"])
    milp_dis = free["discharge"].reshape(N_BLOCKS, BLOCK).any(axis=1).astype(int)
    agree_compute = float((milp_high == qubo_high).mean())
    agree_dis = float((milp_dis == qubo_dis).mean())

    tariff_b = block_mean_tariff()
    # Tariff-arbitrage fidelity: high compute should prefer cheap blocks.
    milp_high_tariff = float(tariff_b[milp_high == 1].mean()) if milp_high.any() else np.nan
    qubo_high_tariff = float(tariff_b[qubo_high == 1].mean()) if qubo_high.any() else np.nan
    milp_low_tariff = float(tariff_b[milp_high == 0].mean()) if (~milp_high.astype(bool)).any() else np.nan
    qubo_low_tariff = float(tariff_b[qubo_high == 0].mean()) if (~qubo_high.astype(bool)).any() else np.nan
    milp_dis_tariff = float(tariff_b[milp_dis == 1].mean()) if milp_dis.any() else np.nan
    qubo_dis_tariff = float(tariff_b[qubo_dis == 1].mean()) if qubo_dis.any() else np.nan

    # Soft QUBO guidance via tiny objective preference (always feasible).
    # Implemented by re-solving with qubo_high_blocks requiring only >=1 high
    # hour in QUBO-high blocks (no restriction on low blocks).
    guided = None
    guided_msg = ""
    for minh in (1, 2):
        try:
            guided = solve_revised_dispatch(
                qubo_high_blocks=qubo_high, qubo_min_high_hours=minh
            )
            guided_min_hours = minh
            break
        except RuntimeError as exc:
            guided_msg = str(exc)
            guided_min_hours = None
    if guided is None:
        # Fall back: unrestricted MILP already computed; report infeasibility
        # of hard QUBO skeleton transfer.
        guided_ok = False
        guided_payload = {"feasible": False, "message": guided_msg}
    else:
        guided_ok = True

        def gap(a, b, key):
            return 100.0 * (a[key] - b[key]) / max(abs(b[key]), 1e-9)

        guided_payload = {
            "feasible": True,
            "min_high_hours_enforced": guided_min_hours,
            "total_cost_usd": guided["total_cost_usd"],
            "emissions_kg": guided["emissions_kg"],
            "solve_time_s": guided["solve_time_s"],
            "cost_gap_pct_vs_free": gap(guided, free, "total_cost_usd"),
            "emissions_gap_pct_vs_free": gap(guided, free, "emissions_kg"),
            "cold_start_time_s": free2["solve_time_s"],
            "time_ratio_guided_over_cold": (
                guided["solve_time_s"] / max(free2["solve_time_s"], 1e-9)
            ),
        }

    report = {
        "free_milp": {
            "total_cost_usd": free["total_cost_usd"],
            "emissions_kg": free["emissions_kg"],
            "solve_time_s": free["solve_time_s"],
            "levels": free["levels"].tolist(),
            "already_block_constant_blocks": n_const_blocks,
            "n_blocks": N_BLOCKS,
        },
        "high_low_only_milp_feasible": high_low_ok,
        "high_low_only_milp": (
            {
                "total_cost_usd": high_low["total_cost_usd"],
                "emissions_kg": high_low["emissions_kg"],
                "cost_gap_pct_vs_free": 100.0 * (
                    high_low["total_cost_usd"] - free["total_cost_usd"]
                ) / free["total_cost_usd"],
            }
            if high_low_ok
            else {"infeasible": True, "message": high_low_msg}
        ),
        "block_qubo": {
            "energy": exact["energy"],
            "time_s": exact["time_s"],
            "high_blocks": int(qubo_high.sum()),
            "dis_blocks": int(qubo_dis.sum()),
            "target_high_blocks": meta["high_blocks"],
            "target_dis_blocks": meta["dis_blocks"],
            "hourly_levels_raw": qubo_levels.tolist(),
            "interactive_lifts_if_expanded": n_lifts,
            "hours_needing_lift_for_interactive_sla": n_lifts,
        },
        "qubo_guided_milp": guided_payload,
        "agreement": {
            "compute_high_block_qubo_vs_free_milp": agree_compute,
            "discharge_block_qubo_vs_free_milp": agree_dis,
            "n_blocks": N_BLOCKS,
        },
        "tariff_arbitrage_fidelity": {
            "block_mean_tariff": tariff_b.tolist(),
            "milp_mean_tariff_on_high_blocks": milp_high_tariff,
            "qubo_mean_tariff_on_high_blocks": qubo_high_tariff,
            "milp_mean_tariff_on_low_blocks": milp_low_tariff,
            "qubo_mean_tariff_on_low_blocks": qubo_low_tariff,
            "milp_mean_tariff_on_discharge_blocks": milp_dis_tariff,
            "qubo_mean_tariff_on_discharge_blocks": qubo_dis_tariff,
            "same_cheap_high_logic": bool(
                qubo_high_tariff < qubo_low_tariff
                and milp_high_tariff < milp_low_tariff
            ),
            "same_expensive_discharge_logic": bool(
                qubo_dis_tariff > qubo_high_tariff
                and milp_dis_tariff > milp_high_tariff
            ),
        },
        "retained_in_block_qubo": [
            "block-mean effective tariff",
            "binary high/low compute mode",
            "binary discharge action",
            "equality budgets on high and discharge counts",
        ],
        "omitted_from_block_qubo": [
            "hourly interactive service constraints",
            "batch arrivals and backlog conservation",
            "maximum batch delay",
            "hourly SOC dynamics and charge actions",
            "PV netting and positive-part grid import",
            "mid utilisation level",
            "battery degradation cost",
        ],
        "omitted_from_120binary_qubo": [
            "hourly backlog state",
            "SOC dynamics",
            "deadline constraints",
            "workload-conserving equal useful work (uses legacy surplus/floor encoding)",
        ],
    }

    (OUT / "surrogate_fidelity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows = [{"scenario": "free_milp",
             "cost": free["total_cost_usd"],
             "emissions": free["emissions_kg"],
             "time_s": free["solve_time_s"]}]
    if guided_ok:
        rows.append({
            "scenario": "qubo_guided_milp",
            "cost": guided["total_cost_usd"],
            "emissions": guided["emissions_kg"],
            "time_s": guided["solve_time_s"],
        })
    pd.DataFrame(rows).to_csv(OUT / "surrogate_fidelity.csv", index=False)

    summary = {
        "agree_compute": agree_compute,
        "agree_discharge": agree_dis,
        "high_low_feasible": high_low_ok,
        "interactive_lifts": n_lifts,
        "const_blocks_in_free": n_const_blocks,
        "guided_ok": guided_ok,
        "same_cheap_high_logic": report["tariff_arbitrage_fidelity"]["same_cheap_high_logic"],
        "same_expensive_discharge_logic": report["tariff_arbitrage_fidelity"][
            "same_expensive_discharge_logic"],
        "free_cost": free["total_cost_usd"],
    }
    if guided_ok:
        summary.update({
            "cost_gap_guided_pct": guided_payload["cost_gap_pct_vs_free"],
            "emissions_gap_guided_pct": guided_payload["emissions_gap_pct_vs_free"],
            "guided_time": guided["solve_time_s"],
            "cold_time": free2["solve_time_s"],
            "guided_min_hours": guided_min_hours,
        })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
