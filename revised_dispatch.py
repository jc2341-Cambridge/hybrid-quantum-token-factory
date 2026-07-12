"""Revised workload-conserving dispatch model.

This model closes the three structural gaps identified in review:
  * finite interactive and batch arrivals, with no service before arrival;
  * a maximum-delay constraint and zero terminal backlog;
  * hourly battery state-of-charge dynamics and degradation cost.

The useful workload is fixed at the BAU demand total.  The objective is
therefore total operating cost plus an internal carbon charge, not a
fractional cost that can be diluted by surplus production.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

import model as m


BATCH_SHARE = 0.30
MAX_BATCH_DELAY_H = 6
BATTERY_CAPACITY_KWH = 2000.0
SOC_MIN_KWH = 200.0
SOC_MAX_KWH = 1800.0
SOC_INITIAL_KWH = 1000.0
ETA_CH = float(np.sqrt(0.90))
ETA_DIS = float(np.sqrt(0.90))
MAX_CHARGE_SLOTS = 4
MAX_DISCHARGE_SLOTS = 4
DEGRADATION_USD_PER_KWH_DISCHARGED = 0.03
GRID_IMPORT_LIMIT_KW = 2000.0


def workload_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return total, interactive, and batch arrivals in Mtoken per slot."""
    total = m.throughput_mtok_h(m.U_LEVELS[m.BAU_LEVELS]) * m.DT_H
    batch = BATCH_SHARE * total
    interactive = total - batch
    return total, interactive, batch


def solve_revised_dispatch(
    carbon_price: float | None = None,
    pv: np.ndarray | None = None,
    price: np.ndarray | None = None,
    carbon: np.ndarray | None = None,
    max_batch_delay_h: int = MAX_BATCH_DELAY_H,
    no_battery: bool = False,
    fixed_levels: np.ndarray | None = None,
    block_hours: int = 3,
    block_compute: bool = False,
    high_low_only: bool = False,
    qubo_high_blocks: np.ndarray | None = None,
    qubo_min_high_hours: int = 2,
    qubo_dis_blocks: np.ndarray | None = None,
    qubo_ch_blocks: np.ndarray | None = None,
    allow_infeasible: bool = False,
) -> dict:
    """Solve the workload-conserving dispatch as a MILP.

    Optional surrogate-fidelity controls:
      fixed_levels        : force hourly compute level indices
      block_compute       : force levels constant on successive block_hours slots
      high_low_only       : forbid the mid utilisation level
      qubo_high_blocks    : length-8 0/1 mask; require qubo_min_high_hours of
                            level-2 slots inside marked high blocks
    """
    T, K = m.T_SLOTS, len(m.U_LEVELS)
    price0, carbon0, pv0, t_amb = m.profiles(T)
    price = price0 if price is None else np.asarray(price, dtype=float)
    carbon = carbon0 if carbon is None else np.asarray(carbon, dtype=float)
    pv = pv0 if pv is None else np.asarray(pv, dtype=float)
    lam = m.CARBON_PRICE if carbon_price is None else carbon_price

    total_arr, interactive, batch_arr = workload_profiles()
    # Raw throughput keeps the contracted 3024 Mtok workload feasible.
    # Path B enters as an SLO shortfall penalty on levels with phi_mix < 1
    # (fixed class SLAs + TTFT contention), so high utilisation yields less
    # useful work per raw token and is economically disfavoured.
    r = np.array([float(m.R_MAX_MTOK_H * u) for u in m.U_LEVELS], dtype=float)
    phi = np.array([float(m.phi_mix(u)) for u in m.U_LEVELS], dtype=float)
    r_raw = r.copy()
    # USD per Mtoken of raw output that fails the mix-weighted class SLAs.
    SLO_SHORTFALL_USD_PER_MTOK = 25.0
    p_fac = np.array([
        [float(m.facility_power_kw(u, t_amb[t])) for u in m.U_LEVELS]
        for t in range(T)
    ])

    # Binary block: x_tk, charge c_t, discharge d_t.
    nb = T * (K + 2)
    # Continuous block: grid g_t, SOC s_0...s_T.
    n = nb + T + T + 1
    x_idx = lambda t, k: t * (K + 2) + k
    c_idx = lambda t: t * (K + 2) + K
    d_idx = lambda t: t * (K + 2) + K + 1
    g_idx = lambda t: nb + t
    s_idx = lambda t: nb + T + t

    obj = np.zeros(n)
    tariff = price + lam * carbon
    for t in range(T):
        obj[g_idx(t)] = tariff[t] * m.DT_H
        obj[d_idx(t)] += (
            DEGRADATION_USD_PER_KWH_DISCHARGED * m.P_DIS_KW * m.DT_H
        )
        for k in range(K):
            # Path B: penalise raw tokens that are not SLO-feasible.
            obj[x_idx(t, k)] += (
                SLO_SHORTFALL_USD_PER_MTOK * (1.0 - phi[k]) * r_raw[k]
            )

    A: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    def add(a: np.ndarray, lo: float, hi: float) -> None:
        A.append(a)
        lb.append(lo)
        ub.append(hi)

    for t in range(T):
        a = np.zeros(n)
        for k in range(K):
            a[x_idx(t, k)] = 1.0
        add(a, 1.0, 1.0)

        a = np.zeros(n)
        a[c_idx(t)] = 1.0
        a[d_idx(t)] = 1.0
        add(a, 0.0, 1.0)

        # Interactive arrivals must be served in their arrival slot.
        a = np.zeros(n)
        for k in range(K):
            a[x_idx(t, k)] = r[k]
        add(a, interactive[t], np.inf)

        # Non-negative grid import linearises [net load - PV]^+.
        a = np.zeros(n)
        a[g_idx(t)] = 1.0
        for k in range(K):
            a[x_idx(t, k)] = -p_fac[t, k]
        a[c_idx(t)] = -m.P_CH_KW
        a[d_idx(t)] = m.P_DIS_KW
        add(a, -pv[t], np.inf)

        # Hourly battery energy conservation.
        a = np.zeros(n)
        a[s_idx(t + 1)] = 1.0
        a[s_idx(t)] = -1.0
        a[c_idx(t)] = -ETA_CH * m.P_CH_KW * m.DT_H
        a[d_idx(t)] = (m.P_DIS_KW / ETA_DIS) * m.DT_H
        add(a, 0.0, 0.0)

    # Cumulative batch service.  Excess capacity above interactive service
    # is batch service; it cannot precede arrivals.
    for t in range(T):
        a = np.zeros(n)
        for tau in range(t + 1):
            for k in range(K):
                a[x_idx(tau, k)] += r[k]
        served_offset = float(interactive[: t + 1].sum())
        arrived = float(batch_arr[: t + 1].sum())
        add(a, -np.inf, served_offset + arrived)

        # Any batch work older than max_batch_delay_h must be complete.
        due = t - max_batch_delay_h
        if due >= 0:
            due_arrivals = float(batch_arr[: due + 1].sum())
            add(a.copy(), served_offset + due_arrivals, np.inf)

    # Complete the finite useful workload and clear backlog by horizon end.
    a = np.zeros(n)
    for t in range(T):
        for k in range(K):
            a[x_idx(t, k)] = r[k]
    add(a, float(total_arr.sum()), float(total_arr.sum()))

    # SOC boundary conditions and daily throughput caps.
    a = np.zeros(n)
    a[s_idx(0)] = 1.0
    add(a, SOC_INITIAL_KWH, SOC_INITIAL_KWH)
    a = np.zeros(n)
    a[s_idx(T)] = 1.0
    add(a, SOC_INITIAL_KWH, SOC_INITIAL_KWH)
    for idx_fun, cap in (
        (c_idx, 0 if no_battery else MAX_CHARGE_SLOTS),
        (d_idx, 0 if no_battery else MAX_DISCHARGE_SLOTS),
    ):
        a = np.zeros(n)
        for t in range(T):
            a[idx_fun(t)] = 1.0
        add(a, 0.0, cap)

    # Surrogate-fidelity restrictions that keep backlog/SOC exact.
    if high_low_only:
        for t in range(T):
            a = np.zeros(n)
            a[x_idx(t, 1)] = 1.0
            add(a, 0.0, 0.0)
    if block_compute:
        for b0 in range(0, T, block_hours):
            for t in range(b0 + 1, min(b0 + block_hours, T)):
                for k in range(K):
                    a = np.zeros(n)
                    a[x_idx(b0, k)] = 1.0
                    a[x_idx(t, k)] = -1.0
                    add(a, 0.0, 0.0)
    if fixed_levels is not None:
        fixed_levels = np.asarray(fixed_levels, dtype=int)
        for t, lv in enumerate(fixed_levels):
            a = np.zeros(n)
            a[x_idx(t, int(lv))] = 1.0
            add(a, 1.0, 1.0)
    if qubo_high_blocks is not None:
        mask = np.asarray(qubo_high_blocks, dtype=int)
        for b, flag in enumerate(mask):
            hours = range(b * block_hours, min((b + 1) * block_hours, T))
            a = np.zeros(n)
            for t in hours:
                a[x_idx(t, K - 1)] = 1.0
            if flag:
                add(a, float(qubo_min_high_hours), float(len(list(hours))))
            else:
                add(a, 0.0, 1.0)
    if qubo_dis_blocks is not None:
        mask = np.asarray(qubo_dis_blocks, dtype=int)
        for b, flag in enumerate(mask):
            hours = list(range(b * block_hours, min((b + 1) * block_hours, T)))
            a = np.zeros(n)
            for t in hours:
                a[d_idx(t)] = 1.0
            if flag:
                add(a, 1.0, float(len(hours)))
            else:
                add(a, 0.0, 0.0)
    if qubo_ch_blocks is not None:
        mask = np.asarray(qubo_ch_blocks, dtype=int)
        for b, flag in enumerate(mask):
            hours = list(range(b * block_hours, min((b + 1) * block_hours, T)))
            a = np.zeros(n)
            for t in hours:
                a[c_idx(t)] = 1.0
            if flag:
                add(a, 1.0, float(len(hours)))
            else:
                add(a, 0.0, 0.0)

    lower = np.zeros(n)
    upper = np.full(n, np.inf)
    upper[:nb] = 1.0
    upper[nb : nb + T] = GRID_IMPORT_LIMIT_KW
    lower[nb + T : nb + T + T + 1] = SOC_MIN_KWH
    upper[nb + T : nb + T + T + 1] = SOC_MAX_KWH
    integrality = np.concatenate([np.ones(nb), np.zeros(T + T + 1)])

    t0 = time.perf_counter()
    res = milp(
        c=obj,
        constraints=LinearConstraint(np.asarray(A), np.asarray(lb), np.asarray(ub)),
        integrality=integrality,
        bounds=Bounds(lower, upper),
        options={"time_limit": 120.0},
    )
    elapsed = time.perf_counter() - t0
    if res.x is None:
        if allow_infeasible:
            return {
                "success": False,
                "feasible": False,
                "message": str(res.message),
                "solve_time_s": elapsed,
                "total_cost_usd": np.inf,
                "emissions_kg": np.inf,
                "objective": np.inf,
            }
        raise RuntimeError(f"Revised MILP failed: {res.message}")

    xb = np.rint(res.x[:nb]).astype(int)
    levels = np.array([
        int(np.argmax([xb[x_idx(t, k)] for k in range(K)]))
        for t in range(T)
    ])
    charge = np.array([xb[c_idx(t)] for t in range(T)])
    discharge = np.array([xb[d_idx(t)] for t in range(T)])
    soc = np.array([res.x[s_idx(t)] for t in range(T + 1)])
    grid = np.array([res.x[g_idx(t)] for t in range(T)])
    throughput = r[levels]
    throughput_raw = r_raw[levels]
    batch_service = throughput - interactive
    backlog = np.cumsum(batch_arr - batch_service)

    energy_cost = float((price * grid * m.DT_H).sum())
    emissions_kg = float((carbon * grid * m.DT_H).sum() / 1e3)
    degradation_cost = float(
        DEGRADATION_USD_PER_KWH_DISCHARGED
        * m.P_DIS_KW
        * m.DT_H
        * discharge.sum()
    )
    slo_shortfall_cost = float(
        SLO_SHORTFALL_USD_PER_MTOK
        * ((1.0 - phi[levels]) * throughput_raw).sum()
    )
    useful_tokens = float(total_arr.sum())
    effective_tokens = float((phi[levels] * throughput_raw).sum())
    return {
        "levels": levels,
        "charge": charge,
        "discharge": discharge,
        "soc_kwh": soc,
        "grid_kw": grid,
        "interactive_arrivals_Mtok": interactive,
        "batch_arrivals_Mtok": batch_arr,
        "batch_service_Mtok": batch_service,
        "backlog_Mtok": backlog,
        "useful_tokens_Mtok": useful_tokens,
        "effective_tokens_Mtok": effective_tokens,
        "raw_tokens_Mtok": float(throughput_raw.sum()),
        "energy_cost_usd": energy_cost,
        "degradation_cost_usd": degradation_cost,
        "slo_shortfall_cost_usd": slo_shortfall_cost,
        "total_cost_usd": energy_cost + degradation_cost,
        "emissions_kg": emissions_kg,
        "usd_per_Mtok": (energy_cost + degradation_cost) / useful_tokens,
        "gco2_per_Mtok": emissions_kg * 1e3 / useful_tokens,
        "peak_grid_kw": float(grid.max()),
        "evening_grid_kwh": float(grid[17:22].sum() * m.DT_H),
        "objective": float(res.fun),
        "solve_time_s": elapsed,
        "mip_gap": float(getattr(res, "mip_gap", np.nan)),
        "success": bool(res.success),
        "feasible": True,
        "message": res.message,
    }


def baseline_metrics() -> dict:
    """Demand-tracking baseline with no battery and the same useful workload."""
    price, carbon, pv, t_amb = m.profiles(m.T_SLOTS)
    levels = m.BAU_LEVELS
    load = m.facility_power_kw(m.U_LEVELS[levels], t_amb)
    grid = np.maximum(load - pv, 0.0)
    total, interactive, batch = workload_profiles()
    cost = float((price * grid * m.DT_H).sum())
    emissions = float((carbon * grid * m.DT_H).sum() / 1e3)
    useful = float(total.sum())
    return {
        "levels": levels.copy(),
        "grid_kw": grid,
        "interactive_arrivals_Mtok": interactive,
        "batch_arrivals_Mtok": batch,
        "batch_service_Mtok": batch.copy(),
        "backlog_Mtok": np.zeros(m.T_SLOTS),
        "useful_tokens_Mtok": useful,
        "energy_cost_usd": cost,
        "degradation_cost_usd": 0.0,
        "total_cost_usd": cost,
        "emissions_kg": emissions,
        "usd_per_Mtok": cost / useful,
        "gco2_per_Mtok": emissions * 1e3 / useful,
        "peak_grid_kw": float(grid.max()),
        "evening_grid_kwh": float(grid[17:22].sum() * m.DT_H),
    }


if __name__ == "__main__":
    base = baseline_metrics()
    opt = solve_revised_dispatch()
    for name, result in (("BAU", base), ("Revised optimum", opt)):
        print(
            f"{name}: cost={result['total_cost_usd']:.2f} $, "
            f"CO2={result['emissions_kg']:.2f} kg, "
            f"unit cost={result['usd_per_Mtok']:.4f} $/Mtok, "
            f"unit CO2={result['gco2_per_Mtok']:.1f} g/Mtok"
        )
    print("levels:", "".join(map(str, opt["levels"])))
    print("charge:", np.flatnonzero(opt["charge"]))
    print("discharge:", np.flatnonzero(opt["discharge"]))
    print("SOC:", np.round(opt["soc_kwh"], 1))
    print("backlog:", np.round(opt["backlog_Mtok"], 1))
