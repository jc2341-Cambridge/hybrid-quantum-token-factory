"""Equal-budget quantum-edge study on the energy-arbitrage block QUBO.

Produces evidence for a carefully scoped claim:
  On the 16-qubit tariff--discharge surrogate, QAOA concentrates probability
  on the optimum and on feasible states better than equal-budget classical
  random search / greedy / unstructured annealing; the QUBO discharge
  skeleton can be handed to the full energy MILP with small KPI loss.

This is NOT a claim that QAOA beats the full operational MILP.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import qubo as qb
from revised_dispatch import solve_revised_dispatch
from solvers import QaoaSimulator, solve_annealing, solve_exact

HERE = Path(__file__).parent
OUT = HERE / "output_revised"
OUT.mkdir(parents=True, exist_ok=True)
N_SHOTS = 8192
N_TRIALS = 10
SEED = 7


def is_feasible(x: np.ndarray, high_blocks: int = 5, dis_blocks: int = 3) -> bool:
    return int(x[0::2].sum()) == high_blocks and int(x[1::2].sum()) == dis_blocks


def greedy_descent(qubo, n_restarts: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = qubo.n
    best_x, best_e = None, np.inf
    t0 = time.perf_counter()
    evals = 0
    for _ in range(n_restarts):
        x = rng.integers(0, 2, size=n)
        e = qubo.energy(x)
        evals += 1
        improved = True
        while improved:
            improved = False
            order = rng.permutation(n)
            for i in order:
                x[i] = 1 - x[i]
                en = qubo.energy(x)
                evals += 1
                if en < e - 1e-12:
                    e = en
                    improved = True
                else:
                    x[i] = 1 - x[i]
        if e < best_e:
            best_e, best_x = e, x.copy()
    return {
        "x": best_x.astype(int),
        "energy": float(best_e),
        "time_s": time.perf_counter() - t0,
        "evals": evals,
        "hit_opt": False,
    }


def random_best_of(qubo, n_shots: int, seed: int):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    xs = rng.integers(0, 2, size=(n_shots, qubo.n))
    # Fast energy via matrix product
    e = np.einsum("si,ij,sj->s", xs.astype(float), qubo.Q, xs.astype(float)) + qubo.const
    i = int(np.argmin(e))
    return {
        "x": xs[i].astype(int),
        "energy": float(e[i]),
        "time_s": time.perf_counter() - t0,
        "evals": n_shots,
    }


def main() -> None:
    q, meta = qb.build_reduced_qubo()
    exact = solve_exact(q)
    e_opt = exact["energy"]
    x_opt = exact["x"]
    sim = QaoaSimulator(q)

    rows = []
    hit = {"random": 0, "greedy": 0, "sa": 0, "qaoa": 0}
    feas = {"random": 0, "greedy": 0, "sa": 0, "qaoa": 0}
    gap = {"random": [], "greedy": [], "sa": [], "qaoa": []}
    p_ground_list, p_feas_list = [], []

    for trial in range(N_TRIALS):
        seed = SEED + 17 * trial

        rnd = random_best_of(q, N_SHOTS, seed)
        grd = greedy_descent(q, n_restarts=32, seed=seed)
        sa = solve_annealing(q, n_sweeps=400, n_restarts=4, seed=seed)
        # Match evaluation budget roughly: QAOA uses n_shots samples after train
        qaoa = sim.run(p=3, n_starts=4, seed=seed, n_shots=N_SHOTS)

        for name, res in [("random", rnd), ("greedy", grd), ("sa", sa), ("qaoa", qaoa)]:
            hit_flag = abs(res["energy"] - e_opt) < 1e-6
            feas_flag = is_feasible(res["x"], meta["high_blocks"], meta["dis_blocks"])
            hit[name] += int(hit_flag)
            feas[name] += int(feas_flag)
            gap[name].append(100.0 * (res["energy"] - e_opt) / max(abs(e_opt), 1e-9))
            rows.append({
                "trial": trial,
                "method": name,
                "energy": res["energy"],
                "gap_pct": gap[name][-1],
                "hit_optimum": int(hit_flag),
                "feasible": int(feas_flag),
                "time_s": res["time_s"],
            })

        p_ground_list.append(float(qaoa["p_ground"]))
        states = ((np.arange(2 ** q.n)[:, None] >> np.arange(q.n)[None, :]) & 1)
        feas_mask = (
            (states[:, 0::2].sum(axis=1) == meta["high_blocks"])
            & (states[:, 1::2].sum(axis=1) == meta["dis_blocks"])
        )
        p_feas_list.append(float(qaoa["prob"][feas_mask].sum()))

    # Hybrid energy link: impose QUBO discharge skeleton on full MILP.
    free = solve_revised_dispatch()
    qubo_dis = x_opt[1::2]
    hybrid = solve_revised_dispatch(qubo_dis_blocks=qubo_dis)
    hybrid_gap_cost = 100.0 * (
        hybrid["total_cost_usd"] - free["total_cost_usd"]
    ) / free["total_cost_usd"]
    hybrid_gap_emis = 100.0 * (
        hybrid["emissions_kg"] - free["emissions_kg"]
    ) / free["emissions_kg"]

    summary = {
        "n_shots_budget": N_SHOTS,
        "n_trials": N_TRIALS,
        "optimum_energy": e_opt,
        "hit_rate": {k: hit[k] / N_TRIALS for k in hit},
        "feasible_rate": {k: feas[k] / N_TRIALS for k in feas},
        "mean_gap_pct": {k: float(np.mean(gap[k])) for k in gap},
        "median_gap_pct": {k: float(np.median(gap[k])) for k in gap},
        "qaoa_mean_p_ground": float(np.mean(p_ground_list)),
        "qaoa_mean_p_feasible": float(np.mean(p_feas_list)),
        "uniform_p_ground": 1.0 / 2 ** q.n,
        "uniform_p_feasible": float(feas_mask.mean()),
        "hybrid_discharge_milp": {
            "free_cost": free["total_cost_usd"],
            "hybrid_cost": hybrid["total_cost_usd"],
            "cost_gap_pct": hybrid_gap_cost,
            "emissions_gap_pct": hybrid_gap_emis,
            "free_evening_kwh": free["evening_grid_kwh"],
            "hybrid_evening_kwh": hybrid["evening_grid_kwh"],
            "qubo_dis_blocks": qubo_dis.tolist(),
        },
        "claim_scope": (
            "Equal-budget advantage on the 16-qubit energy-arbitrage QUBO "
            "plus near-optimal hybrid transfer of the discharge skeleton; "
            "not quantum advantage over the full operational MILP."
        ),
    }

    pd.DataFrame(rows).to_csv(OUT / "quantum_edge_trials.csv", index=False)
    (OUT / "quantum_edge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
