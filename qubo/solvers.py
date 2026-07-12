"""Solvers: exact enumeration, simulated (quantum-inspired) annealing, and a
pure-numpy statevector QAOA for NISQ-scale sub-problems (n <= ~20 qubits)."""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize


# ------------------------------ exact ---------------------------------------
def solve_exact(qubo):
    t0 = time.perf_counter()
    e = qubo.energies_all()
    best = int(np.argmin(e))
    x = (best >> np.arange(qubo.n)) & 1
    return {"x": x.astype(int), "energy": float(e[best]),
            "time_s": time.perf_counter() - t0, "spectrum": e}


# ------------------------------ annealing -----------------------------------
def solve_annealing(qubo, n_sweeps: int = 2000, n_restarts: int = 6, seed: int = 0,
                    x0: np.ndarray | None = None):
    """Metropolis annealing with full sweeps and local-field bookkeeping.

    One restart may be warm-started from x0 (e.g. a feasible baseline
    schedule); the rest start from random strings.
    """
    rng = np.random.default_rng(seed)
    Q, n = qubo.Q, qubo.n
    Qs = Q + Q.T - np.diag(np.diag(Q))  # symmetrised couplings
    diag = np.diag(Q).copy()
    t0 = time.perf_counter()
    best_x, best_e = None, np.inf
    trace = []
    betas = np.geomspace(1e-3, 5.0, n_sweeps)
    for r in range(n_restarts):
        if x0 is not None and r == 0:
            x = np.asarray(x0, dtype=float).copy()
        else:
            x = rng.integers(0, 2, n).astype(float)
        e = qubo.energy(x)
        h = Qs @ x  # local field: h_i = sum_j (Q_ij + Q_ji symmetrised) x_j
        for beta in betas:
            for i in rng.permutation(n):
                delta = (1.0 - 2.0 * x[i]) * (h[i] - diag[i] * x[i] + diag[i])
                if delta <= 0 or rng.random() < np.exp(-min(beta * delta, 60.0)):
                    sgn = 1.0 - 2.0 * x[i]
                    x[i] = 1.0 - x[i]
                    h += Qs[:, i] * sgn
                    e += delta
                    if e < best_e:
                        best_e, best_x = e, x.copy()
        trace.append(best_e)
    best_e = qubo.energy(best_x)  # exact recompute (guards against drift)
    return {"x": best_x.astype(int), "energy": float(best_e),
            "time_s": time.perf_counter() - t0, "restart_trace": trace}


# --------------------------- structured annealing ---------------------------
def solve_structured_annealing(qubo, meta, n_sweeps: int = 4000,
                               n_restarts: int = 6, seed: int = 0,
                               x0: np.ndarray | None = None,
                               record_trace: bool = False):
    """Metropolis annealing with schedule-structured moves.

    Moves preserve the one-hot compute encoding and the battery cycle
    budgets by construction:
    (a) switch the compute level of a random slot,
    (b) relocate a battery charge slot (flip one set and one unset bit),
    (c) relocate a battery discharge slot,
    (d) occasionally toggle a single battery bit (repairs infeasible
        starts; the budget penalty makes it expensive otherwise).
    Same QUBO energy as the bit-flip annealer; only the proposal
    distribution is structured (as in domain-wall / integer-encoded
    annealing practice).
    """
    rng = np.random.default_rng(seed)
    Q, n = qubo.Q, qubo.n
    Qs = Q + Q.T - np.diag(np.diag(Q))
    diag = np.diag(Q).copy()
    T, K = meta["T"], meta["K"]
    x_idx, c_idx, d_idx = meta["x_idx"], meta["c_idx"], meta["d_idx"]
    t0 = time.perf_counter()

    def flip_delta(x, h, i):
        return (1.0 - 2.0 * x[i]) * (h[i] - diag[i] * x[i] + diag[i])

    def apply_flip(x, h, i):
        sgn = 1.0 - 2.0 * x[i]
        x[i] = 1.0 - x[i]
        h += Qs[:, i] * sgn

    best_x, best_e = None, np.inf
    betas = np.geomspace(2e-3, 8.0, n_sweeps)
    moves_per_sweep = 3 * T
    traces = []  # per restart: best-so-far energy after each sweep
    for r in range(n_restarts):
        if x0 is not None and r == 0:
            x = np.asarray(x0, dtype=float).copy()
        else:
            x = np.zeros(n)
            for t in range(T):
                x[x_idx(t, rng.integers(K))] = 1.0
        e = qubo.energy(x)
        h = Qs @ x
        best_e_r = e
        if e < best_e:
            best_e, best_x = e, x.copy()
        trace_r = np.empty(n_sweeps) if record_trace else None
        for i_sweep, beta in enumerate(betas):
            for _ in range(moves_per_sweep):
                kind = rng.integers(8)
                if kind < 4:  # switch compute level of one slot
                    t = rng.integers(T)
                    k_old = int(np.argmax([x[x_idx(t, k)] for k in range(K)]))
                    k_new = int((k_old + 1 + rng.integers(K - 1)) % K)
                    flips = [x_idx(t, k_old), x_idx(t, k_new)]
                elif kind in (4, 5):  # relocate a charge / discharge slot
                    idx_fun = c_idx if kind == 4 else d_idx
                    bits = np.array([x[idx_fun(t)] for t in range(T)])
                    on = np.flatnonzero(bits > 0.5)
                    off = np.flatnonzero(bits < 0.5)
                    if len(on) == 0 or len(off) == 0:
                        continue
                    flips = [idx_fun(int(rng.choice(on))),
                             idx_fun(int(rng.choice(off)))]
                elif kind == 6:
                    flips = [c_idx(rng.integers(T))]
                else:
                    flips = [d_idx(rng.integers(T))]
                delta = 0.0
                for i in flips:
                    delta += flip_delta(x, h, i)
                    apply_flip(x, h, i)
                if delta <= 0 or rng.random() < np.exp(-min(beta * delta, 60.0)):
                    e += delta
                    if e < best_e_r:
                        best_e_r = e
                    if e < best_e:
                        best_e, best_x = e, x.copy()
                else:
                    for i in reversed(flips):
                        apply_flip(x, h, i)  # revert
            if record_trace:
                trace_r[i_sweep] = best_e_r
        if record_trace:
            traces.append(trace_r)
    best_e = qubo.energy(best_x)
    out = {"x": best_x.astype(int), "energy": float(best_e),
           "time_s": time.perf_counter() - t0}
    if record_trace:
        out["traces"] = np.array(traces)
    return out


# ------------------------------ greedy --------------------------------------
def solve_greedy(qubo, seed: int = 0, x0: np.ndarray | None = None):
    """Deterministic 1-flip local descent (from x0 if given, else zeros)."""
    t0 = time.perf_counter()
    x = np.zeros(qubo.n) if x0 is None else np.asarray(x0, dtype=float).copy()
    e = qubo.energy(x)
    improved = True
    Q = qubo.Q
    Qs = Q + Q.T - np.diag(np.diag(Q))
    while improved:
        improved = False
        deltas = (1.0 - 2.0 * x) * (Qs @ x) + np.diag(Q) * (1.0 - 2.0 * x)
        i = int(np.argmin(deltas))
        if deltas[i] < -1e-12:
            x[i] = 1.0 - x[i]
            e += deltas[i]
            improved = True
    return {"x": x.astype(int), "energy": float(qubo.energy(x)),
            "time_s": time.perf_counter() - t0}


# ------------------------------ MILP ----------------------------------------
def solve_milp_full(token_target_mtok: float, carbon_price: float | None = None,
                    sla_floor: bool = True, theta: float | None = None):
    """Exact mixed-integer reference for the full daily schedule (HiGHS).

    Solves the original linear ILP (objective and constraints of the QUBO
    before penalty embedding), providing the true optimum for auditing the
    quantum-inspired solvers. With theta given, the token-equality row is
    dropped and the objective becomes cost - theta * tokens (Dinkelbach
    subproblem; matches build_full_qubo(theta=...)).
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    import model as m

    T, K = m.T_SLOTS, len(m.U_LEVELS)
    price, carbon, pv, t_amb = m.profiles(T)
    tariff = m.effective_tariff(price, carbon, carbon_price)
    p_levels = m.facility_power_kw(m.U_LEVELS, 25.0)
    r_levels = m.throughput_mtok_h(m.U_LEVELS) * m.DT_H

    n = T * (K + 2)
    x_idx = lambda t, k: t * (K + 2) + k
    c_idx = lambda t: t * (K + 2) + K
    d_idx = lambda t: t * (K + 2) + K + 1

    cost = np.zeros(n)
    for t in range(T):
        for k in range(K):
            cost[x_idx(t, k)] = tariff[t] * p_levels[k] * m.DT_H
            if theta is not None:
                cost[x_idx(t, k)] -= theta * r_levels[k]
        cost[c_idx(t)] = tariff[t] * m.P_CH_KW * m.DT_H
        cost[d_idx(t)] = -tariff[t] * m.P_DIS_KW * m.DT_H

    rows_A, rows_lb, rows_ub = [], [], []
    for t in range(T):  # one-hot per slot
        a = np.zeros(n)
        for k in range(K):
            a[x_idx(t, k)] = 1.0
        rows_A.append(a); rows_lb.append(1.0); rows_ub.append(1.0)
        a2 = np.zeros(n)  # no simultaneous charge/discharge
        a2[c_idx(t)] = 1.0
        a2[d_idx(t)] = 1.0
        rows_A.append(a2); rows_lb.append(0.0); rows_ub.append(1.0)
    if theta is None:  # daily token demand (fixed-output mode)
        a = np.zeros(n)
        for t in range(T):
            for k in range(K):
                a[x_idx(t, k)] = r_levels[k]
        rows_A.append(a); rows_lb.append(token_target_mtok)
        rows_ub.append(token_target_mtok)
    a = np.zeros(n)  # charge budget
    for t in range(T):
        a[c_idx(t)] = 1.0
    rows_A.append(a); rows_lb.append(m.N_CYCLE_SLOTS); rows_ub.append(m.N_CYCLE_SLOTS)
    a = np.zeros(n)  # discharge budget
    for t in range(T):
        a[d_idx(t)] = 1.0
    rows_A.append(a); rows_lb.append(m.N_CYCLE_SLOTS); rows_ub.append(m.N_CYCLE_SLOTS)

    upper = np.ones(n)
    if sla_floor:  # QoS floor: forbid levels below model.SLA_FLOOR_LEVELS
        for t in range(T):
            for k in range(int(m.SLA_FLOOR_LEVELS[t])):
                upper[x_idx(t, k)] = 0.0

    t0 = time.perf_counter()
    res = milp(c=cost,
               constraints=LinearConstraint(np.array(rows_A), rows_lb, rows_ub),
               integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), upper))
    x = np.round(res.x).astype(int)
    return {"x": x, "objective": float(res.fun),
            "time_s": time.perf_counter() - t0, "status": res.message}


# ------------------------------ QAOA ----------------------------------------
class QaoaSimulator:
    """Statevector QAOA for a diagonal cost Hamiltonian given by a QUBO."""

    def __init__(self, qubo):
        self.n = qubo.n
        self.cost = qubo.energies_all()          # diagonal H_C over 2^n states
        self.cost_shift = self.cost - self.cost.min()
        self.scale = max(self.cost_shift.max(), 1e-12)
        self.cost_norm = self.cost_shift / self.scale

    def _apply_mixer(self, psi: np.ndarray, beta: float) -> np.ndarray:
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(self.n):
            psi = psi.reshape(2 ** q, 2, -1)
            a = psi[:, 0, :].copy()
            b = psi[:, 1, :].copy()
            psi[:, 0, :] = c * a + s * b
            psi[:, 1, :] = s * a + c * b
            psi = psi.reshape(-1)
        return psi

    def state(self, gammas, betas) -> np.ndarray:
        psi = np.full(2 ** self.n, 1.0 / np.sqrt(2 ** self.n), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.cost_norm)
            psi = self._apply_mixer(psi, b)
        return psi

    def expectation(self, params) -> float:
        p = len(params) // 2
        psi = self.state(params[:p], params[p:])
        return float(np.abs(psi) ** 2 @ self.cost)

    def run(self, p: int = 3, n_starts: int = 6, seed: int = 0, n_shots: int = 16384,
            record_trace: bool = False):
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        best, best_trace = None, None
        for _ in range(n_starts):
            x0 = np.concatenate([rng.uniform(0, 2.0, p), rng.uniform(0, np.pi, p)])
            evals = [] if record_trace else None

            def f(params):
                v = self.expectation(params)
                if record_trace:
                    evals.append(v)
                return v

            res = minimize(f, x0, method="COBYLA",
                           options={"maxiter": 250, "rhobeg": 0.4})
            if best is None or res.fun < best.fun:
                best = res
                if record_trace:
                    best_trace = np.minimum.accumulate(np.array(evals))
        psi = self.state(best.x[:p], best.x[p:])
        prob = np.abs(psi) ** 2
        # sampled best bitstring
        samples = rng.choice(len(prob), size=n_shots, p=prob / prob.sum())
        s_best = samples[np.argmin(self.cost[samples])]
        x = ((s_best >> np.arange(self.n)) & 1).astype(int)
        e_min, e_max = self.cost.min(), self.cost.max()
        out = {
            "x": x,
            "energy": float(self.cost[s_best]),
            "expectation": float(prob @ self.cost),
            "approx_ratio": float(1.0 - (prob @ self.cost - e_min) / (e_max - e_min)),
            "p_ground": float(prob[np.argmin(self.cost)]),
            "prob": prob,
            "time_s": time.perf_counter() - t0,
            "params": best.x,
        }
        if record_trace:
            out["trace"] = best_trace
        return out
