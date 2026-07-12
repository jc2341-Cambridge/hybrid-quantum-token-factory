"""QUBO construction for the joint compute-battery-grid schedule.

Full instance  : T slots x (K one-hot compute bits + charge bit + discharge bit)
Reduced instance: T blocks x (1 compute bit + 1 discharge bit)  -> NISQ scale

The underlying problem is a linear ILP; the QUBO embeds constraints as
quadratic penalties so quantum heuristics (QAOA, annealing) can address it.
"""
from __future__ import annotations

import numpy as np

import model as m


class Qubo:
    def __init__(self, n: int):
        self.n = n
        self.Q = np.zeros((n, n))
        self.const = 0.0

    def add_linear(self, i: int, w: float) -> None:
        self.Q[i, i] += w

    def add_quad(self, i: int, j: int, w: float) -> None:
        if i == j:
            self.Q[i, i] += w
        else:
            a, b = min(i, j), max(i, j)
            self.Q[a, b] += w

    def add_equality_penalty(self, idx, coeffs, target: float, weight: float) -> None:
        """weight * (sum_i coeffs_i x_i - target)^2"""
        idx = list(idx)
        coeffs = list(coeffs)
        for a, i in enumerate(idx):
            self.add_linear(i, weight * (coeffs[a] ** 2 - 2.0 * coeffs[a] * target))
            for b in range(a + 1, len(idx)):
                self.add_quad(i, idx[b], 2.0 * weight * coeffs[a] * coeffs[b])
        self.const += weight * target ** 2

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.const)

    def energies_all(self) -> np.ndarray:
        """Enumerate all 2^n states (n <= 22). Returns vector of energies."""
        n = self.n
        states = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        return np.einsum("si,ij,sj->s", states, self.Q, states) + self.const


# --------------------------------------------------------------------------
def build_full_qubo(token_target_mtok: float, penalty_struct: float = 800.0,
                    penalty_token: float = 0.05, penalty_batt: float = 500.0,
                    carbon_price: float | None = None,
                    sla_floor: bool = True, penalty_floor: float = 5000.0,
                    theta: float | None = None):
    # penalty_batt = 500: must exceed the largest single-slot battery gain,
    # P_dis * max(tariff) ~ 270 kW x 0.74 $/kWh ~ 200 $ at lambda = 600 $/t,
    # otherwise the annealer profitably violates the cycle budget at high
    # carbon prices.
    """Full 24-h instance: K=3 one-hot levels + charge/discharge bits per slot.

    penalty_struct : one-hot / simultaneous charge-discharge penalties ($ scale)
    penalty_token  : weight on (Mtok deviation)^2 -- one mid->low step (54 Mtok)
                     costs ~ 146 $, above the largest per-slot tariff saving
    carbon_price   : internal carbon price $/gCO2 (default model.CARBON_PRICE)
    sla_floor      : penalise compute levels below model.SLA_FLOOR_LEVELS
                     (QoS floor); penalty_floor = 5000 exceeds the largest
                     single-slot gain of a floor violation (~1.9 $/kWh x
                     977 kW at lambda = 2500 $/t)
    theta          : Dinkelbach unit-cost parameter (effective $ per Mtok).
                     When given, the daily token-equality penalty is REPLACED
                     by the linear reward -theta * R_k on every level bit, so
                     the QUBO optimum minimises cost - theta * tokens (the
                     Dinkelbach subproblem of blended-unit-cost minimisation;
                     surplus production above the QoS floor is free).
    """
    T, K = m.T_SLOTS, len(m.U_LEVELS)
    price, carbon, pv, t_amb = m.profiles(T)
    tariff = m.effective_tariff(price, carbon, carbon_price)

    n = T * (K + 2)
    x_idx = lambda t, k: t * (K + 2) + k
    c_idx = lambda t: t * (K + 2) + K
    d_idx = lambda t: t * (K + 2) + K + 1
    q = Qubo(n)

    p_levels = m.facility_power_kw(m.U_LEVELS, 25.0)  # nominal PUE for encoding
    r_levels = m.throughput_mtok_h(m.U_LEVELS) * m.DT_H

    for t in range(T):
        for k in range(K):
            q.add_linear(x_idx(t, k), tariff[t] * p_levels[k] * m.DT_H)
            if theta is not None:
                q.add_linear(x_idx(t, k), -theta * r_levels[k])
        q.add_linear(c_idx(t), tariff[t] * m.P_CH_KW * m.DT_H)
        q.add_linear(d_idx(t), -tariff[t] * m.P_DIS_KW * m.DT_H)
        # one-hot compute level
        q.add_equality_penalty([x_idx(t, k) for k in range(K)], [1.0] * K, 1.0,
                               penalty_struct)
        # never charge and discharge simultaneously
        q.add_quad(c_idx(t), d_idx(t), penalty_struct)
        # QoS floor: levels below the SLA floor are penalised out
        if sla_floor:
            for k in range(int(m.SLA_FLOOR_LEVELS[t])):
                q.add_linear(x_idx(t, k), penalty_floor)

    # token demand over the day (only in fixed-output mode; in Dinkelbach
    # mode the hourly QoS floor guarantees demand and surplus is rewarded)
    if theta is None:
        q.add_equality_penalty(
            [x_idx(t, k) for t in range(T) for k in range(K)],
            [r_levels[k] for _ in range(T) for k in range(K)],
            token_target_mtok,
            penalty_token,
        )
    # battery cycle budget: exactly N charge and N discharge slots
    q.add_equality_penalty([c_idx(t) for t in range(T)], [1.0] * T,
                           m.N_CYCLE_SLOTS, penalty_batt)
    q.add_equality_penalty([d_idx(t) for t in range(T)], [1.0] * T,
                           m.N_CYCLE_SLOTS, penalty_batt)

    meta = {"T": T, "K": K, "x_idx": x_idx, "c_idx": c_idx, "d_idx": d_idx}
    return q, meta


def decode_full(x: np.ndarray, meta):
    T, K = meta["T"], meta["K"]
    levels = np.zeros(T, dtype=int)
    b_ch = np.zeros(T, dtype=int)
    b_dis = np.zeros(T, dtype=int)
    for t in range(T):
        bits = [x[meta["x_idx"](t, k)] for k in range(K)]
        levels[t] = int(np.argmax(bits)) if sum(bits) > 0 else 0
        b_ch[t] = int(x[meta["c_idx"](t)])
        b_dis[t] = int(x[meta["d_idx"](t)])
    return levels, b_ch, b_dis


# --------------------------------------------------------------------------
def build_reduced_qubo(n_blocks: int = 8, high_blocks: int = 5,
                       dis_blocks: int = 3, penalty: float = 2000.0,
                       carbon_price: float | None = None):
    # penalty = 2000: the marginal cost of the first unit of constraint
    # deviation (2000 $) exceeds the largest single-block tariff saving
    # (~1340 $ for one high->low switch at the evening-peak tariff), so the
    # QUBO optimum provably coincides with the constrained optimum.
    """NISQ-scale block instance: per 3-h block, 1 compute bit (high/low) and
    1 battery-discharge bit. n = 2 * n_blocks qubits (16 for 8 blocks)."""
    T = m.T_SLOTS
    hours_per_block = T // n_blocks
    price, carbon, pv, t_amb = m.profiles(T)
    tariff_b = m.effective_tariff(price, carbon, carbon_price).reshape(
        n_blocks, hours_per_block).mean(axis=1)

    p_low = float(m.facility_power_kw(m.U_LEVELS[0], 25.0))
    p_high = float(m.facility_power_kw(m.U_LEVELS[-1], 25.0))
    dt_block = hours_per_block * m.DT_H

    n = 2 * n_blocks
    x_idx = lambda b: 2 * b
    d_idx = lambda b: 2 * b + 1
    q = Qubo(n)
    for b in range(n_blocks):
        q.add_linear(x_idx(b), tariff_b[b] * (p_high - p_low) * dt_block)
        q.add_linear(d_idx(b), -tariff_b[b] * m.P_DIS_KW * dt_block)
        q.const += tariff_b[b] * p_low * dt_block
    q.add_equality_penalty([x_idx(b) for b in range(n_blocks)],
                           [1.0] * n_blocks, high_blocks, penalty)
    q.add_equality_penalty([d_idx(b) for b in range(n_blocks)],
                           [1.0] * n_blocks, dis_blocks, penalty)
    meta = {"n_blocks": n_blocks, "x_idx": x_idx, "d_idx": d_idx,
            "high_blocks": high_blocks, "dis_blocks": dis_blocks}
    return q, meta


# --------------------------------------------------------------------------
def build_storage_master_qubo(
    n_blocks: int = 24,
    charge_blocks: int = 4,
    dis_blocks: int = 4,
    penalty: float = 5000.0,
    balance_weight: float = 0.15,
    order_weight: float = 25.0,
    carbon_price: float | None = None,
    soc_path_weight: float = 0.012,
    adj_weight: float = 120.0,
    exclusivity_weight: float = 800.0,
    tariff_pair_weight: float = 0.35,
    pair_jitter: float = 2200.0,
    jitter_seed: int = 48,
):
    """Hybrid storage master: ``n_blocks`` charge + ``n_blocks`` discharge bits.

    Default manuscript encoding: **48 qubits** (24 hourly charge + 24 hourly
    discharge), budgets \(\sum c=4\), \(\sum d=4\).

    Linear tariff/order terms sit on the diagonal. Off-diagonal structure is
    time-resolved:

    * cardinality equality penalties (hour budgets),
    * nested soft SOC-path penalties on prefix charge/discharge sums,
    * terminal round-trip balance,
    * same-hour exclusivity ``c_t d_t``,
    * adjacent-hour clustering and tariff-product pair terms,
    * seeded Gaussian pair jitter (``jitter_seed``) for reproducibility.

    Hourly exclusivity and SOC bounds remain hard in the MILP subproblem;
    the master only ranks masks.
    """
    from revised_dispatch import (
        ETA_CH,
        ETA_DIS,
        SOC_INITIAL_KWH,
        SOC_MAX_KWH,
        SOC_MIN_KWH,
    )

    T = m.T_SLOTS
    hours_per_block = T // n_blocks
    price, carbon, _, _ = m.profiles(T)
    tariff_b = m.effective_tariff(price, carbon, carbon_price).reshape(
        n_blocks, hours_per_block
    ).mean(axis=1)
    dt = m.DT_H
    e_ch = ETA_CH * m.P_CH_KW * dt
    e_dis = (m.P_DIS_KW / ETA_DIS) * dt
    s_star = 0.5 * (SOC_MIN_KWH + SOC_MAX_KWH)

    n = 2 * n_blocks
    c_idx = lambda b: b
    d_idx = lambda b: n_blocks + b
    q = Qubo(n)

    for b in range(n_blocks):
        q.add_linear(c_idx(b), tariff_b[b] * m.P_CH_KW * dt)
        q.add_linear(d_idx(b), -tariff_b[b] * m.P_DIS_KW * dt)
        q.add_linear(c_idx(b), order_weight * (b / max(n_blocks - 1, 1)))
        q.add_linear(
            d_idx(b), order_weight * (1.0 - b / max(n_blocks - 1, 1))
        )

    q.add_equality_penalty(
        [c_idx(b) for b in range(n_blocks)],
        [1.0] * n_blocks,
        charge_blocks,
        penalty,
    )
    q.add_equality_penalty(
        [d_idx(b) for b in range(n_blocks)],
        [1.0] * n_blocks,
        dis_blocks,
        penalty,
    )

    if soc_path_weight > 0.0:
        for t in range(n_blocks):
            idx = [c_idx(i) for i in range(t + 1)] + [
                d_idx(j) for j in range(t + 1)
            ]
            coeffs = [e_ch] * (t + 1) + [-e_dis] * (t + 1)
            target = s_star - SOC_INITIAL_KWH
            q.add_equality_penalty(idx, coeffs, target, soc_path_weight)

    for i in range(n_blocks):
        q.add_linear(c_idx(i), balance_weight * (e_ch ** 2))
        q.add_linear(d_idx(i), balance_weight * (e_dis ** 2))
        for j in range(i + 1, n_blocks):
            q.add_quad(c_idx(i), c_idx(j), 2.0 * balance_weight * e_ch * e_ch)
            q.add_quad(d_idx(i), d_idx(j), 2.0 * balance_weight * e_dis * e_dis)
        for j in range(n_blocks):
            q.add_quad(
                c_idx(i), d_idx(j), -2.0 * balance_weight * e_ch * e_dis
            )

    if exclusivity_weight > 0.0:
        for t in range(n_blocks):
            q.add_quad(c_idx(t), d_idx(t), exclusivity_weight)

    if adj_weight > 0.0:
        t_mean = float(np.mean(tariff_b)) + 1e-9
        for i in range(n_blocks - 1):
            w_c = -adj_weight * (
                t_mean / (0.5 * (tariff_b[i] + tariff_b[i + 1]) + 1e-9)
            )
            q.add_quad(c_idx(i), c_idx(i + 1), w_c)
            w_d = -adj_weight * (
                0.5 * (tariff_b[i] + tariff_b[i + 1]) / t_mean
            )
            q.add_quad(d_idx(i), d_idx(i + 1), w_d)

    if tariff_pair_weight > 0.0:
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                q.add_quad(
                    c_idx(i),
                    c_idx(j),
                    tariff_pair_weight * tariff_b[i] * tariff_b[j],
                )
                q.add_quad(
                    d_idx(i),
                    d_idx(j),
                    -tariff_pair_weight * tariff_b[i] * tariff_b[j],
                )

    if pair_jitter > 0.0:
        rng = np.random.default_rng(jitter_seed)
        for i in range(n_blocks):
            for j in range(i + 1, n_blocks):
                q.add_quad(
                    c_idx(i), c_idx(j), float(rng.normal(0.0, pair_jitter))
                )
                q.add_quad(
                    d_idx(i), d_idx(j), float(rng.normal(0.0, pair_jitter))
                )
            for j in range(n_blocks):
                q.add_quad(
                    c_idx(i), d_idx(j), float(rng.normal(0.0, pair_jitter))
                )

    meta = {
        "n_blocks": n_blocks,
        "c_idx": c_idx,
        "d_idx": d_idx,
        "charge_blocks": charge_blocks,
        "dis_blocks": dis_blocks,
        "tariff_b": tariff_b,
        "kind": "storage_master",
        "soc_path_weight": soc_path_weight,
        "adj_weight": adj_weight,
        "exclusivity_weight": exclusivity_weight,
        "tariff_pair_weight": tariff_pair_weight,
        "pair_jitter": pair_jitter,
        "jitter_seed": jitter_seed,
    }
    return q, meta


def decode_storage_master(x: np.ndarray, meta) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = int(meta["n_blocks"])
    x = np.asarray(x, dtype=int)
    charge = np.array([int(x[meta["c_idx"](b)]) for b in range(n_blocks)])
    discharge = np.array([int(x[meta["d_idx"](b)]) for b in range(n_blocks)])
    return charge, discharge


def add_nogood_cut(q: Qubo, x_star: np.ndarray, weight: float = 5000.0) -> None:
    """Forbid revisiting bitstring x_star in subsequent master solves.

    Adds ``weight * (number of agreeing bits)^2``, which is maximised at
    ``x = x_star`` and therefore pushes the master away from that point.
    With ``weight`` larger than typical objective gaps, the cut makes
    ``x_star`` strictly suboptimal.
    """
    x_star = np.asarray(x_star, dtype=int)
    n = q.n
    # agree_i = 1 - x_i - x*_i + 2 x_i x*_i
    # (sum agree)^2 = sum_i agree_i + 2 sum_{i<j} agree_i agree_j
    # Expand in x and push into Q / const.
    # agree_i = a_i + b_i x_i with a_i = 1 - x*_i, b_i = 2 x*_i - 1.
    a = 1 - x_star
    b = 2 * x_star - 1
    # const from sum_i a_i and 2 sum_{i<j} a_i a_j
    q.const += weight * float(a.sum() ** 2)
    for i in range(n):
        # linear from 2 a_i b_i x_i inside (a_i + b_i x_i) contributions
        # d/dx_i of (sum agree)^2 = 2 (sum agree) * b_i, at x=0 is 2 a.sum()*b_i
        # Full expansion:
        # sum_i (a_i + b_i x_i)^2 wait no - we need (sum_i agree_i)^2 not sum squares.
        # (sum agree)^2 = sum_i agree_i^2 + 2 sum_{i<j} agree_i agree_j
        # agree_i^2 = agree_i for 0/1, so sum agree + 2 sum_{i<j} ...
        # = (sum agree)^2 already. Expand:
        # sum_i sum_j (a_i + b_i x_i)(a_j + b_j x_j)
        # = sum_{i,j} a_i a_j + sum_{i,j} a_i b_j x_j + sum_{i,j} b_i a_j x_i
        #   + sum_{i,j} b_i b_j x_i x_j
        # const = (sum a)^2 already added.
        # linear coef of x_k: 2 b_k (sum a) + b_k^2  (from i=j=k term in last sum
        #   handled below) -- do pairwise carefully.
        pass
    # Linear: for each k, coef = 2 * b_k * sum_i a_i
    # from 2 sum_i a_i * (b_k x_k) and from diagonal b_k^2 x_k (since x_k^2=x_k)
    suma = float(a.sum())
    for k in range(n):
        q.add_linear(k, weight * (2.0 * b[k] * suma + float(b[k] ** 2)))
    # Quadratic i<j: 2 b_i b_j
    for i in range(n):
        for j in range(i + 1, n):
            q.add_quad(i, j, weight * 2.0 * float(b[i] * b[j]))


def add_optimality_cut_qubo(
    q: Qubo,
    x_star: np.ndarray,
    true_cost: float,
    ub: float,
    scale: float = 1.0,
) -> None:
    """Path-C strengthening: heavier no-good when ``x_star`` misses the UB."""
    gap = max(0.0, float(true_cost) - float(ub))
    weight = 5000.0 + scale * gap
    add_nogood_cut(q, x_star, weight=weight)


def greedy_storage_masks(
    n_blocks: int = 24,
    charge_blocks: int = 4,
    dis_blocks: int = 4,
    carbon_price: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reviewer baseline: charge cheapest hours, discharge most expensive."""
    T = m.T_SLOTS
    hpb = T // n_blocks
    price, carbon, _, _ = m.profiles(T)
    tariff_b = m.effective_tariff(price, carbon, carbon_price).reshape(
        n_blocks, hpb
    ).mean(axis=1)
    d_order = np.argsort(-tariff_b)
    c_order = np.argsort(tariff_b)
    d = np.zeros(n_blocks, dtype=int)
    c = np.zeros(n_blocks, dtype=int)
    for b in d_order[:dis_blocks]:
        d[b] = 1
    taken = set(np.flatnonzero(d).tolist())
    filled = 0
    for b in c_order:
        if b in taken:
            continue
        c[b] = 1
        filled += 1
        if filled >= charge_blocks:
            break
    return c, d

