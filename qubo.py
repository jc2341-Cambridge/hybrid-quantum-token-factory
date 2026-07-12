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
                       dis_blocks: int = 3, penalty: float = 2000.0):
    # penalty = 2000: the marginal cost of the first unit of constraint
    # deviation (2000 $) exceeds the largest single-block tariff saving
    # (~1340 $ for one high->low switch at the evening-peak tariff), so the
    # QUBO optimum provably coincides with the constrained optimum.
    """NISQ-scale block instance: per 3-h block, 1 compute bit (high/low) and
    1 battery-discharge bit. n = 2 * n_blocks qubits (16 for 8 blocks)."""
    T = m.T_SLOTS
    hours_per_block = T // n_blocks
    price, carbon, pv, t_amb = m.profiles(T)
    tariff_b = m.effective_tariff(price, carbon).reshape(n_blocks, hours_per_block).mean(axis=1)

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
