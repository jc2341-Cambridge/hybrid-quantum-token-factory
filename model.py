"""Token-factory energy model and 24-h context profiles.

Market price and carbon default to published East China sources under
``data/real_market/``. Units: power kW, energy kWh, price $/kWh,
carbon gCO2/kWh, throughput Mtoken/h.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ----------------------------- cluster parameters ---------------------------
P_IDLE_KW = 250.0          # IT idle power (modern clusters with sleep states)
P_PEAK_KW = 1200.0         # IT power at u = 1
ALPHA = 1.4                # super-linear power-utilisation exponent
R_MAX_MTOK_H = 180.0       # token throughput at u = 1 (Mtoken/h)

U_LEVELS = np.array([0.40, 0.70, 1.00])   # discrete compute levels (one-hot)
LEVEL_NAMES = ["low", "mid", "high"]

# Battery: fixed-power slot actions, round-trip efficiency inside P_DIS
P_CH_KW = 300.0
P_DIS_KW = 270.0
N_CYCLE_SLOTS = 4          # exactly N charge and N discharge slots per day

CARBON_PRICE = 1.5e-4      # $ per gCO2 (internal carbon price, 150 $/tCO2)

T_SLOTS = 24
DT_H = 1.0

# Business-as-usual reference: a demand-tracking schedule that follows the
# canonical diurnal inference-request curve (night valley 23-06 h, shoulder
# 07-11 h and 20-22 h, midday/evening plateau 12-19 h). With 8 slots at each
# level it delivers exactly the same 3024 Mtok/day as the optimised
# schedules (8*(72+126+180) = 24*126 Mtok), so all comparisons stay
# iso-output. No battery cycling (price-agnostic operation).
BAU_LEVELS = np.array([0] * 7 + [1] * 5 + [2] * 8 + [1] * 3 + [0], dtype=int)

# Quality-of-service floor for the optimised schedules, derived from the
# traffic mix rather than from any hand-picked time window: a beta = 30 %
# share of inference traffic is delay-tolerant batch work (carbon-/price-
# aware schedulers at Google and Microsoft report 20-40 % temporally
# flexible load), while the remaining 70 % is latency-critical interactive
# traffic that must be served in place in EVERY hour. In level terms the
# floor therefore permits a one-level dip only where demand is high
# (126/180 = 70 % of the instantaneous request rate still served); in mid-
# and low-demand hours a dip would cut below the interactive share
# (72/126 = 57 %), so the full demand-tracking level is required there.
# Legacy SLA-floor helpers below are retained for pre-revision scripts.
# The current paper uses revised_dispatch.py: fixed useful workload,
# interactive same-slot service, and deadline-constrained batch backlog.
BATCH_SHARE = 0.30
SLA_FLOOR_LEVELS = np.where(BAU_LEVELS == 2, 1, BAU_LEVELS).astype(int)


def it_power_kw(u: np.ndarray | float) -> np.ndarray | float:
    return P_IDLE_KW + (P_PEAK_KW - P_IDLE_KW) * np.asarray(u) ** ALPHA


def throughput_mtok_h(u: np.ndarray | float) -> np.ndarray | float:
    return R_MAX_MTOK_H * np.asarray(u)


# ----------------------------- TTFT / SLA token economics -------------------
# Minimal Path-B coupling: fixed class latency SLAs + utilisation-dependent
# TTFT (SLIT-style processing contention). Effective useful yield falls when
# TTFT exceeds the class SLA (Patole-style hard latency limits). Prefill /
# decode mix is not fixed by a constant rho.
U_STAR = float(U_LEVELS[1])
TTFT_LOAD = 0.25
TTFT_PROC0 = 0.75
TTFT_KAPPA = 2.2
ELL_I = 1.15   # interactive latency SLA (normalised TTFT units)
ELL_B = 1.85   # batch latency SLA (looser)


def ttft_norm(u: np.ndarray | float) -> np.ndarray | float:
    """Normalised TTFT proxy; equals 1 at u <= u*."""
    uu = np.asarray(u, dtype=float)
    return TTFT_LOAD + TTFT_PROC0 * (
        1.0 + TTFT_KAPPA * np.maximum(0.0, (uu - U_STAR) / (1.0 - U_STAR)) ** 2
    )


def phi_I(u: np.ndarray | float) -> np.ndarray | float:
    return np.minimum(1.0, ELL_I / ttft_norm(u))


def phi_B(u: np.ndarray | float) -> np.ndarray | float:
    return np.minimum(1.0, ELL_B / ttft_norm(u))


def phi_mix(u: np.ndarray | float, batch_share: float = BATCH_SHARE) -> np.ndarray | float:
    """Traffic-mix effective useful-token yield under fixed class SLAs."""
    return (1.0 - batch_share) * phi_I(u) + batch_share * phi_B(u)


def effective_throughput_mtok_h(u: np.ndarray | float) -> np.ndarray | float:
    """SLO-feasible useful throughput (Mtoken/h)."""
    return throughput_mtok_h(u) * phi_mix(u)


def pue(t_amb_c: np.ndarray | float) -> np.ndarray | float:
    return 1.15 + 0.02 * np.maximum(0.0, np.asarray(t_amb_c) - 15.0)


def facility_power_kw(u, t_amb_c):
    return pue(t_amb_c) * it_power_kw(u)


def joule_per_token(u, t_amb_c=25.0):
    """Facility-level energy intensity, J/token."""
    p_w = facility_power_kw(u, t_amb_c) * 1e3
    r_tok_s = throughput_mtok_h(u) * 1e6 / 3600.0
    return p_w / r_tok_s


# ----------------------------- 24-h context profiles ------------------------
# Illustrative East China regional mean-day irradiance (W/m2) from PVGIS
# v5.2 / PVGIS-SARAH2 at a reference location near 31.23 N, 121.47 E.
# https://re.jrc.ec.europa.eu/api/v5_2/DRcalc?lat=31.23&lon=121.47&month=M
# Values are shifted from UTC to local time UTC+8 (no DST). July is the
# paper default; other months feed the seasonal sensitivity study. April
# noon can exceed July because of the East-Asia plum-rain (meiyu) season.
PVGIS_G_W_M2 = {
    1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 47.67, 164.36, 284.82, 378.43, 426.67,      # 06-11 h
                 418.45, 367.75, 277.36, 167.66, 43.94, 0.0,      # 12-17 h
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.38,
                 99.27, 247.17, 404.76, 547.70, 651.85, 700.54,   # 06-11 h
                 696.02, 636.40, 523.03, 380.22, 224.24, 80.06,   # 12-17 h
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.80,
                 133.95, 277.00, 421.91, 545.89, 636.40, 681.81,  # 06-11 h
                 672.09, 620.17, 531.42, 405.01, 263.83, 128.54,  # 12-17 h
                 26.10, 0.0, 0.0, 0.0, 0.0, 0.0]),
    10: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  38.18, 160.83, 305.21, 428.79, 506.46, 534.88,  # 06-11 h
                  515.48, 443.47, 340.28, 209.12, 75.67, 1.43,    # 12-17 h
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}
PVGIS_G_JULY_W_M2 = PVGIS_G_W_M2[7]
PV_STC_KW = 500.0          # array nameplate at standard test conditions
PV_PERFORMANCE_RATIO = 0.85


def pv_profile_kw(month: int = 7) -> np.ndarray:
    """AC PV output (kW) for the PVGIS mean day of the given month."""
    return PV_STC_KW * (PVGIS_G_W_M2[month] / 1000.0) * PV_PERFORMANCE_RATIO


_REAL_PROFILE_CSV = (
    Path(__file__).resolve().parent.parent / "data" / "real_market"
    / "east_china_july_profiles.csv"
)


def profiles(t_slots: int = T_SLOTS):
    """Hourly price, carbon intensity, PV and ambient temperature.

    Default market signals use published East China sources in
    ``data/real_market/east_china_july_profiles.csv``:
    Jiangsu industrial TOU to-door prices, MEE 2022 East China average
    electricity CO2 factor, and Open-Meteo Shanghai July mean temperature.
    PV remains the PVGIS mean-day series.
    """
    h = np.arange(t_slots)
    pv = pv_profile_kw(7)[:t_slots]
    if _REAL_PROFILE_CSV.exists():
        import pandas as pd
        df = pd.read_csv(_REAL_PROFILE_CSV)
        price = df["price_usd_kwh"].to_numpy(dtype=float)[:t_slots]
        carbon = df["carbon_g_kwh"].to_numpy(dtype=float)[:t_slots]
        t_amb = df["temperature_c"].to_numpy(dtype=float)[:t_slots]
        return price, carbon, pv, t_amb
    # Fallback synthetic profiles if the real-market CSV is absent.
    price = (0.08
             + 0.06 * np.exp(-0.5 * ((h - 9.0) / 2.0) ** 2)
             + 0.32 * np.exp(-0.5 * ((h - 19.0) / 2.2) ** 2))
    carbon = (430.0
              - 330.0 * np.exp(-0.5 * ((h - 13.0) / 3.0) ** 2)
              + 130.0 * np.exp(-0.5 * ((h - 20.0) / 2.5) ** 2))
    t_amb = 22.0 + 8.0 * np.sin((h - 8.0) / 24.0 * 2.0 * np.pi)
    return price, carbon, pv, t_amb


def effective_tariff(price, carbon, carbon_price: float | None = None):
    """$/kWh equivalent combining energy price and internal carbon price."""
    lam = CARBON_PRICE if carbon_price is None else carbon_price
    return price + lam * carbon


# ----------------------------- KPI helpers ----------------------------------
def schedule_kpis(levels_idx, b_ch, b_dis, t_slots: int = T_SLOTS,
                  pv_override: np.ndarray | None = None):
    """Compute grid series and token-centric KPIs for a schedule.

    levels_idx : (T,) int array of compute-level indices
    b_ch, b_dis : (T,) 0/1 arrays of battery charge/discharge slots
    pv_override : optional PV series (kW) replacing the July default
    """
    price, carbon, pv, t_amb = profiles(t_slots)
    if pv_override is not None:
        pv = np.asarray(pv_override, dtype=float)[:t_slots]
    u = U_LEVELS[np.asarray(levels_idx)]
    load = facility_power_kw(u, t_amb)
    grid = load + P_CH_KW * np.asarray(b_ch) - P_DIS_KW * np.asarray(b_dis) - pv
    grid = np.maximum(grid, 0.0)  # no export remuneration
    energy_kwh = grid * DT_H
    tokens = throughput_mtok_h(u).sum() * DT_H
    cost = float((price * energy_kwh).sum())
    co2_kg = float((carbon * energy_kwh).sum() / 1e3)
    return {
        "tokens_Mtok": float(tokens),
        "grid_kwh": float(energy_kwh.sum()),
        "cost_usd": cost,
        "co2_kg": co2_kg,
        "usd_per_Mtok": cost / tokens,
        "gco2_per_Mtok": co2_kg * 1e3 / tokens,
        "peak_grid_kw": float(grid.max()),
        "evening_grid_kwh": float(energy_kwh[17:22].sum()),
        "grid_series_kw": grid,
        "load_series_kw": load,
    }


def bau_baseline_kpis(pv_override: np.ndarray | None = None):
    """KPIs of the demand-tracking business-as-usual reference schedule."""
    z = np.zeros(T_SLOTS, dtype=int)
    return schedule_kpis(BAU_LEVELS, z, z, pv_override=pv_override)
