"""Token-factory energy model and 24-h context profiles.

Market price, carbon intensity, and temperature are loaded from
``data/01_market/east_china_july_profiles.csv``. Seasonal PV may be loaded
from ``data/01_market/seasonal_pv_profiles.csv`` when present.

Units: power kW, energy kWh, price $/kWh, carbon gCO2/kWh,
throughput Mtoken/h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import JULY_PROFILES_CSV, SEASONAL_PV_CSV, require_csv

# ----------------------------- cluster parameters ---------------------------
P_IDLE_KW = 250.0          # IT idle power
P_PEAK_KW = 1200.0         # IT-level peak at u = 1
ALPHA = 1.4                # super-linear power-utilisation exponent
R_MAX_MTOK_H = 180.0       # token throughput at u = 1 (Mtoken/h)

U_LEVELS = np.array([0.40, 0.70, 1.00])
LEVEL_NAMES = ["low", "mid", "high"]

P_CH_KW = 300.0
P_DIS_KW = 270.0
N_CYCLE_SLOTS = 4

CARBON_PRICE = 1.5e-4      # $ per gCO2 (150 $/tCO2)

T_SLOTS = 24
DT_H = 1.0

# Demand-tracking BAU levels (iso-output with optimised schedules: 3024 Mtok).
BAU_LEVELS = np.array([0] * 7 + [1] * 5 + [2] * 8 + [1] * 3 + [0], dtype=int)

BATCH_SHARE = 0.30
SLA_FLOOR_LEVELS = np.where(BAU_LEVELS == 2, 1, BAU_LEVELS).astype(int)

# Fallback PVGIS mean-day irradiance (W/m2) if seasonal CSV is absent.
PVGIS_G_W_M2 = {
    1: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 47.67, 164.36, 284.82, 378.43, 426.67,
                 418.45, 367.75, 277.36, 167.66, 43.94, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    4: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.38,
                 99.27, 247.17, 404.76, 547.70, 651.85, 700.54,
                 696.02, 636.40, 523.03, 380.22, 224.24, 80.06,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    7: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.80,
                 133.95, 277.00, 421.91, 545.89, 636.40, 681.81,
                 672.09, 620.17, 531.42, 405.01, 263.83, 128.54,
                 26.10, 0.0, 0.0, 0.0, 0.0, 0.0]),
    10: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                  38.18, 160.83, 305.21, 428.79, 506.46, 534.88,
                  515.48, 443.47, 340.28, 209.12, 75.67, 1.43,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}
PVGIS_G_JULY_W_M2 = PVGIS_G_W_M2[7]
PV_STC_KW = 500.0
PV_PERFORMANCE_RATIO = 0.85


def it_power_kw(u: np.ndarray | float) -> np.ndarray | float:
    return P_IDLE_KW + (P_PEAK_KW - P_IDLE_KW) * np.asarray(u) ** ALPHA


def throughput_mtok_h(u: np.ndarray | float) -> np.ndarray | float:
    return R_MAX_MTOK_H * np.asarray(u)


def pue(t_amb_c: np.ndarray | float) -> np.ndarray | float:
    return 1.15 + 0.02 * np.maximum(0.0, np.asarray(t_amb_c) - 15.0)


def facility_power_kw(u, t_amb_c):
    return pue(t_amb_c) * it_power_kw(u)


def joule_per_token(u, t_amb_c=25.0):
    """Facility-level energy intensity, J/token."""
    p_w = facility_power_kw(u, t_amb_c) * 1e3
    r_tok_s = throughput_mtok_h(u) * 1e6 / 3600.0
    return p_w / r_tok_s


def pv_profile_kw(month: int = 7) -> np.ndarray:
    """AC PV output (kW) for the mean day of the given month.

    Prefers ``data/01_market/seasonal_pv_profiles.csv`` column ``pv_kw``.
    Falls back to embedded PVGIS irradiance if the CSV is absent.
    """
    if SEASONAL_PV_CSV.exists():
        df = pd.read_csv(SEASONAL_PV_CSV)
        sub = df.loc[df["month"] == month].sort_values("hour")
        if len(sub) >= T_SLOTS:
            return sub["pv_kw"].to_numpy(dtype=float)[:T_SLOTS]
    return PV_STC_KW * (PVGIS_G_W_M2[month] / 1000.0) * PV_PERFORMANCE_RATIO


def profiles(t_slots: int = T_SLOTS):
    """Hourly price, carbon intensity, PV and ambient temperature.

    Required CSV: ``data/01_market/east_china_july_profiles.csv``
    Expected columns: hour, price_usd_kwh, carbon_g_kwh, temperature_c
    (optional: tou_period, price_cny_kwh, pv_kw).
    """
    path = require_csv(
        JULY_PROFILES_CSV,
        hint="Copy data/_templates/east_china_july_profiles.TEMPLATE.csv "
             "and fill 24 hourly rows.",
    )
    df = pd.read_csv(path)
    for col in ("price_usd_kwh", "carbon_g_kwh", "temperature_c"):
        if col not in df.columns:
            raise ValueError(
                f"{path} is missing required column '{col}'. "
                "See README.md Expected columns."
            )
    price = df["price_usd_kwh"].to_numpy(dtype=float)[:t_slots]
    carbon = df["carbon_g_kwh"].to_numpy(dtype=float)[:t_slots]
    t_amb = df["temperature_c"].to_numpy(dtype=float)[:t_slots]
    if "pv_kw" in df.columns and df["pv_kw"].notna().any():
        pv = df["pv_kw"].to_numpy(dtype=float)[:t_slots]
    else:
        pv = pv_profile_kw(7)[:t_slots]
    return price, carbon, pv, t_amb


def effective_tariff(price, carbon, carbon_price: float | None = None):
    """$/kWh equivalent combining energy price and internal carbon price."""
    lam = CARBON_PRICE if carbon_price is None else carbon_price
    return price + lam * carbon


def schedule_kpis(levels_idx, b_ch, b_dis, t_slots: int = T_SLOTS,
                  pv_override: np.ndarray | None = None):
    """Compute grid series and token-centric KPIs for a schedule."""
    price, carbon, pv, t_amb = profiles(t_slots)
    if pv_override is not None:
        pv = np.asarray(pv_override, dtype=float)[:t_slots]
    u = U_LEVELS[np.asarray(levels_idx)]
    load = facility_power_kw(u, t_amb)
    grid = load + P_CH_KW * np.asarray(b_ch) - P_DIS_KW * np.asarray(b_dis) - pv
    grid = np.maximum(grid, 0.0)
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
