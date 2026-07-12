"""One-shot packager: export paper CSVs into code-to-commit/data."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "code"
OLD_DATA = ROOT / "data"
DST = Path(__file__).resolve().parent
DATA = DST / "data"

cats = {
    "01_market": DATA / "01_market",
    "02_facility": DATA / "02_facility",
    "03_milp_results": DATA / "03_milp_results",
    "04_quantum": DATA / "04_quantum",
    "05_sensitivity": DATA / "05_sensitivity",
}
for p in cats.values():
    p.mkdir(parents=True, exist_ok=True)
(DST / "output").mkdir(parents=True, exist_ok=True)

# --- 01 market ---
shutil.copy2(
    OLD_DATA / "real_market" / "east_china_july_profiles.csv",
    cats["01_market"] / "east_china_july_profiles.csv",
)
ctx = OLD_DATA / "01_context_profiles" / "hourly_context_profiles.csv"
if ctx.exists():
    shutil.copy2(ctx, cats["01_market"] / "hourly_context_profiles.csv")

sys.path.insert(0, str(SRC))
import model as m  # noqa: E402

rows = []
for month, name in [(1, "Jan"), (4, "Apr"), (7, "Jul"), (10, "Oct")]:
    g = m.PVGIS_G_W_M2[month]
    pv = m.pv_profile_kw(month)
    for h in range(24):
        rows.append({
            "month": month,
            "month_name": name,
            "hour": h,
            "irradiance_w_m2": float(g[h]),
            "pv_kw": float(pv[h]),
        })
pd.DataFrame(rows).to_csv(cats["01_market"] / "seasonal_pv_profiles.csv", index=False)

prov = json.loads((OLD_DATA / "real_market" / "provenance.json").read_text(encoding="utf-8"))
flat: list[dict] = []


def flatten(prefix: str, obj) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v)
    else:
        flat.append({
            "key": prefix,
            "value": obj if isinstance(obj, (str, int, float, bool)) else json.dumps(obj),
        })


flatten("", prov)
pd.DataFrame(flat).to_csv(cats["01_market"] / "market_provenance.csv", index=False)

# --- 02 facility ---
for name in ["compute_levels.csv", "token_energy_curves.csv"]:
    src = OLD_DATA / "02_token_energy_model" / name
    if src.exists():
        shutil.copy2(src, cats["02_facility"] / name)

# --- 03 milp results ---
rev = SRC / "output_revised"
for name in ["headline_kpis.csv", "hourly_dispatch.csv", "supported_tradeoff.csv"]:
    shutil.copy2(rev / name, cats["03_milp_results"] / name)

# --- 04 quantum ---
fid = json.loads((rev / "surrogate_fidelity.json").read_text(encoding="utf-8"))
pd.DataFrame([
    {"metric": "compute_high_block_agreement",
     "value": fid["agreement"]["compute_high_block_qubo_vs_free_milp"]},
    {"metric": "discharge_block_agreement",
     "value": fid["agreement"]["discharge_block_qubo_vs_free_milp"]},
    {"metric": "n_blocks", "value": fid["agreement"]["n_blocks"]},
    {"metric": "milp_mean_tariff_on_discharge_blocks",
     "value": fid["tariff_arbitrage_fidelity"]["milp_mean_tariff_on_discharge_blocks"]},
    {"metric": "qubo_mean_tariff_on_discharge_blocks",
     "value": fid["tariff_arbitrage_fidelity"]["qubo_mean_tariff_on_discharge_blocks"]},
    {"metric": "qubo_mean_tariff_on_high_blocks",
     "value": fid["tariff_arbitrage_fidelity"]["qubo_mean_tariff_on_high_blocks"]},
]).to_csv(cats["04_quantum"] / "surrogate_fidelity_summary.csv", index=False)

pd.DataFrame({
    "block": list(range(8)),
    "block_mean_tariff_usd_kwh": fid["tariff_arbitrage_fidelity"]["block_mean_tariff"],
}).to_csv(cats["04_quantum"] / "block_mean_tariffs.csv", index=False)

Q = np.load(SRC / "output" / "reduced_qubo_matrix.npy")
pd.DataFrame(
    Q,
    index=[f"q{i}" for i in range(Q.shape[0])],
    columns=[f"q{i}" for i in range(Q.shape[1])],
).to_csv(cats["04_quantum"] / "reduced_qubo_matrix.csv", index_label="row")

spec = np.load(SRC / "output" / "reduced_spectrum.npy")
n = 16
states = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1
feasible = ((states[:, 0::2].sum(1) == 5) & (states[:, 1::2].sum(1) == 3))
emin = float(spec.min())
pd.DataFrame({
    "state_index": np.arange(len(spec)),
    "energy": spec,
    "energy_above_optimum": spec - emin,
    "feasible": feasible.astype(int),
}).to_csv(cats["04_quantum"] / "reduced_spectrum.csv", index=False)

Qf = np.load(SRC / "output" / "full_qubo_matrix.npy")
pd.DataFrame(Qf, columns=[f"q{i}" for i in range(Qf.shape[1])]).to_csv(
    cats["04_quantum"] / "full_qubo_matrix.csv", index=False
)

shutil.copy2(SRC / "output" / "qaoa_depth_sweep.csv",
             cats["04_quantum"] / "qaoa_depth_sweep.csv")
shutil.copy2(rev / "quantum_edge_trials.csv",
             cats["04_quantum"] / "quantum_edge_trials.csv")

edge = json.loads((rev / "quantum_edge_summary.json").read_text(encoding="utf-8"))
pd.DataFrame([{
    "method": method,
    "hit_rate": edge["hit_rate"][method],
    "feasible_rate": edge["feasible_rate"][method],
    "mean_gap_pct": edge["mean_gap_pct"][method],
    "median_gap_pct": edge["median_gap_pct"][method],
    "n_shots_budget": edge["n_shots_budget"],
    "n_trials": edge["n_trials"],
    "optimum_energy": edge["optimum_energy"],
} for method in ["random", "greedy", "sa", "qaoa"]]).to_csv(
    cats["04_quantum"] / "quantum_edge_summary.csv", index=False
)

hy = edge["hybrid_discharge_milp"]
pd.DataFrame([{
    "free_cost_usd": hy["free_cost"],
    "hybrid_cost_usd": hy["hybrid_cost"],
    "cost_gap_pct": hy["cost_gap_pct"],
    "emissions_gap_pct": hy["emissions_gap_pct"],
    "free_evening_kwh": hy["free_evening_kwh"],
    "hybrid_evening_kwh": hy["hybrid_evening_kwh"],
    "qubo_dis_blocks": json.dumps(hy["qubo_dis_blocks"]),
}]).to_csv(cats["04_quantum"] / "hybrid_discharge_transfer.csv", index=False)

# --- 05 sensitivity ---
for name in ["seasonal.csv", "deadline_sensitivity.csv"]:
    shutil.copy2(rev / name, cats["05_sensitivity"] / name)

# header-only templates (tracked via exception in .gitignore if desired)
tpl = DATA / "_templates"
tpl.mkdir(exist_ok=True)
pd.DataFrame(columns=[
    "hour", "tou_period", "price_cny_kwh", "price_usd_kwh",
    "carbon_g_kwh", "pv_kw", "temperature_c",
]).to_csv(tpl / "east_china_july_profiles.TEMPLATE.csv", index=False)

print("Exported:")
for p in sorted(DATA.rglob("*.csv")):
    print(f"  {p.relative_to(DST)}  ({p.stat().st_size:,} bytes)")
