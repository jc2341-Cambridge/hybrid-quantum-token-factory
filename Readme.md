# Hybrid classical–quantum scheduling for AI token factories

Reproducible analysis package for the journal manuscript on **energy- and carbon-aware AI token-factory scheduling** with a **quantum-compatible optimisation layer** (block QUBO / Ising → CIM + QAOA).

This folder is self-contained. All runnable scripts load inputs from relative `data/` and write regenerated artefacts to `output/` (and figures to `figures/`).

---

## 1. Big picture

```
Market + facility profiles  →  classical MILP (workload + SOC)
                                      │
                                      ├─ headline KPIs, dispatch, trade-off
                                      │
                                      └─ tariff–discharge arbitrage core
                                               │
                                               ▼
                                         16-qubit block QUBO / Ising
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
                   CIM (physical)                          QAOA (gate-model)
                         │                                           │
                         └──────── fidelity + hybrid transfer ───────┘
```

**Classical layer.** A mixed-integer linear program (MILP) conserves useful token work (interactive same-slot + batch backlog with deadline), co-optimises battery state of charge (SOC), and reports absolute cost / emissions / evening-peak grid import at fixed useful workload (3024 Mtoken).

**Quantum layer.** The discrete tariff–compute–discharge skeleton is aggregated into eight three-hour blocks (16 spins/qubits) and encoded as a QUBO / Ising instance. The same Hamiltonian is executed on a coherent Ising machine (CIM) and studied with QAOA statevector simulation. Equal-budget trials and hybrid transfer of the discharge skeleton link the quantum subproblem back to the energy MILP.

**Data philosophy.** Raw proprietary operator traces are not redistributed. Users must place (or regenerate) CSV inputs under `data/` following the schemas below. A filled East China case used in the paper is packaged locally for development; `data/.gitignore` excludes those CSVs from version control.

---

## 2. Repository layout

```
code-to-commit/
├── README.md                 ← this file
├── requirements.txt
├── paths.py                  ← DATA / OUTPUT path helpers
├── model.py                  ← facility physics + profile loader
├── revised_dispatch.py       ← workload-conserving MILP
├── qubo.py                   ← full + reduced QUBO builders
├── solvers.py                ← exact / annealing / QAOA statevector
├── run_milp_study.py         ← regenerate MILP result tables
├── surrogate_fidelity.py     ← QUBO ↔ MILP agreement audit
├── quantum_edge_study.py     ← equal-budget edge + hybrid transfer
├── make_figures.py           ← paper-facing result figures
├── make_cim_spin_graph.py    ← CIM spin-graph schematic
├── make_qaoa_circuit.py      ← QAOA circuit schematic
├── data/
│   ├── .gitignore
│   ├── _templates/           ← header-only CSV templates (tracked)
│   ├── 01_market/            ← exogenous profiles (user-supplied)
│   ├── 02_facility/          ← compute-level / intensity tables
│   ├── 03_milp_results/      ← MILP KPI / dispatch / trade-off tables
│   ├── 04_quantum/           ← QUBO / QAOA / edge / fidelity tables
│   └── 05_sensitivity/       ← seasonal + deadline tables
├── output/                   ← runtime regenerations (gitignored)
└── figures/                  ← rendered PNGs (gitignored)
```

---

## 3. Environment

```bash
cd code-to-commit
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+ (type hints use `|` unions).

---

## 4. How to run (reproducible pipeline)

Place required market CSV(s) under `data/01_market/` (see Section 6). Then:

```bash
# 1) Classical energy study
python run_milp_study.py

# 2) Quantum surrogate audits
python surrogate_fidelity.py
python quantum_edge_study.py

# 3) Figures (reads model + output CSVs; QUBO panels can use data/04_quantum CSVs)
python make_figures.py
python make_cim_spin_graph.py
python make_qaoa_circuit.py
```

Optional: re-export packaged result CSVs into `data/0x_*` after a run by copying from `output/` into the matching category folders.

---

## 5. Data categories (what lives where)

| Folder | Role |
|--------|------|
| `01_market` | Exogenous hourly price, carbon, PV, temperature; seasonal PV; provenance keys |
| `02_facility` | Discrete compute levels and joule-per-token curves |
| `03_milp_results` | Headline KPIs, 24 h dispatch, supported cost–emissions trade-off |
| `04_quantum` | Reduced/full QUBO matrices, spectrum, QAOA depth sweep, edge trials, fidelity |
| `05_sensitivity` | Seasonal PV days; batch-deadline sweep |
| `_templates` | Header-only templates for user-supplied inputs |

---

## 6. Expected columns (English schemas)

Scripts validate required columns at load time and raise `FileNotFoundError` / `ValueError` with a pointer here if something is missing.

### 6.1 `data/01_market/east_china_july_profiles.csv` (**required**)

24 hourly rows. Template: `data/_templates/east_china_july_profiles.TEMPLATE.csv`.

| Column | Type | Meaning |
|--------|------|---------|
| `hour` | int 0–23 | Hour of day |
| `tou_period` | str | Optional TOU label (`valley` / `flat` / `peak` / `tip` or `critical_peak`) |
| `price_cny_kwh` | float | Optional local-currency price |
| `price_usd_kwh` | float | **Required.** Electricity price ($/kWh) |
| `carbon_g_kwh` | float | **Required.** Grid carbon intensity (gCO₂/kWh) |
| `pv_kw` | float | Optional on-site PV (kW). If blank, July PV from seasonal table / fallback is used |
| `temperature_c` | float | **Required.** Ambient temperature (°C) for PUE |

Loaded by `model.profiles()`.

### 6.2 `data/01_market/seasonal_pv_profiles.csv` (recommended)

| Column | Type | Meaning |
|--------|------|---------|
| `month` | int | Calendar month (1, 4, 7, 10 in the paper case) |
| `month_name` | str | Optional label |
| `hour` | int 0–23 | Hour of day |
| `irradiance_w_m2` | float | Optional plane-of-array irradiance |
| `pv_kw` | float | **Required for seasonal runs.** AC PV power (kW) |

### 6.3 `data/01_market/hourly_context_profiles.csv` (optional diagnostic)

| Column | Meaning |
|--------|---------|
| `hour` | Hour of day |
| `price_usd_kwh` | Price |
| `carbon_g_kwh` | Carbon intensity |
| `pv_kw` | PV |
| `t_amb_c` | Temperature |
| `effective_tariff_usd_kwh` | Price + λ × carbon |

### 6.4 `data/01_market/market_provenance.csv` (optional metadata)

| Column | Meaning |
|--------|---------|
| `key` | Dot-path provenance key |
| `value` | Source note / parameter |

### 6.5 `data/02_facility/compute_levels.csv`

| Column | Meaning |
|--------|---------|
| `level` | Level name (`low` / `mid` / `high`) |
| `utilisation` | Utilisation \(u_k\in(0,1]\) |
| `it_power_kw` | IT power at that level |
| `facility_power_kw_nominal` | Facility power at nominal temperature |
| `throughput_mtok_h` | Token throughput (Mtoken/h) |

### 6.6 `data/02_facility/token_energy_curves.csv`

| Column | Meaning |
|--------|---------|
| `utilisation` | Continuous utilisation grid |
| `t_amb_c` | Ambient temperature (°C) |
| `j_per_token` | Facility energy intensity (J/token) |
| `facility_power_kw` | Facility power (kW) |

### 6.7 `data/03_milp_results/headline_kpis.csv`

| Column | Meaning |
|--------|---------|
| `scenario` | `BAU` / `compute_only` / `joint_compute_storage` |
| `useful_tokens_Mtok` | Conserved useful work |
| `total_cost_usd` | Daily cost including degradation |
| `energy_cost_usd` | Energy bill |
| `degradation_cost_usd` | Battery degradation charge |
| `emissions_kg` | Absolute CO₂ (kg) |
| `usd_per_Mtok` | Secondary unit cost |
| `gco2_per_Mtok` | Secondary unit emissions |
| `peak_grid_kw` | Peak grid import |
| `evening_grid_kwh` | Evening-peak window import (kWh) |
| `solve_time_s` | MILP wall time |
| `mip_gap` | Solver gap |

### 6.8 `data/03_milp_results/hourly_dispatch.csv`

| Column | Meaning |
|--------|---------|
| `hour` | Hour of day |
| `price_usd_kwh`, `carbon_g_kwh`, `pv_kw`, `temperature_c` | Exogenous signals |
| `total_arrivals_Mtok` | Total arrivals |
| `interactive_arrivals_Mtok` | Interactive arrivals |
| `batch_arrivals_Mtok` | Batch arrivals |
| `batch_service_Mtok` | Batch service |
| `batch_backlog_Mtok` | End-of-slot backlog |
| `level_BAU`, `level_revised` | Compute level indices 0/1/2 |
| `grid_BAU_kw`, `grid_revised_kw` | Grid import |
| `charge`, `discharge` | Battery binary actions |
| `soc_start_kwh`, `soc_end_kwh` | SOC trajectory |

### 6.9 `data/03_milp_results/supported_tradeoff.csv`

Same KPI columns as headline, plus:

| Column | Meaning |
|--------|---------|
| `lambda_usd_tco2` | Internal carbon price ($/tCO₂) used in the weighted sweep |

### 6.10 `data/04_quantum/reduced_qubo_matrix.csv`

16×16 QUBO coupling matrix. First column `row` labels `q0`…`q15`; remaining columns `q0`…`q15` are matrix entries \(Q_{ij}\).

### 6.11 `data/04_quantum/reduced_spectrum.csv`

| Column | Meaning |
|--------|---------|
| `state_index` | Integer 0 … 2¹⁶−1 |
| `energy` | QUBO energy of that bitstring |
| `energy_above_optimum` | Energy − min energy |
| `feasible` | 1 if compute/discharge equality budgets hold |

### 6.12 `data/04_quantum/full_qubo_matrix.csv`

120×120 denser structural QUBO (comparison object; not the revised operational MILP). Columns `q0`…`q119`.

### 6.13 `data/04_quantum/qaoa_depth_sweep.csv`

| Column | Meaning |
|--------|---------|
| `p` | QAOA depth |
| `approx_ratio` | Penalty-normalised approximation ratio |
| `p_ground` | Ground-state probability |
| `energy` | Best / expectation energy as recorded |
| `time_s` | Simulation time |

### 6.14 `data/04_quantum/quantum_edge_trials.csv`

| Column | Meaning |
|--------|---------|
| `trial` | Trial index |
| `method` | `random` / `greedy` / `sa` / `qaoa` |
| `energy` | Best energy under the equal sample budget |
| `gap_pct` | Optimality gap (%) |
| `hit_optimum` | 1 if enumerated optimum hit |
| `feasible` | 1 if budget-feasible |
| `time_s` | Runtime |

### 6.15 `data/04_quantum/quantum_edge_summary.csv`

Aggregated hit rates / mean gaps per method (`method`, `hit_rate`, `feasible_rate`, `mean_gap_pct`, `median_gap_pct`, `n_shots_budget`, `n_trials`, `optimum_energy`).

### 6.16 `data/04_quantum/hybrid_discharge_transfer.csv`

| Column | Meaning |
|--------|---------|
| `free_cost_usd` | Unconstrained MILP cost |
| `hybrid_cost_usd` | MILP with QUBO discharge skeleton imposed |
| `cost_gap_pct` | Relative cost gap |
| `emissions_gap_pct` | Relative emissions gap |
| `free_evening_kwh`, `hybrid_evening_kwh` | Evening-peak import |
| `qubo_dis_blocks` | JSON list of 8 discharge-block bits |

### 6.17 `data/04_quantum/surrogate_fidelity_summary.csv`

| Column | Meaning |
|--------|---------|
| `metric` | Agreement / tariff-fidelity metric name |
| `value` | Numeric value |

### 6.18 `data/04_quantum/block_mean_tariffs.csv`

| Column | Meaning |
|--------|---------|
| `block` | Block index 0…7 |
| `block_mean_tariff_usd_kwh` | Mean effective tariff on that 3 h block |

### 6.19 `data/05_sensitivity/seasonal.csv`

| Column | Meaning |
|--------|---------|
| `season` | Month / season label |
| `baseline_cost_usd`, `revised_cost_usd` | BAU vs joint costs |
| `baseline_emissions_kg`, `revised_emissions_kg` | Absolute emissions |
| `cost_saving_pct`, `emissions_saving_pct` | Percentage changes |

### 6.20 `data/05_sensitivity/deadline_sensitivity.csv`

Headline-like KPI columns plus:

| Column | Meaning |
|--------|---------|
| `max_batch_delay_h` | Maximum batch delay \(L\) (h) |
| `max_backlog_Mtok` | Peak backlog observed |

---

## 7. Module map (analysis code)

| Script | Reads from `data/` | Writes |
|--------|--------------------|--------|
| `model.py` | `01_market/*.csv` | — |
| `revised_dispatch.py` | via `model.profiles()` | — |
| `run_milp_study.py` | market profiles | `output/*.csv` |
| `qubo.py` / `solvers.py` | via model tariffs | in-memory / caller |
| `surrogate_fidelity.py` | market + model | `output/surrogate_fidelity.*` |
| `quantum_edge_study.py` | market + model | `output/quantum_edge_*.csv` |
| `make_figures.py` | market + `04_quantum` CSVs (fallback) + `output/` | `figures/*.png` |
| `make_cim_spin_graph.py` | — (schematic) | `figures/fig05b_cim_spin_graph.png` |
| `make_qaoa_circuit.py` | — (schematic) | `figures/fig05_qaoa_circuit.png` |

---

## 8. Scope notes (for re-users)

- Headline energy KPIs are **classical MILP** outcomes.
- The 16-qubit block QUBO **omits** backlog, SOC, deadlines, and mid utilisation; see the manuscript Table on retained/omitted elements.
- Equal-budget QAOA evidence is on the **block surrogate**, not a claim that QAOA replaces the full operational MILP.
- CIM hardware execution of \(H_C\) and QAOA statevector reference share the **same** Ising instance.

---

## 9. Citation

Please cite the accompanying journal manuscript when using this package.
