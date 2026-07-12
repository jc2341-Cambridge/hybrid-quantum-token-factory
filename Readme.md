# Hybrid classical–quantum scheduling for AI token factories

Reproducible analysis package for the journal manuscript on **energy- and carbon-aware AI token-factory scheduling** with a **hybrid quantum–classical storage master**.

**Canonical quantum encoding: 48 qubits** (24 hourly charge + 24 hourly discharge; budgets 4+4; search space \(2^{48}\)).

This folder is self-contained. Runnable scripts load inputs from relative `data/` and write regenerated artefacts to `output/` (figures to `figures/`).

---

## Quantum / QUBO package (`qubo/`)

**All QUBO code and coupling matrices live under [`qubo/`](qubo/).** See [`qubo/README.md`](qubo/README.md).

| Upload? | Content |
|---------|---------|
| **Yes — commit / CIM** | `qubo/matrices/storage_master_48_Q.npy` + `_const.npy` + `_meta.json` |
| **Yes — commit** | `qubo/matrices/legacy_separable_compute_discharge_*` (negative control, `.npy` + meta) |
| **Yes — commit** | `qubo/builders.py`, `solvers.py`, `hybrid_decomposition.py`, `export_matrices.py` |
| **No** | CSV dumps of \(Q\) (not used for CIM load) |
| **No** | Raw market / operator CSVs (`data/01_market/` is typically gitignored) |
| **No** | Runtime `qubo/output/`, `output/`, `figures/` |

```bash
cd code-to-commit
python -m qubo.export_matrices
python -m qubo.hybrid_decomposition   # optional Path B/C report
```

```python
import numpy as np
Q = np.load("qubo/matrices/storage_master_48_Q.npy")
const = float(np.load("qubo/matrices/storage_master_48_const.npy")[0])
# Energy: E = x @ Q @ x + const,  x in {0,1}^48
```

**CIM upload:** `storage_master_48_Q.npy` (primary). Details in [`qubo/README.md`](qubo/README.md).

---

## 1. Big picture

```
Market + facility profiles  →  classical MILP (workload + SOC)
                                      │
                                      ├─ headline KPIs, dispatch, trade-off
                                      │
                                      └─ hybrid loop
                                               │
                    ┌──────────────────────────┴──────────────────────────┐
                    ▼                                                     ▼
          48-qubit storage master QUBO                          MILP subproblem
          (hourly c_t, d_t + SOC-path /                       (queue, SOC, PV–grid,
           tariff coupling)                                    useful work)
                    │
                    ▼
          Path B feasibility cuts / Path C optimality cuts
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
    CIM (physical)     QAOA (gate-model, same H_C)
```

**Classical layer.** MILP conserves useful token work, co-optimises SOC, and reports absolute cost / emissions / evening-peak import at fixed useful workload (3024 Mtoken).

**Quantum layer.** Discrete hourly charge/discharge masks are ranked by a coupled **48-qubit** Ising master; the MILP restores omitted constraints. A legacy 16-qubit separable QUBO is retained only as a negative control (top-tariff / old QUBO).

**Data philosophy.** Raw proprietary operator traces are not redistributed. **QUBO Q matrices are derived artefacts and are versioned under `qubo/matrices/`.**

---

## 2. Repository layout

```
code-to-commit/
├── Readme.md
├── requirements.txt
├── model.py
├── revised_dispatch.py
├── run_milp_study.py
├── make_figures.py
├── solvers.py                  ← classical MILP helpers
├── qubo/                       ← canonical quantum package
│   ├── README.md
│   ├── builders.py             ← build_storage_master_qubo (default 48)
│   ├── solvers.py
│   ├── hybrid_decomposition.py
│   ├── export_matrices.py
│   ├── matrices/               ← commit Q matrices here
│   │   ├── storage_master_48_Q.npy   ← CIM upload
│   │   ├── storage_master_48_*.{csv,json,npy}
│   │   └── legacy_separable_compute_discharge_*
│   ├── scripts/                ← CIM / QAOA / hybrid figure generators
│   └── output/                 ← gitignored
├── data/                       ← market CSVs often not uploaded
├── output/
└── figures/
```

---

## 3. Environment

```bash
cd code-to-commit
python -m venv .venv
pip install -r requirements.txt
```

Python 3.10+.

---

## 4. How to run

```bash
python run_milp_study.py
python -m qubo.export_matrices
python -m qubo.hybrid_decomposition
python make_figures.py
python -m qubo.scripts.make_cim_spin_graph   # or: python qubo/scripts/make_cim_spin_graph.py
python qubo/scripts/make_qaoa_circuit.py
```

---

## 5. Upload policy (short)

| Path | Commit? |
|------|---------|
| `qubo/matrices/storage_master_48_*` | **Yes** (canonical) |
| `qubo/matrices/legacy_separable_*` | **Yes** (control) |
| `qubo/*.py`, `qubo/README.md` | **Yes** |
| `data/01_market/*.csv` (raw profiles) | **Usually no** |
| `qubo/output/`, `output/`, `figures/` | **No** (runtime) |

Details and matrix schemas: [`qubo/README.md`](qubo/README.md).

---

## 6. Citation

Please cite the accompanying journal manuscript when using this package.
