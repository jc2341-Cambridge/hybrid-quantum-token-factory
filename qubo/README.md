# QUBO / Ising package

Self-contained folder for the **quantum layer** of the hybrid classical–quantum
token-factory study: QUBO builders, solvers, hybrid decomposition (Paths B/C),
and **versionable coupling matrices**.

## Canonical encoding (manuscript)

| Item | Value |
|------|-------|
| Qubits | **48** |
| Hours | 24 × 1 h |
| Charge bits | `q0..q23` = \(c_t\) |
| Discharge bits | `q24..q47` = \(d_t\) |
| Budgets | \(\sum c = 4\), \(\sum d = 4\) |
| Search space | \(2^{48}\) (sample-based diagnostics) |
| CIM upload | `matrices/storage_master_48_Q.npy` |

Energy: \(E = x^\top Q x + q_0\) with \(x \in \{0,1\}^{48}\).

---

## What is uploaded vs what is not

| Content | In this folder? | Notes |
|---------|-----------------|-------|
| **48-qubit** storage-master Q (`.npy` + `.csv` + meta) | **Yes — commit** | Derived; no proprietary market traces |
| Legacy separable 16-qubit Q (negative control) | **Yes — commit** | Top-tariff / old QUBO |
| Builders (`builders.py`) | **Yes** | Default `build_storage_master_qubo()` → 48 qubits |
| Exact / annealing / QAOA solvers (`solvers.py`) | **Yes** | Sample-based for \(n=48\) |
| Hybrid Path B / Path C (`hybrid_decomposition.py`) | **Yes** | Master ↔ MILP with cuts |
| Schematic scripts (CIM / QAOA / fig13) | **Yes** under `scripts/` | Figure generators |
| Raw market / operator CSVs | **No** | Parent `data/` (often gitignored) |
| Runtime `output/` | **No** | gitignored |

**Removed (outdated):** 16-qubit block-aggregated `storage_master_*`, `full_120binary_*`.

---

## Layout

```
qubo/
├── README.md
├── __init__.py
├── builders.py               ← Qubo + build_storage_master_qubo (48 default)
├── solvers.py
├── hybrid_decomposition.py
├── export_matrices.py
├── matrices/                 ← **commit these**
│   ├── MANIFEST.json
│   ├── storage_master_48_Q.npy / .csv / _const.npy / _meta.json / _metrics.json
│   └── legacy_separable_compute_discharge_Q.*
├── scripts/
└── output/                   ← runtime (gitignored)
```

Run commands from **`code-to-commit/`**.

---

## Matrices

### 1. `storage_master_48` — paper master (CIM / Path B–C)

- **Size:** 48 × 48  
- **Coupling:** tariff biases + nested SOC-path + terminal balance + adjacency +
  tariff-product pairs + same-hour exclusivity + seeded pair jitter  
- **Not** equal to the top-tariff greedy / old-QUBO control  

```python
import numpy as np
Q = np.load("qubo/matrices/storage_master_48_Q.npy")
const = float(np.load("qubo/matrices/storage_master_48_const.npy")[0])
```

```python
from qubo.builders import build_storage_master_qubo
q, meta = build_storage_master_qubo()  # defaults: 24 h, 4+4 budgets
```

### 2. `legacy_separable_compute_discharge` — negative control

- **Size:** 16 × 16 (8 three-hour blocks)  
- Discharge optimum ≡ top-3 block-mean tariff mask  
- Used only as the soft control in the hybrid audit (fig13)  

---

## Commands

```bash
cd code-to-commit
python -m qubo.export_matrices
python -m qubo.hybrid_decomposition
```

---

## Simulator note

`storage_master_48_metrics.json` may report classical Metropolis / SA proxy
draws of the same \(H(z)\). The manuscript attributes master execution to the
CIM — replace metrics with hardware samples after loading `Q`.

---

## Citation

Cite the accompanying journal manuscript when reusing these matrices or the
hybrid decomposition.
