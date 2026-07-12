# QUBO / Ising package

Quantum layer of the hybrid classical–quantum token-factory study:
builders, solvers, hybrid cuts, and **versioned coupling matrices for CIM upload**.

## CIM upload (use these three files)

Upload the **48-qubit storage master** from `matrices/`:

| File | Role |
|------|------|
| **`storage_master_48_Q.npy`** | Dense \(48\times 48\) QUBO matrix \(Q\) (primary CIM / Ising input) |
| **`storage_master_48_const.npy`** | Scalar offset \(q_0\) (shape `(1,)`) |
| **`storage_master_48_meta.json`** | Encoding, budgets, tariff biases, coupling weights |

**Do not use CSV.** Machine loaders should take the NumPy matrix.

### Encoding

| Item | Value |
|------|-------|
| Qubits | **48** |
| Charge | `q0..q23` = hourly \(c_t\) |
| Discharge | `q24..q47` = hourly \(d_t\) |
| Budgets | \(\sum_t c_t = 4\), \(\sum_t d_t = 4\) |
| Search space | \(2^{48}\) (sample-based diagnostics) |

### Energy

\[
E(\mathbf{x}) = \mathbf{x}^{\top} Q \mathbf{x} + q_0,
\qquad \mathbf{x}\in\{0,1\}^{48}.
\]

```python
import numpy as np

Q = np.load("qubo/matrices/storage_master_48_Q.npy")          # (48, 48)
q0 = float(np.load("qubo/matrices/storage_master_48_const.npy")[0])
# E = x @ Q @ x + q0
```

If the CIM expects Ising \((J,h)\) rather than QUBO \(Q\), convert with the paper map
\(s_i = 1-2z_i\) (standard QUBO⇄Ising); the uploaded object of record remains `storage_master_48_Q.npy`.

### Hardware sampling checklist

1. Load `storage_master_48_Q.npy` (+ `const` if the driver needs the absolute energy).
2. Draw bitstrings (equal sample budget vs any classical baseline).
3. Score each sample with \(E = x^{\top}Qx + q_0\).
4. Optional: filter or report cardinality feasibility (\(\sum c=4\), \(\sum d=4\)).
5. Histogram energy above the incumbent for Fig.~4(b)-style panels.

---

## Layout

```
qubo/
├── README.md
├── builders.py
├── solvers.py
├── hybrid_decomposition.py
├── export_matrices.py
├── matrices/                 ← commit .npy + meta (no CSV)
│   ├── MANIFEST.json
│   ├── storage_master_48_Q.npy      ← CIM upload
│   ├── storage_master_48_const.npy
│   ├── storage_master_48_meta.json
│   └── legacy_separable_compute_discharge_*   (negative control only)
└── scripts/
```

Run from **`code-to-commit/`**.

```bash
python -m qubo.export_matrices
```

Rebuild overwrites `.npy` / `_meta.json` only (no CSV).

---

## Other matrices

`legacy_separable_compute_discharge_*` (16 qubits) is the **negative control**
(top-tariff / old QUBO). Not the manuscript CIM master.

---

## Citation

Cite the accompanying journal manuscript when reusing these matrices or the hybrid loop.
