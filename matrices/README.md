# Matrices — quantum / classical solver inputs

This directory is the **data boundary** of the public package. Files here are
**derived QUBO matrices** (and locked rebuild recipes). They are the numerical
objects a quantum processor driver or classical QUBO solver should load.

Raw market CSVs, workload traces, NLR power dumps, and Aquila shot logs are
**not** stored here. Download those from the publishers listed in the root
README.

## Files

| Stem | Role |
|------|------|
| `register_unit_weight_Q.npy` | Unit-weight MWIS on the 240-site physical unit-disk graph |
| `move_mwis_k64_seed104_Q.npy` | Move-space MWIS for confirmatory pricing (\(K=64\), seed 104) |
| `move_mwis_k64_seed104_instance.json` | Locked weights/edges used to rebuild the move \(Q\) |
| `*_meta.json` | Shape, penalty, SHA-256 |
| `MANIFEST.json` | Index of published matrices |

## Energy

\[
E(\mathbf{x}) = \mathbf{x}^{\top} Q \mathbf{x}, \qquad \mathbf{x}\in\{0,1\}^n.
\]

```python
import numpy as np
Q = np.load("matrices/move_mwis_k64_seed104_Q.npy")
# Pass Q to a QUBO/Ising backend (convert to Ising if required).
```

## Rebuild

```bash
python scripts/export_matrices.py
```

Hashes must match `*_meta.json`. Any mismatch means the recipe or encoder
diverged from the locked public contract.
