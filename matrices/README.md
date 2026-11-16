# Matrices

Versioned **derived** QUBO artefacts. Raw market and workload traces are not
redistributed.

| Stem | Role |
|------|------|
| `register_unit_weight_Q.npy` | Unit-weight MWIS on the 240-site physical unit-disk graph |
| `move_mwis_k64_seed104_Q.npy` | Move-space MWIS for confirmatory pricing (\(K=64\), seed 104) |
| `move_mwis_k64_seed104_instance.json` | Locked weights/edges recipe used to rebuild the move Q |

Energy convention:

\[
E(\mathbf{x}) = \mathbf{x}^{\top} Q \mathbf{x}, \qquad \mathbf{x}\in\{0,1\}^n.
\]

```bash
python scripts/export_matrices.py
```

Hashes are recorded in `*_meta.json` and summarised in `MANIFEST.json`.
