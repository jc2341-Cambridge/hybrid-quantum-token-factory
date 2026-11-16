# `algorithms.move_wmis`

Move-space maximum-weight independent set (MWIS) pricing around a feasible
240-bit window reference.

| Module | Role |
|--------|------|
| `generators` | Local compute / battery / batch / swap moves |
| `conflict` | Conflict graph and exact MWIS (SciPy MILP) |
| `scoring` | Reduced-cost guided move weights |
| `diversity` | Pruning of near-duplicate moves |
| `samplers` | AHS-surrogate independent-set sampling |
| `move_space_sa` | Simulated annealing on the move graph |
| `pipeline` | End-to-end propose → sample → apply (no bit repair) |
| `apply` | Apply an independent set of moves to the reference |

Physical feasibility is the unit-disk independent set on
`algorithms.encoding` (240 sites, 192 occupied).
