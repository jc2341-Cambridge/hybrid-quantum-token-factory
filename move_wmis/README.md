# Move-WMIS pricing

Local **move-space MWIS** package used for Aquila / classical / exact pricing
around a feasible 240-bit window reference.

## What it does

- Generate local compute / battery / batch / swap moves (`generators.py`)
- Build the move conflict graph and solve exact MWIS (`conflict.py`)
- Score moves, diversity-prune, and run AHS-surrogate sampling (`samplers.py`)
- Apply an independent set of moves without bit repair (`apply.py`, `pipeline.py`)

Physical validity is the unit-disk independent set on the
`integrated_model.encoding` register (240 sites, 192 occupied).

## Quick import

```python
from move_wmis import (
    generate_window_moves,
    build_conflict_adjacency,
    solve_exact_mwis,
    run_move_wmis_pipeline,
)
from integrated_model import encoding

print(encoding.N_SITES, encoding.N_WEIGHTED_SITES)
```

Bits returned by the pipeline are never repaired; forward audit and exact
reduced-cost admission belong to the full campaign stack (not redistributed
here).
