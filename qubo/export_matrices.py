"""Export QUBO coupling matrices for the manuscript quantum layer.

Writes NumPy (``.npy``) and CSV copies under ``qubo/matrices/``.

Canonical manuscript master: **48-qubit** hourly storage QUBO
(``storage_master_48_*``). Legacy 16-qubit separable QUBO is retained as
the negative control (top-tariff / old QUBO).

Usage (from ``code-to-commit/``)::

    python -m qubo.export_matrices
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qubo.builders import build_reduced_qubo, build_storage_master_qubo

HERE = Path(__file__).resolve().parent
MAT = HERE / "matrices"
MAT.mkdir(parents=True, exist_ok=True)


def _save_Q(name: str, Q: np.ndarray, const: float, meta: dict) -> None:
    np.save(MAT / f"{name}_Q.npy", Q)
    np.save(MAT / f"{name}_const.npy", np.asarray([const], dtype=float))
    n = Q.shape[0]
    cols = [f"q{i}" for i in range(n)]
    df = pd.DataFrame(Q, columns=cols)
    df.insert(0, "row", cols)
    df.to_csv(MAT / f"{name}_Q.csv", index=False)
    # Drop non-JSON-serialisable callables from meta copies.
    clean = {k: v for k, v in meta.items() if k not in ("c_idx", "d_idx")}
    if "tariff_b" in clean:
        clean["tariff_b"] = [float(x) for x in np.asarray(clean["tariff_b"]).ravel()]
    payload = {"name": name, "n": n, "const": float(const), **clean}
    with open(MAT / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {name}: Q shape {Q.shape}, const={const:.6g}")


def main() -> None:
    print("Exporting QUBO matrices →", MAT)

    # --- Canonical 48-qubit hourly master (manuscript / CIM upload) --------
    q_m, meta_m = build_storage_master_qubo(
        n_blocks=24, charge_blocks=4, dis_blocks=4, penalty=5000.0
    )
    _save_Q(
        "storage_master_48",
        q_m.Q,
        q_m.const,
        {
            "kind": "storage_master",
            "n_qubits": 48,
            "n_blocks": int(meta_m["n_blocks"]),
            "hours_per_block": 1,
            "charge_blocks": int(meta_m["charge_blocks"]),
            "dis_blocks": int(meta_m["dis_blocks"]),
            "encoding": "q0..q23=charge; q24..q47=discharge",
            "tariff_b": meta_m["tariff_b"],
            "soc_path_weight": float(meta_m["soc_path_weight"]),
            "adj_weight": float(meta_m["adj_weight"]),
            "exclusivity_weight": float(meta_m["exclusivity_weight"]),
            "tariff_pair_weight": float(meta_m["tariff_pair_weight"]),
            "pair_jitter": float(meta_m["pair_jitter"]),
            "jitter_seed": int(meta_m["jitter_seed"]),
            "note": (
                "48-qubit hourly charge--discharge master with SOC-path, "
                "tariff-pair, adjacency, exclusivity, and seeded pair jitter. "
                "Upload storage_master_48_Q.npy to CIM."
            ),
        },
    )

    # --- Legacy separable 16-qubit control (top-tariff / old QUBO) ---------
    q_r, meta_r = build_reduced_qubo()
    _save_Q(
        "legacy_separable_compute_discharge",
        q_r.Q,
        q_r.const,
        {
            "kind": "legacy_separable",
            "n_blocks": int(meta_r["n_blocks"]),
            "high_blocks": int(meta_r["high_blocks"]),
            "dis_blocks": int(meta_r["dis_blocks"]),
            "encoding": "even = high/low compute; odd = discharge",
            "note": (
                "Separable linear cardinalities; discharge optimum equals "
                "the top-3 high-tariff greedy rule. Negative control only."
            ),
        },
    )

    manifest = {
        "description": (
            "QUBO / Ising coupling matrices for the hybrid quantum--classical "
            "token-factory study (48-qubit hourly storage master). "
            "Market/operator raw data are not redistributed; these Q matrices "
            "are derived artefacts that may be versioned."
        ),
        "canonical_master": "storage_master_48_Q.npy",
        "n_qubits": 48,
        "files": sorted(p.name for p in MAT.glob("*") if p.is_file()),
    }
    with open(MAT / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("done. See matrices/MANIFEST.json")


if __name__ == "__main__":
    main()
