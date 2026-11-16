"""Export or rebuild versioned QUBO matrices (derived artefacts only).

No raw market, workload, or campaign logs are written. Rebuilds use either:

* the locked move-graph instance JSON under ``matrices/``, or
* (maintainers) the local hardware artefact path when present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms import encoding  # noqa: E402
from algorithms.qubo import build_qubo_matrix, qubo_sha256  # noqa: E402

OUT = ROOT / "matrices"
LOCKED_INSTANCE = OUT / "move_mwis_k64_seed104_instance.json"
MAINTAINER_ARTIFACT = (
    Path(r"D:\codex-writing\Rydberg-algorithms\route4-sciencebest")
    / "hardware"
    / "artifacts"
    / "move_graph_seed104.json"
)


def _save_q(stem: str, q: np.ndarray, meta: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    q_path = OUT / f"{stem}_Q.npy"
    meta_path = OUT / f"{stem}_meta.json"
    np.save(q_path, np.asarray(q, dtype=np.float64))
    meta = {
        **meta,
        "shape": list(q.shape),
        "dtype": "float64",
        "energy": "E(x) = x.T @ Q @ x",
        "sha256": qubo_sha256(q),
        "path_Q": q_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {q_path.name}  sha256={meta['sha256']}")
    return meta


def export_register_unit_weight() -> dict:
    weights = np.ones(encoding.N_SITES, dtype=float)
    edges = encoding.unit_disk_edges()
    q = build_qubo_matrix(weights, edges)
    meta = {
        "id": "register_unit_weight",
        "kind": "physical_unit_disk_mwis",
        "n_sites": encoding.N_SITES,
        "n_occupied": encoding.N_WEIGHTED_SITES,
        "n_edges": len(edges),
        "blockade_radius_um": encoding.BLOCKADE_RADIUS_UM,
        "edge_sha256": encoding.edge_sha256(edges),
        "convention": "Q_ii=-w_i; Q_ij=Q_ji=P/2; P=1+2 max|w|",
        "weights": "unit weight on all 240 register sites",
    }
    return _save_q("register_unit_weight", q, meta)


def _load_move_instance() -> dict:
    if LOCKED_INSTANCE.is_file():
        return json.loads(LOCKED_INSTANCE.read_text(encoding="utf-8"))
    if not MAINTAINER_ARTIFACT.is_file():
        raise FileNotFoundError(
            "missing matrices/move_mwis_k64_seed104_instance.json "
            "(and no maintainer artefact path)"
        )
    artifact = json.loads(MAINTAINER_ARTIFACT.read_text(encoding="utf-8"))
    weights = [float(v) for v in artifact["weights"]]
    edges = [list(map(int, edge)) for edge in artifact["edges"]]
    q = build_qubo_matrix(weights, [tuple(e) for e in edges])
    digest = qubo_sha256(q)
    if digest != artifact["qubo"]["sha256"]:
        raise ValueError("move Q hash mismatch against hardware artefact")
    instance = {
        "id": "move_mwis_k64_seed104",
        "kind": "move_space_mwis",
        "seed": int(artifact["seed"]),
        "top_k": int(artifact["top_k"]),
        "window_index": artifact.get("window_index"),
        "n_vertices": len(weights),
        "n_edges": len(edges),
        "penalty": float(artifact["qubo"]["penalty"]),
        "formula": artifact["qubo"]["formula"],
        "sense": artifact["qubo"]["sense"],
        "symmetric_matrix_convention": artifact["qubo"]["symmetric_matrix_convention"],
        "weights": weights,
        "edges": edges,
        "qubo_sha256": digest,
        "source_artifact_sha256": artifact.get("artifact_sha256"),
        "note": "Derived move-graph instance for confirmatory pricing; not raw market traces.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOCKED_INSTANCE.write_text(
        json.dumps(instance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {LOCKED_INSTANCE.name}")
    return instance


def export_move_mwis() -> dict:
    instance = _load_move_instance()
    weights = instance["weights"]
    edges = [tuple(edge) for edge in instance["edges"]]
    q = build_qubo_matrix(weights, edges)
    digest = qubo_sha256(q)
    expected = instance["qubo_sha256"]
    if digest != expected:
        raise ValueError(f"move Q hash mismatch: {digest} != {expected}")
    meta = {
        "id": "move_mwis_k64_seed104",
        "kind": "move_space_mwis",
        "seed": int(instance["seed"]),
        "top_k": int(instance["top_k"]),
        "n_vertices": len(weights),
        "n_edges": len(edges),
        "penalty": float(instance["penalty"]),
        "convention": instance["symmetric_matrix_convention"],
        "instance_json": LOCKED_INSTANCE.name,
        "locked_sha256": expected,
    }
    return _save_q("move_mwis_k64_seed104", q, meta)


def write_manifest(entries: list[dict]) -> None:
    manifest = {
        "schema_version": "hqtf.matrices.v1",
        "policy": {
            "included": "Derived QUBO matrices and rebuild recipes (weights/edges).",
            "excluded": "Raw market CSVs, workload traces, and campaign run logs.",
        },
        "matrices": [
            {"id": e["id"], "path_Q": e["path_Q"], "sha256": e["sha256"], "shape": e["shape"]}
            for e in entries
        ],
    }
    path = OUT / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


def main() -> None:
    entries = [export_register_unit_weight(), export_move_mwis()]
    write_manifest(entries)


if __name__ == "__main__":
    main()
