"""Route C+ 240-site triangular unit-disk encoding.

The physical unit-disk graph is the hard codebook.  Only six same-slot
proposition pairs per slot are interpreted as logical exclusions; every other
geometric edge is disclosed as a conservative codebook restriction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Sequence

import numpy as np

WINDOW_SLOTS = 16
ROWS_PER_SLOT = 15
N_SITES = WINDOW_SLOTS * ROWS_PER_SLOT
WEIGHTED_ROWS = tuple(range(12))
GUARD_ROWS = (12, 13, 14)
N_WEIGHTED_SITES = WINDOW_SLOTS * len(WEIGHTED_ROWS)
N_GUARD_SITES = WINDOW_SLOTS * len(GUARD_ROWS)

# Hardware-readiness orientation: semantic row is the x coordinate and slot
# is the y coordinate.  The 0.1 um-snapped staggered-row triangular layout is
# intentionally slightly stretched vertically: horizontal neighbours are
# exactly 5.0 um apart and diagonal neighbours are about 5.061 um apart.
SEMANTIC_COLUMNS = ROWS_PER_SLOT
SLOT_ROWS = WINDOW_SLOTS
SEMANTIC_PITCH_UM = 5.0
SLOT_PITCH_UM = 4.4
PARITY_OFFSET_UM = 2.5
POSITION_RESOLUTION_UM = 0.1
BLOCKADE_RADIUS_UM = 8.9
MIN_SPACING_UM = 5.0
ORIGIN_UM = (1.2, 5.0)
AQUILA_AREA_UM = (75.0, 76.0)
AQUILA_SITE_LIMIT = 256
OPERATIONAL_SITE_LIMIT = 240
LOCAL_WEIGHTED_SITE_LIMIT = 200

COMPUTE_LEVEL_NAMES = ("low", "medium", "high")
BATTERY_ACTION_NAMES = ("idle", "charge", "discharge")
BATCH_ACTION_NAMES = ("work_conserving", "deadline_only", "defer")


@dataclass(frozen=True)
class RowSemantic:
    row: int
    family: str
    asset: int | None
    proposition: str
    vacuum_action: str | None
    local_weighted: bool
    mathematical_claim: str


ROW_SEMANTICS = (
    RowSemantic(0, "compute", 0, "island_0_medium", "low", True, "medium/high are mutually exclusive"),
    RowSemantic(1, "compute", 0, "island_0_high", "low", True, "medium/high are mutually exclusive"),
    RowSemantic(2, "compute", 1, "island_1_medium", "low", True, "medium/high are mutually exclusive"),
    RowSemantic(3, "compute", 1, "island_1_high", "low", True, "medium/high are mutually exclusive"),
    RowSemantic(4, "compute", 2, "island_2_medium", "low", True, "medium/high are mutually exclusive"),
    RowSemantic(5, "compute", 2, "island_2_high", "low", True, "medium/high are mutually exclusive"),
    RowSemantic(6, "battery", 0, "battery_0_charge", "idle", True, "charge/discharge are mutually exclusive"),
    RowSemantic(7, "battery", 0, "battery_0_discharge", "idle", True, "charge/discharge are mutually exclusive"),
    RowSemantic(8, "battery", 1, "battery_1_charge", "idle", True, "charge/discharge are mutually exclusive"),
    RowSemantic(9, "battery", 1, "battery_1_discharge", "idle", True, "charge/discharge are mutually exclusive"),
    RowSemantic(
        10,
        "batch_control",
        None,
        "deadline_only",
        "work_conserving",
        True,
        "the two excited batch states are mutually exclusive",
    ),
    RowSemantic(
        11,
        "batch_control",
        None,
        "defer",
        "work_conserving",
        True,
        "the two excited batch states are mutually exclusive",
    ),
    RowSemantic(
        12,
        "guard_state",
        None,
        "queue_marker",
        None,
        False,
        "no queue-feasibility claim; zero-local-weight marker only",
    ),
    RowSemantic(
        13,
        "guard_state",
        None,
        "soc_marker",
        None,
        False,
        "no SOC-feasibility claim; zero-local-weight marker only",
    ),
    RowSemantic(
        14,
        "guard_state",
        None,
        "peak_marker",
        None,
        False,
        "no peak-feasibility claim; zero-local-weight marker only",
    ),
)

PAIR_ROWS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
)


@dataclass(frozen=True)
class DecodedTrajectory:
    """A complete operational action trajectory decoded without bit changes."""

    bits: tuple[int, ...]
    selected_sites: tuple[int, ...]
    compute_levels: tuple[tuple[int, ...], ...]
    battery_actions: tuple[tuple[str, ...], ...]
    batch_actions: tuple[str, ...]
    guard_bits: tuple[tuple[int, ...], ...]
    vacuum_is_action: bool = True
    repair_applied: bool = False


def atom_index(slot_offset: int, row: int) -> int:
    if not 0 <= int(slot_offset) < WINDOW_SLOTS:
        raise IndexError("slot_offset outside the 16-slot pricing window")
    if not 0 <= int(row) < ROWS_PER_SLOT:
        raise IndexError("row outside the 15-row semantic register")
    return int(slot_offset) * ROWS_PER_SLOT + int(row)


def atom_semantic(index: int) -> dict:
    if not 0 <= int(index) < N_SITES:
        raise IndexError("site index outside the 240-site register")
    slot_offset, row = divmod(int(index), ROWS_PER_SLOT)
    payload = asdict(ROW_SEMANTICS[row])
    payload.update({"index": int(index), "slot_offset": slot_offset})
    return payload


def site_register() -> np.ndarray:
    """Return 15 semantic columns x 16 slot rows, ordered slot-major."""
    coordinates = np.zeros((N_SITES, 2), dtype=float)
    for slot in range(SLOT_ROWS):
        for semantic_row in range(SEMANTIC_COLUMNS):
            coordinates[atom_index(slot, semantic_row)] = (
                ORIGIN_UM[0]
                + semantic_row * SEMANTIC_PITCH_UM
                + (PARITY_OFFSET_UM if slot % 2 else 0.0),
                ORIGIN_UM[1] + slot * SLOT_PITCH_UM,
            )
    return coordinates


def unit_disk_edges(
    coordinates: np.ndarray | None = None,
    radius_um: float = BLOCKADE_RADIUS_UM,
) -> tuple[tuple[int, int], ...]:
    xy = site_register() if coordinates is None else np.asarray(coordinates, dtype=float)
    if xy.shape != (N_SITES, 2) or not np.all(np.isfinite(xy)):
        raise ValueError(f"coordinates must have shape ({N_SITES}, 2)")
    edges: list[tuple[int, int]] = []
    for left in range(N_SITES):
        distances = np.linalg.norm(xy[left + 1 :] - xy[left], axis=1)
        edges.extend(
            (left, left + 1 + int(offset))
            for offset in np.flatnonzero(distances <= float(radius_um) + 1e-10)
        )
    return tuple(edges)


def logical_edges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (atom_index(slot, left), atom_index(slot, right))
        for slot in range(WINDOW_SLOTS)
        for left, right in PAIR_ROWS
    )


def adjacency(
    edges: Iterable[tuple[int, int]] | None = None,
) -> tuple[frozenset[int], ...]:
    neighbours = [set() for _ in range(N_SITES)]
    for left, right in unit_disk_edges() if edges is None else edges:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))
    return tuple(frozenset(values) for values in neighbours)


def _binary_tuple(bits: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    raw = np.asarray(bits)
    if raw.shape != (N_SITES,):
        raise ValueError(f"sample must be a length-{N_SITES} binary vector")
    if not np.all(np.isfinite(raw)) or np.any((raw != 0) & (raw != 1)):
        raise ValueError("sample contains a value outside {0, 1}")
    return tuple(int(value) for value in raw.tolist())


def conflicting_edges(
    bits: Sequence[int] | np.ndarray,
    edges: Iterable[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int], ...]:
    values = _binary_tuple(bits)
    graph_edges = unit_disk_edges() if edges is None else tuple(edges)
    return tuple((left, right) for left, right in graph_edges if values[left] and values[right])


def is_independent(bits: Sequence[int] | np.ndarray) -> bool:
    return not conflicting_edges(bits)


def decode_independent_set(
    bits: Sequence[int] | np.ndarray,
    *,
    verify_independent: bool = True,
) -> DecodedTrajectory:
    """Decode every independent set; pair vacuum is an explicit base action."""
    values = _binary_tuple(bits)
    if verify_independent:
        conflicts = conflicting_edges(values)
        if conflicts:
            raise ValueError(
                f"sample is not a physical independent set ({len(conflicts)} conflicts)"
            )

    compute = [[0] * WINDOW_SLOTS for _ in range(3)]
    batteries = [["idle"] * WINDOW_SLOTS for _ in range(2)]
    batch = ["work_conserving"] * WINDOW_SLOTS
    guards = [[0] * WINDOW_SLOTS for _ in range(3)]
    for slot in range(WINDOW_SLOTS):
        for island, (medium_row, high_row) in enumerate(((0, 1), (2, 3), (4, 5))):
            medium = values[atom_index(slot, medium_row)]
            high = values[atom_index(slot, high_row)]
            if medium and high:
                raise ValueError("same-island medium/high double selection")
            compute[island][slot] = 2 if high else 1 if medium else 0
        for battery, (charge_row, discharge_row) in enumerate(((6, 7), (8, 9))):
            charge = values[atom_index(slot, charge_row)]
            discharge = values[atom_index(slot, discharge_row)]
            if charge and discharge:
                raise ValueError("same-battery charge/discharge double selection")
            batteries[battery][slot] = (
                "charge" if charge else "discharge" if discharge else "idle"
            )
        deadline = values[atom_index(slot, 10)]
        defer = values[atom_index(slot, 11)]
        if deadline and defer:
            raise ValueError("double-selected aggregate batch action")
        batch[slot] = "deadline_only" if deadline else "defer" if defer else "work_conserving"
        for guard in range(3):
            guards[guard][slot] = values[atom_index(slot, 12 + guard)]

    return DecodedTrajectory(
        bits=values,
        selected_sites=tuple(index for index, value in enumerate(values) if value),
        compute_levels=tuple(tuple(row) for row in compute),
        battery_actions=tuple(tuple(row) for row in batteries),
        batch_actions=tuple(batch),
        guard_bits=tuple(tuple(row) for row in guards),
    )


def edge_sha256(edges: Iterable[tuple[int, int]]) -> str:
    canonical = json.dumps(
        [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in edges)],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _component_sizes(edges: tuple[tuple[int, int], ...]) -> list[int]:
    neighbours = adjacency(edges)
    unseen = set(range(N_SITES))
    sizes: list[int] = []
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for other in neighbours[node]:
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def encoding_certificate(representative_q: np.ndarray | None = None) -> dict:
    """Build a self-contained, falsifiable geometry and support certificate."""
    xy = site_register()
    physical = unit_disk_edges(xy)
    logical = logical_edges()
    physical_set = set(physical)
    logical_set = set(logical)
    extra = tuple(sorted(physical_set - logical_set))
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    distances[distances == 0.0] = np.inf
    degrees = np.array([len(values) for values in adjacency(physical)], dtype=int)
    same_slot = sum(
        divmod(left, ROWS_PER_SLOT)[0] == divmod(right, ROWS_PER_SLOT)[0]
        for left, right in physical
    )
    cross_slot = len(physical) - same_slot
    complete_48_edges = 48 * 47 // 2
    logical_pair_distances = np.array(
        [np.linalg.norm(xy[left] - xy[right]) for left, right in logical],
        dtype=float,
    )
    minimum_pair_spacing = float(np.round(np.min(distances), 12))
    snapped = (
        np.rint(xy / POSITION_RESOLUTION_UM) * POSITION_RESOLUTION_UM
    )
    maximum_resolution_residual = float(np.max(np.abs(xy - snapped)))

    q_support: dict | None = None
    if representative_q is not None:
        q = np.asarray(representative_q, dtype=float)
        if q.shape != (N_SITES, N_SITES):
            raise ValueError("representative Q must be 240x240")
        if not np.allclose(q, q.T, rtol=0.0, atol=1e-12):
            raise ValueError("representative Q must be symmetric")
        support = {
            (left, right)
            for left in range(N_SITES)
            for right in range(left + 1, N_SITES)
            if abs(float(q[left, right])) > 0.0
        }
        q_support = {
            "shape": [N_SITES, N_SITES],
            "local_diagonal_nonzero_count": int(np.count_nonzero(np.diag(q))),
            "local_diagonal_nonzero_count_expected": N_WEIGHTED_SITES,
            "local_diagonal_nonzero_count_matches_weighted_sites": int(
                np.count_nonzero(np.diag(q))
            )
            == N_WEIGHTED_SITES,
            "pairwise_nonzero_count": len(support),
            "symmetric_offdiagonal_nonzero_count": int(
                np.count_nonzero(q - np.diag(np.diag(q)))
            ),
            "pairwise_support_equals_physical_udg_edges": support == physical_set,
            "pairwise_support_sha256": edge_sha256(support),
            "physical_edge_sha256": edge_sha256(physical),
        }

    return {
        "schema_version": "iteration5.route-c-plus.encoding-certificate.v2",
        "window": {
            "slots": WINDOW_SLOTS,
            "resolution_minutes": 15,
            "hours": 4,
        },
        "register": {
            "layout": (
                "transposed resolution-snapped triangular unit-disk lattice: "
                "15 semantic columns x 16 slot rows"
            ),
            "orientation": {
                "x_axis": "15 semantic columns (row indices 0..14)",
                "y_axis": "16 slot rows (slot offsets 0..15)",
                "parity_offset_axis": "x",
                "parity_offset_applied_to": "odd slot rows",
                "semantic_columns": SEMANTIC_COLUMNS,
                "slot_rows": SLOT_ROWS,
                "site_index_preserved": "atom_index(slot,row) = 15*slot + row",
            },
            "ordering": "slot-major; site = 15*slot_offset + row",
            "site_count": N_SITES,
            "meaningful_sites": N_SITES,
            "pad_sites": 0,
            "pad_policy": "none; all 240 sites have declared semantics",
            "weighted_sites": N_WEIGHTED_SITES,
            "zero_local_weight_guard_state_sites": N_GUARD_SITES,
            "operational_site_limit": OPERATIONAL_SITE_LIMIT,
            "aquila_site_limit": AQUILA_SITE_LIMIT,
            "within_operational_site_limit": N_SITES == OPERATIONAL_SITE_LIMIT,
            "within_aquila_site_limit": N_SITES <= AQUILA_SITE_LIMIT,
            "local_weighted_site_limit": LOCAL_WEIGHTED_SITE_LIMIT,
            "within_local_weighted_site_limit": N_WEIGHTED_SITES <= LOCAL_WEIGHTED_SITE_LIMIT,
            "coordinates_um": xy.tolist(),
            "minimum_pair_spacing_um": minimum_pair_spacing,
            "declared_minimum_spacing_um": MIN_SPACING_UM,
            "floating_point_spacing_tolerance_um": 1e-10,
            "minimum_radial_spacing_assumption_um": MIN_SPACING_UM,
            "minimum_radial_spacing_assumption_status": (
                "conservative offline local-detuning capability assumption; "
                "live device properties remain authoritative at preflight"
            ),
            "minimum_spacing_satisfied": bool(
                np.min(distances) >= MIN_SPACING_UM - 1e-10
            ),
            "nearest_neighbor_design": {
                "semantic_pitch_um": SEMANTIC_PITCH_UM,
                "slot_pitch_um": SLOT_PITCH_UM,
                "odd_slot_x_offset_um": PARITY_OFFSET_UM,
                "horizontal_same_slot_distance_um": SEMANTIC_PITCH_UM,
                "diagonal_adjacent_slot_distance_um": float(
                    np.hypot(PARITY_OFFSET_UM, SLOT_PITCH_UM)
                ),
            },
            "intended_same_slot_pair_distances_um": {
                "minimum": float(np.round(np.min(logical_pair_distances), 12)),
                "maximum": float(np.round(np.max(logical_pair_distances), 12)),
                "all_at_least_minimum_spacing": bool(
                    np.all(logical_pair_distances >= MIN_SPACING_UM - 1e-10)
                ),
                "all_within_model_blockade": bool(
                    np.all(
                        logical_pair_distances
                        <= BLOCKADE_RADIUS_UM + 1e-10
                    )
                ),
            },
            "coordinate_resolution_um": POSITION_RESOLUTION_UM,
            "maximum_coordinate_resolution_residual_um": (
                maximum_resolution_residual
            ),
            "coordinate_resolution_ready": bool(
                maximum_resolution_residual <= 1e-10
            ),
            "bounds_um": {
                "x": [float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))],
                "y": [float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))],
            },
            "width_um": float(np.ptp(xy[:, 0])),
            "height_um": float(np.ptp(xy[:, 1])),
            "aquila_area_um": list(AQUILA_AREA_UM),
            "fits_declared_aquila_area": bool(
                np.min(xy) >= 0.0
                and np.max(xy[:, 0]) <= AQUILA_AREA_UM[0]
                and np.max(xy[:, 1]) <= AQUILA_AREA_UM[1]
            ),
            "blockade_radius_um": BLOCKADE_RADIUS_UM,
            "blockade_radius_status": (
                "conservative operational UDG model parameter; not a live-device "
                "calibration and subject to live hardware-script preflight"
            ),
        },
        "semantics": {
            "rows": [asdict(item) for item in ROW_SEMANTICS],
            "weighted_rows_per_slot": 12,
            "guard_rows_per_slot": 3,
            "vacuum_decode": {
                "compute_pair": "low",
                "battery_pair": "idle",
                "batch_pair": "work_conserving",
                "interpretation": "absence is a complete action, never missing data",
            },
            "guard_claim_boundary": (
                "guard/state sites carry zero local weight and no mathematical "
                "queue, SOC, grid, deadline, or peak-feasibility proof"
            ),
        },
        "edges": {
            "logical_edge_count": len(logical),
            "physical_udg_edge_count": len(physical),
            "extra_geometric_edge_count": len(extra),
            "same_slot_physical_edge_count": int(same_slot),
            "cross_slot_physical_edge_count": int(cross_slot),
            "logical_subset_physical": logical_set <= physical_set,
            "missing_logical_edges": [list(edge) for edge in sorted(logical_set - physical_set)],
            "same_pair_double_selections_physically_blocked": logical_set <= physical_set,
            "logical_edge_sha256": edge_sha256(logical),
            "physical_edge_sha256": edge_sha256(physical),
            "extra_edge_sha256": edge_sha256(extra),
            "logical_edges": [list(edge) for edge in logical],
            "physical_udg_edges": [list(edge) for edge in physical],
            "extra_geometric_edges": [list(edge) for edge in extra],
            "extra_edge_interpretation": (
                "Every extra UDG edge is a disclosed conservative codebook "
                "restriction.  It is not asserted to encode an operational constraint."
            ),
        },
        "graph_nontriviality": {
            "edge_density": 2.0 * len(physical) / (N_SITES * (N_SITES - 1)),
            "minimum_degree": int(np.min(degrees)),
            "maximum_degree": int(np.max(degrees)),
            "mean_degree": float(np.mean(degrees)),
            "component_sizes": _component_sizes(physical),
            "connected": len(_component_sizes(physical)) == 1,
            "has_extra_geometric_structure": len(extra) > 0,
        },
        "legacy_48_bit_dimensional_comparison": {
            "new_dimension": N_SITES,
            "legacy_dimension": 48,
            "dimension_ratio": N_SITES / 48,
            "maximum_unique_pairs_in_any_48_variable_qubo": complete_48_edges,
            "new_unique_physical_pair_couplings": len(physical),
            "new_support_exceeds_complete_48_pair_count": len(physical) > complete_48_edges,
            "absolute_pair_count_increase_over_complete_48_bound": len(physical)
            - complete_48_edges,
            "claim_boundary": (
                "This is an absolute support-count comparison, not a claim that "
                "the 240-site graph has higher normalized density."
            ),
        },
        "qubo_support": q_support,
        "hardware_claim": {
            "submitted": False,
            "hardware_verified": False,
            "scope": (
                "Aquila-envelope encoding and UDG model only; no hardware counts, "
                "calibration, or sampling-advantage claim"
            ),
        },
    }
