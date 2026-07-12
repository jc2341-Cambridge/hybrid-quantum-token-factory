"""Path helpers for the reproducible code package.

All analysis scripts load inputs from ``data/`` relative to this package
root and write regenerated artefacts to ``output/``.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

MARKET = DATA / "01_market"
FACILITY = DATA / "02_facility"
MILP_RESULTS = DATA / "03_milp_results"
QUANTUM = DATA / "04_quantum"
SENSITIVITY = DATA / "05_sensitivity"

JULY_PROFILES_CSV = MARKET / "east_china_july_profiles.csv"
SEASONAL_PV_CSV = MARKET / "seasonal_pv_profiles.csv"
REDUCED_QUBO_CSV = QUANTUM / "reduced_qubo_matrix.csv"
REDUCED_SPECTRUM_CSV = QUANTUM / "reduced_spectrum.csv"
QAOA_DEPTH_CSV = QUANTUM / "qaoa_depth_sweep.csv"


def ensure_output() -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    return OUTPUT


def require_csv(path: Path, hint: str = "") -> Path:
    """Raise a clear error if a required user-supplied CSV is missing."""
    if not path.exists():
        msg = (
            f"Missing required data file:\n  {path}\n"
            "Place the CSV under data/ following the column schema in README.md."
        )
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
    return path
