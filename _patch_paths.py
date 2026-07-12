"""Patch copied scripts to use local data/ and output/ paths."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def patch_text(path: Path, replacers: list[tuple[str, str]]) -> None:
    t = path.read_text(encoding="utf-8")
    for a, b in replacers:
        t = t.replace(a, b)
    path.write_text(t, encoding="utf-8")
    print("patched", path.name)


for name in ["run_milp_study.py", "quantum_edge_study.py", "surrogate_fidelity.py"]:
    p = ROOT / name
    t = p.read_text(encoding="utf-8")
    t = t.replace("output_revised", "output")
    if "from paths import" not in t:
        t = t.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\nfrom paths import ensure_output\n",
        )
    t = t.replace(
        'HERE = Path(__file__).parent\nOUT = HERE / "output"\nOUT.mkdir(parents=True, exist_ok=True)',
        "HERE = Path(__file__).parent\nOUT = ensure_output()",
    )
    t = t.replace(
        "HERE = Path(__file__).parent\nOUT = HERE / 'output'\nOUT.mkdir(parents=True, exist_ok=True)",
        "HERE = Path(__file__).parent\nOUT = ensure_output()",
    )
    # run_milp_study style
    t = t.replace(
        'OUT = Path(__file__).parent / "output"\nOUT.mkdir(exist_ok=True)',
        "from paths import ensure_output\nOUT = ensure_output()",
    )
    p.write_text(t, encoding="utf-8")
    print("patched", name)

p = ROOT / "make_figures.py"
t = p.read_text(encoding="utf-8")
t = t.replace("output_revised", "output")
t = t.replace(
    'FIG = HERE.parent / "latex" / "REVISED-FIGURES"',
    'FIG = HERE / "figures"',
)
if "from paths import" not in t:
    t = t.replace(
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom paths import ensure_output\n",
    )
t = t.replace(
    'OUT = HERE / "output"\n',
    "OUT = ensure_output()\n",
)
p.write_text(t, encoding="utf-8")
print("patched make_figures.py")

# CIM / QAOA circuit writers: redirect figure outputs into ./figures
for name in ["make_cim_spin_graph.py", "make_qaoa_circuit.py"]:
    p = ROOT / name
    t = p.read_text(encoding="utf-8")
    # common patterns writing to latex folders
    t2 = t
    # prepend local FIG dir if script has hardcoded paths
    if "FINAL-FIGURES" in t or "real-data-figures" in t or "latex" in t:
        header = (
            "from paths import ROOT\n"
            "_LOCAL_FIG = ROOT / 'figures'\n"
            "_LOCAL_FIG.mkdir(parents=True, exist_ok=True)\n"
        )
        if "from paths import ROOT" not in t2:
            # insert after imports block start
            t2 = t2.replace(
                "from __future__ import annotations\n",
                "from __future__ import annotations\n\n" + header,
                1,
            )
        t2 = t2.replace(
            'Path(__file__).resolve().parent.parent / "latex" / "FINAL-FIGURES"',
            "_LOCAL_FIG",
        )
        t2 = t2.replace(
            "Path(__file__).resolve().parent.parent / 'latex' / 'FINAL-FIGURES'",
            "_LOCAL_FIG",
        )
        # blunt: any remaining latex figure dirs -> local
        import re
        t2 = re.sub(
            r'Path\([^)]*\)\s*/\s*["\']latex["\']\s*/\s*["\'][^"\']+["\']',
            "_LOCAL_FIG",
            t2,
        )
    p.write_text(t2, encoding="utf-8")
    print("patched", name)
