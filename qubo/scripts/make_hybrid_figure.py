"""Plasma-style hybrid decomposition figure (fig13).

Matches revised Plasma suite conventions: serif type, panel() titles,
figsize ~ (7.0 x 3.0), frameon=False legends, no top/right spines.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_revised"
FIG = HERE.parent / "latex" / "Plasma-style"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "figure.dpi": 300,
})

pl = plt.get_cmap("plasma")
C = {
    "quantum": to_hex(pl(0.02)),
    "classic": to_hex(pl(0.22)),
    "green": to_hex(pl(0.45)),
    "red": to_hex(pl(0.62)),
    "orange": to_hex(pl(0.82)),
    "base": "#8c8c8c",
}


def panel(ax, title: str) -> None:
    ax.set_title(title, fontsize=9, loc="left", pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load():
    with open(OUT / "hybrid_decomposition_report.json", encoding="utf-8") as f:
        return json.load(f)


def fig_hybrid_decomposition():
    rep = _load()
    base = rep["baselines"]
    free = base["free_milp"]["total_cost_usd"]
    greedy = base["greedy_top3_dis_only"]["total_cost_usd"]

    rows = [
        ("Free MILP", free, C["classic"]),
        ("Top-3 tariff / old QUBO", greedy, C["base"]),
        ("Path B (discharge only)", rep["path_b_dis_only"]["incumbent_cost"],
         C["orange"]),
        ("Path B (charge+discharge)", rep["path_b_both"]["incumbent_cost"],
         C["red"]),
        ("Path C (charge+discharge)", rep["path_c_enum_both"]["incumbent_cost"],
         C["quantum"]),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.32},
    )

    # (a) cost bars — no vertical guide line
    ax = axes[0]
    names = [r[0] for r in rows]
    costs = np.asarray([r[1] for r in rows], dtype=float)
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, costs, color=colors, height=0.58, edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.2)
    ax.set_xlabel("Daily operating cost (\\$)")
    ax.set_xlim(2818, 2885)
    for yi, c in zip(y, costs):
        gap = c - free
        label = f"{c:.0f}" if abs(gap) < 0.05 else f"{c:.0f} ({gap:+.1f})"
        ax.text(c + 1.2, yi, label, va="center", fontsize=6.8, color="#333333")
    panel(ax, "(a) Cost versus free MILP")

    # (b) Path B vs Path C incumbent when both C/D fixed
    ax = axes[1]
    b_feas = [it for it in rep["path_b_both"]["iterations"] if it["feasible"]]
    c_feas = [it for it in rep["path_c_enum_both"]["iterations"] if it["feasible"]]
    c_x, c_y, best = [], [], np.inf
    for it in c_feas:
        best = min(best, it["total_cost_usd"])
        c_x.append(it["iteration"])
        c_y.append(best)

    ax.plot(c_x, c_y, color=C["quantum"], lw=1.5, marker="o", ms=3.2,
            label="Path C incumbent")
    if b_feas:
        ax.scatter(
            [b_feas[0]["iteration"]], [b_feas[0]["total_cost_usd"]],
            color=C["red"], s=36, zorder=4, label="Path B first feasible",
        )
    ax.axhline(free, color=C["classic"], ls=":", lw=1.0, label="Free MILP")
    ax.axhline(greedy, color=C["base"], ls="--", lw=0.9, label="Top-3 greedy")
    ax.set_xlabel("Master proposals evaluated")
    ax.set_ylabel("Incumbent daily cost (\\$)")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    panel(ax, "(b) Path B versus Path C")

    fig.tight_layout()
    out = FIG / "fig13_hybrid_decomposition.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for dest in (
        HERE.parent / "latex" / "real-data-figures",
        HERE.parent / "latex" / "figures",
        HERE.parent / "latex" / "FINAL-FIGURES",
    ):
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, dest / out.name)
    print("wrote", out)


if __name__ == "__main__":
    fig_hybrid_decomposition()
