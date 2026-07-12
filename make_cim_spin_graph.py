"""CIM spin-graph diagram for the 16-spin scheduling Ising instance.

Matches the vector style of make_qaoa_circuit.py. Title follows the QAOA
circuit caption pattern. Panel headers, the H_C formula line, and the bottom
notation strip are omitted by request.
"""
from __future__ import annotations

from paths import ROOT
_LOCAL_FIG = ROOT / 'figures'
_LOCAL_FIG.mkdir(parents=True, exist_ok=True)

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
OUTS = [_LOCAL_FIG]
for d in OUTS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "figure.dpi": 300,
})

GREEN = "#e7f2e4"
GREEN_E = "#4e8f4a"
BLUE = "#e8eef9"
BLUE_E = "#3c6dbf"
ORANGE = "#fdf0e2"
ORANGE_E = "#d97e2a"
GREY_E = "#777777"
WIRE = "#222222"
X_FC = "#dce8f8"
D_FC = "#fde8d8"


def spin_node(ax, xy, label, fc, ec, r=0.34):
    ax.add_patch(Circle(xy, r, facecolor=fc, edgecolor=ec, lw=1.15, zorder=5))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=7.8,
            zorder=6, color=WIRE)


def complete_edges(ax, pts, color, lw=0.7, alpha=0.45, z=2):
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                    color=color, lw=lw, alpha=alpha, zorder=z)


def field_arrow(ax, xy, strength, color, r_node=0.34):
    if abs(strength) < 1e-12:
        return
    sgn = np.sign(strength)
    y0 = xy[1] + sgn * (r_node + 0.06)
    y1 = xy[1] + sgn * (r_node + 0.38)
    ax.annotate(
        "",
        xy=(xy[0], y1),
        xytext=(xy[0], y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                        mutation_scale=8),
        zorder=7,
    )


def main():
    h_x = np.array([0.6, 0.6, -0.4, -0.2, -0.1, 0.2, -0.8, -0.5])
    h_d = np.array([-0.5, -0.5, 0.2, 0.5, 0.3, 0.1, 0.9, 0.7])

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    ax.set_xlim(0.0, 14.6)
    ax.set_ylim(0.85, 8.35)
    ax.axis("off")

    # Title in the same pattern as the QAOA circuit figure.
    ax.text(
        0.15, 8.05,
        r"CIM spin graph for the 16-spin scheduling block "
        r"(compute / discharge cliques; same $H_C$ as gate-model QAOA)",
        fontsize=11, fontweight="bold", va="center",
    )

    # Spin panels without in-box legends.
    box_y0, box_h = 4.05, 3.55
    ax.add_patch(Rectangle((0.35, box_y0), 6.55, box_h, facecolor=BLUE,
                           edgecolor=BLUE_E, lw=0.9, linestyle=(0, (4, 3)),
                           zorder=1))
    ax.add_patch(Rectangle((7.55, box_y0), 6.55, box_h, facecolor=ORANGE,
                           edgecolor=ORANGE_E, lw=0.9, linestyle=(0, (4, 3)),
                           zorder=1))
    ax.text(0.55, box_y0 + 0.18, "compute spin", ha="left", va="bottom",
            fontsize=8.2, color=BLUE_E, zorder=8)
    ax.text(7.75, box_y0 + 0.18, "discharge spin", ha="left", va="bottom",
            fontsize=8.2, color=ORANGE_E, zorder=8)

    ax.add_patch(Rectangle((0.35, 1.15), 13.75, 2.55, facecolor=GREEN,
                           edgecolor=GREEN_E, lw=0.9, linestyle=(0, (4, 3)),
                           zorder=1))
    ax.text(7.225, 3.55, "Coherent Ising machine mapping",
            ha="center", va="top", fontsize=8.8, color=GREEN_E,
            fontweight="bold", zorder=3)

    # Rings sit higher so bottom-right vertices clear the panel border and
    # corner labels.
    cx_x, cy_x, rx_x, ry_x = 3.55, 5.95, 2.35, 1.22
    cx_d, cy_d, rx_d, ry_d = 10.85, 5.95, 2.35, 1.22
    ang = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 8, endpoint=False)
    pts_x = [(cx_x + rx_x * np.cos(a), cy_x + ry_x * np.sin(a)) for a in ang]
    pts_d = [(cx_d + rx_d * np.cos(a), cy_d + ry_d * np.sin(a)) for a in ang]

    complete_edges(ax, pts_x, BLUE_E)
    complete_edges(ax, pts_d, ORANGE_E)

    ax.plot([pts_x[0][0], pts_x[3][0]], [pts_x[0][1], pts_x[3][1]],
            color=BLUE_E, lw=1.7, zorder=3)
    ax.text(0.5 * (pts_x[0][0] + pts_x[3][0]) - 0.18,
            0.5 * (pts_x[0][1] + pts_x[3][1]) + 0.22,
            r"$J_{x_1x_4}$", fontsize=7.4, color=BLUE_E, ha="center")
    ax.plot([pts_d[1][0], pts_d[6][0]], [pts_d[1][1], pts_d[6][1]],
            color=ORANGE_E, lw=1.7, zorder=3)
    ax.text(0.5 * (pts_d[1][0] + pts_d[6][0]) + 0.18,
            0.5 * (pts_d[1][1] + pts_d[6][1]) + 0.22,
            r"$J_{d_2d_7}$", fontsize=7.4, color=ORANGE_E, ha="center")

    for b, (px, pd) in enumerate(zip(pts_x, pts_d), start=1):
        spin_node(ax, px, rf"$s_{{x_{b}}}$", X_FC, BLUE_E)
        spin_node(ax, pd, rf"$s_{{d_{b}}}$", D_FC, ORANGE_E)
        # No field arrows on bottom-ring vertices (b=1,8): they press the
        # panel floor and the bottom-right corner.
        if b not in (1, 8):
            field_arrow(ax, px, h_x[b - 1], BLUE_E)
            field_arrow(ax, pd, h_d[b - 1], ORANGE_E)

    # Keep block tags off the bottom-right vertex.
    ax.text(pts_x[0][0] - 0.55, pts_x[0][1], "block 1", ha="right",
            va="center", fontsize=6.6, color=GREY_E)
    ax.text(pts_x[6][0] + 0.55, pts_x[6][1], "block 7", ha="left",
            va="center", fontsize=6.6, color=GREY_E)
    ax.text(pts_d[6][0] + 0.55, pts_d[6][1] + 0.15, "evening peak",
            ha="left", va="bottom", fontsize=6.6, color=GREY_E)

    y_opo = 2.15
    xs = list(np.linspace(1.05, 4.55, 7)) + [5.35]
    for k, x in enumerate(xs):
        if k == 7:
            ax.text(x - 0.35, y_opo, r"$\cdots$", ha="center", va="center",
                    fontsize=12, color=GREEN_E, zorder=5)
            ax.add_patch(Ellipse((x + 0.35, y_opo), 0.36, 0.52,
                                 facecolor="white", edgecolor=GREEN_E, lw=1.0,
                                 zorder=4))
            ax.plot([x + 0.35, x + 0.35], [y_opo - 0.16, y_opo + 0.16],
                    color=GREEN_E, lw=1.0, zorder=5)
            ax.text(x + 0.35, y_opo - 0.48, r"$p_{16}$", ha="center",
                    fontsize=7.0, color=GREEN_E)
            continue
        ax.add_patch(Ellipse((x, y_opo), 0.36, 0.52,
                             facecolor="white", edgecolor=GREEN_E, lw=1.0,
                             zorder=4))
        ax.plot([x, x], [y_opo - 0.16, y_opo + 0.16], color=GREEN_E, lw=1.0,
                zorder=5)
        if k in (0, 6):
            ax.text(x, y_opo - 0.48, rf"$p_{{{k+1}}}$", ha="center",
                    fontsize=7.0, color=GREEN_E)
    ax.text(3.15, 3.05, r"OPO pulse train  (16 pulses $\leftrightarrow$ 16 spins)",
            ha="center", fontsize=8.0, color=GREEN_E)

    ax.annotate(
        "",
        xy=(8.35, y_opo),
        xytext=(5.85, y_opo),
        arrowprops=dict(arrowstyle="-|>", color=WIRE, lw=1.2,
                        mutation_scale=11),
    )
    ax.text(7.1, y_opo + 0.42, "measurement\nfeedback $J_{ij}$",
            ha="center", va="bottom", fontsize=7.6, color=WIRE)

    ax.add_patch(FancyBboxPatch((8.55, 1.55), 4.70, 1.45,
                                boxstyle="round,pad=0.06",
                                facecolor="#f7fbf6", edgecolor=GREEN_E,
                                lw=1.15, zorder=4))
    ax.text(10.90, 2.72,
            r"Ising readout:  $s_i=\mathrm{sign}(c_i)$",
            ha="center", va="center", fontsize=8.4, color=GREEN_E,
            fontweight="bold", zorder=6)
    ax.text(10.90, 2.12,
            "decode $(x_b,d_b)$ to block schedule\n"
            r"same $H_C$ as gate-model QAOA ($s_i=1-2x_i$)",
            ha="center", va="center", fontsize=7.6, color=WIRE, zorder=6)

    name = "fig05b_cim_spin_graph.png"
    primary = OUTS[0] / name
    fig.savefig(primary, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for dest in OUTS[1:]:
        shutil.copy2(primary, dest / name)
    print(f"Wrote {primary}")


if __name__ == "__main__":
    main()
