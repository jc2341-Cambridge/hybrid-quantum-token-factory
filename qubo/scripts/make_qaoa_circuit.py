"""Gate-level QAOA circuit diagram for the 16-qubit storage master.

Strictly reflects the hybrid master of the paper:
  * 16 qubits = 8 charge bits c_b (q0..q7) + 8 discharge bits d_b (q8..q15);
  * state preparation |+>^16 by Hadamards;
  * cost layer exp(-i gamma_l H_C): RZ(2 gamma_l h_i) on every qubit plus
    ZZ couplings; representative charge-pair and discharge-pair drawn;
  * mixer layer exp(-i beta_l H_M): RX(2 beta_l) on every qubit;
  * layers repeated l = 1..p; Pauli-Z measurement decodes to (c,d) masks.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arc

HERE = Path(__file__).parent
FIG = Path(os.environ.get("CIRCUIT_FIG_DIR",
                          str(HERE.parent / "latex" / "FINAL-FIGURES")))
FIG.mkdir(parents=True, exist_ok=True)

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
GREY = "#eeeeee"
GREY_E = "#777777"
WIRE = "#222222"

if os.environ.get("CIRCUIT_PALETTE") == "plasma":
    GREEN, GREEN_E = "#e9e8f8", "#0d0887"
    BLUE, BLUE_E = "#f5e5f3", "#9c179e"
    ORANGE, ORANGE_E = "#fdeedd", "#e16462"

Y = {0: 6.4, 1: 5.5, 2: 4.6, 3: 3.7, 14: 2.2, 15: 1.3}
Y_DOTS = 2.95
X0, X1 = 1.05, 13.9

X_H = 1.75
X_RZ = 3.05
X_C1, X_RJ, X_C2 = 4.35, 5.15, 5.95
X_C3, X_RJ2, X_C4 = 6.75, 7.55, 8.35
X_DOTS = 8.95
X_RX = 9.85
X_REP0, X_REP1 = 10.65, 12.05
X_M = 12.95

BW, BH = 0.62, 0.46


def gate(ax, x, y, text, fc="white", ec=WIRE, fs=8.2, w=BW, h=BH):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.055",
                                facecolor=fc, edgecolor=ec, lw=1.0, zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=5)


def cnot(ax, x, y_ctrl, y_targ):
    ax.plot([x, x], [y_ctrl, y_targ], color=WIRE, lw=1.1, zorder=3)
    ax.add_patch(Circle((x, y_ctrl), 0.075, facecolor=WIRE,
                        edgecolor=WIRE, zorder=4))
    ax.add_patch(Circle((x, y_targ), 0.155, facecolor="white",
                        edgecolor=WIRE, lw=1.1, zorder=4))
    ax.plot([x - 0.155, x + 0.155], [y_targ, y_targ], color=WIRE, lw=1.1,
            zorder=5)
    ax.plot([x, x], [y_targ - 0.155, y_targ + 0.155], color=WIRE, lw=1.1,
            zorder=5)


def meter(ax, x, y):
    w, h = 0.66, 0.5
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.055",
                                facecolor="white", edgecolor=WIRE, lw=1.0,
                                zorder=4))
    ax.add_patch(Arc((x, y - 0.10), 0.42, 0.40, theta1=15, theta2=165,
                     color=WIRE, lw=1.0, zorder=5))
    ax.plot([x, x + 0.16], [y - 0.10, y + 0.14], color=WIRE, lw=1.0, zorder=5)


def region(ax, x_lo, x_hi, fc, ec, label, label_color, y_top_extra=0.0):
    y_lo, y_hi = 0.75, 7.0
    ax.add_patch(Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                           facecolor=fc, edgecolor=ec, lw=0.9,
                           linestyle=(0, (4, 3)), zorder=1))
    ax.text((x_lo + x_hi) / 2, y_hi + 0.28 + y_top_extra, label,
            ha="center", va="bottom", fontsize=8.8, color=label_color,
            fontweight="bold")


fig, ax = plt.subplots(figsize=(9.6, 7.9))
ax.set_xlim(0, 14.6)
ax.set_ylim(-3.15, 8.7)
ax.axis("off")

ax.text(0.05, 8.4,
        "QAOA circuit for the 16-qubit storage master "
        "(one of $p$ layers shown; $2p$ parameters "
        "$\\gamma_1,\\beta_1,\\ldots,\\gamma_p,\\beta_p$)",
        fontsize=11, fontweight="bold", va="center")

region(ax, X_H - 0.55, X_H + 0.55, GREEN, GREEN_E,
       "Prepare $|+\\rangle^{\\otimes 16}$", GREEN_E)
region(ax, X_RZ - 0.65, X_DOTS + 0.45, BLUE, BLUE_E,
       "Cost layer  $e^{-i\\gamma_1 H_C}$", BLUE_E)
region(ax, X_RX - 0.6, X_RX + 0.6, ORANGE, ORANGE_E,
       "Mixer\n$e^{-i\\beta_1 H_M}$", ORANGE_E)
region(ax, X_REP0 - 0.35, X_REP1 + 0.35, "#f7f7f7", GREY_E,
       "Layers\n$l = 2,\\ldots,p$", GREY_E)
region(ax, X_M - 0.6, X_M + 0.6, GREY, GREY_E, "Measure\n($Z$ basis)",
       GREY_E)

labels = {
    0: "$q_0$: $c_1$", 1: "$q_1$: $c_2$", 2: "$q_2$: $c_3$",
    3: "$q_3$: $c_4$", 14: "$q_{14}$: $d_7$", 15: "$q_{15}$: $d_8$",
}
for q, y in Y.items():
    ax.plot([X0, X1], [y, y], color=WIRE, lw=1.0, zorder=2)
    ax.text(X0 - 0.12, y, f"$|0\\rangle$  {labels[q]}", ha="right",
            va="center", fontsize=9)
ax.text((X0 + X1) / 2 - 6.0, Y_DOTS, "$\\vdots$", ha="center", va="center",
        fontsize=12)
ax.text(X0 - 0.55, Y_DOTS, "$\\vdots$", ha="center", va="center",
        fontsize=12)

for q, y in Y.items():
    gate(ax, X_H, y, "H", fc="white", ec=GREEN_E)
    gate(ax, X_RZ, y, "RZ\n$(2\\gamma_1 h_i)$", fc="white", ec=BLUE_E,
         fs=6.8, w=0.92, h=0.64)
    gate(ax, X_RX, y, "RX\n$(2\\beta_1)$", fc="white", ec=ORANGE_E,
         fs=6.8, w=0.84, h=0.64)
    meter(ax, X_M, y)

cnot(ax, X_C1, Y[0], Y[1])
gate(ax, X_RJ, Y[1], "RZ\n$(2\\gamma_1 J_{01})$", fc="white", ec=BLUE_E,
     fs=6.4, w=0.98, h=0.64)
cnot(ax, X_C2, Y[0], Y[1])

cnot(ax, X_C3, Y[14], Y[15])
gate(ax, X_RJ2, Y[15], "RZ\n$(2\\gamma_1 J_{14,15})$", fc="white", ec=BLUE_E,
     fs=6.0, w=1.10, h=0.64)
cnot(ax, X_C4, Y[14], Y[15])

ax.text(X_DOTS, (Y[3] + Y[14]) / 2 + 0.4, "$\\cdots$", ha="center",
        va="center", fontsize=14)

gate(ax, 11.00, (Y[0] + Y[15]) / 2, "$e^{-i\\gamma_l H_C}$",
     fc=BLUE, ec=BLUE_E, fs=8.2, w=0.78, h=5.0)
gate(ax, 11.92, (Y[0] + Y[15]) / 2, "$e^{-i\\beta_l H_M}$",
     fc=ORANGE, ec=ORANGE_E, fs=8.2, w=0.78, h=5.0)

ax.add_patch(FancyBboxPatch((0.95, -3.0), 12.95, 2.85,
                            boxstyle="round,pad=0.08",
                            facecolor="white", edgecolor="#999999", lw=0.8,
                            zorder=1))
col1 = (
    "Notation\n"
    "H — Hadamard, prepares $|+\\rangle$\n"
    "RZ$(2\\gamma_l h_i)$ — QUBO linear term\n"
    "CNOT$\\cdot$RZ$(2\\gamma_l J_{ij})\\cdot$CNOT — ZZ coupling\n"
    "RX$(2\\beta_l)$ — transverse-field mixer\n"
    "meter — Pauli-$Z$ readout"
)
col2 = (
    "Wires (16 qubits = 8 blocks)\n"
    "$q_0\\ldots q_7$ — charge bits $c_b$\n"
    "$q_8\\ldots q_{15}$ — discharge bits $d_b$\n"
    "one ZZ pair per group drawn;\n"
    "full cost layer also includes\n"
    "charge--discharge balance couplings"
)
col3 = (
    "Parameters and readout\n"
    "$2p$ trainable angles $(\\gamma_l, \\beta_l)$\n"
    "problem data enter only through\n"
    "    the fixed $h_i$ and $J_{ij}$\n"
    "output bitstring = storage masks;\n"
    "classical MILP closes the schedule"
)
for x_col, txt in [(1.25, col1), (5.55, col2), (10.05, col3)]:
    ax.text(x_col, -0.38, txt, ha="left", va="top", fontsize=8,
            linespacing=1.55, zorder=2)

out = FIG / "fig05_qaoa_circuit.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
for dest in (
    HERE.parent / "latex" / "figures",
    HERE.parent / "latex" / "real-data-figures",
    HERE.parent / "latex" / "Plasma-style",
    HERE.parent / "latex" / "FINAL-FIGURES",
):
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "fig05_qaoa_circuit.png"
    if target.resolve() != out.resolve():
        shutil.copy2(out, target)
print(f"written to {out}")
