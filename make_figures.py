"""Figures for the workload-conserving, SOC-feasible revision.

Only figures whose data change under the revised formulation are rendered
here.  Unaffected model/QAOA figures remain available in FINAL-FIGURES.
"""
from __future__ import annotations

from pathlib import Path
from paths import ensure_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model as m
from revised_dispatch import baseline_metrics, solve_revised_dispatch


HERE = Path(__file__).parent
OUT = ensure_output()
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})
C = {
    "base": "#8c8c8c",
    "quantum": "#3f007d",
    "classic": "#1f78b4",
    "green": "#33a02c",
    "red": "#e31a1c",
    "orange": "#ff7f00",
}
# Overridable by make_revised_figures_plasma.py
CMAP_HOUR = "twilight_shifted"
CMAP_LEVELS = "Purples"
BOX_FC = "#f0eef7"
H = np.arange(m.T_SLOTS)


def panel(ax, title):
    ax.set_title(title, fontsize=9)


def fig01_framework():
    """Top-to-bottom method framework (8 boxes)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    boxes = [
        ("1  Real East China inputs",
         "Jiangsu TOU price  ·  MEE carbon factor\n"
         "PVGIS PV  ·  Open-Meteo temperature"),
        ("2  Workload model",
         "Interactive same-slot service\n"
         "Batch backlog, 6 h deadline, zero terminal backlog"),
        ("3  Facility energy model",
         "Discrete compute levels  ·  PV netting\n"
         "Battery SOC, efficiency, degradation"),
        ("4  Operational MILP",
         "Workload-conserving schedule\n"
         "Absolute cost and emissions objectives"),
        ("5  Block QUBO / Ising encoding",
         "8 three-hour blocks  ·  16 spins\n"
         "Compute and discharge equality budgets"),
        ("6  Quantum solvers",
         "CIM execution of $H_C$\n"
         "QAOA statevector reference on the same surrogate"),
        ("7  Fidelity and hybrid transfer",
         "Discharge-skeleton agreement audit\n"
         "Impose QUBO discharge blocks on full MILP"),
        ("8  Audited outputs",
         "Equal useful work  ·  SOC / backlog certificates\n"
         "Cost, evening import, supported trade-off"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(boxes)
    top, bottom = 0.965, 0.045
    box_h = 0.088
    gap = (top - bottom - n * box_h) / (n - 1)
    x0, width = 0.10, 0.80

    centers = []
    for i, (title, body) in enumerate(boxes):
        y1 = top - i * (box_h + gap)
        y0 = y1 - box_h
        centers.append(0.5 * (y0 + y1))
        ax.add_patch(FancyBboxPatch(
            (x0, y0), width, box_h,
            boxstyle="round,pad=0.012",
            facecolor=BOX_FC, edgecolor=C["quantum"], lw=1.25, zorder=2))
        ax.text(0.50, y1 - 0.022, title, ha="center", va="top",
                fontsize=8.2, fontweight="bold", color=C["quantum"], zorder=3)
        ax.text(0.50, y0 + 0.028, body, ha="center", va="bottom",
                fontsize=7.2, color="#222222", linespacing=1.35, zorder=3)

    for i in range(n - 1):
        y_top = centers[i] - box_h / 2 + 0.004
        y_bot = centers[i + 1] + box_h / 2 - 0.004
        ax.add_patch(FancyArrowPatch(
            (0.50, y_top), (0.50, y_bot),
            arrowstyle="-|>", mutation_scale=11,
            lw=1.25, color=C["quantum"], zorder=1))

    fig.savefig(FIG / "fig01_framework.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig03_context_profiles():
    """East China case-study context profiles."""
    price, carbon, _, _ = m.profiles()
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.2))
    ax = axes[0, 0]
    ax.step(H, price, where="mid", color=C["red"], lw=1.4)
    ax.set_ylabel("Price (\\$/kWh)")
    ax.set_xlabel("Hour of day")
    panel(ax, "(a) Electricity price")

    ax = axes[0, 1]
    ax.step(H, carbon, where="mid", color=C["base"], lw=1.4)
    ax.set_ylabel("CI (gCO$_2$/kWh)")
    ax.set_xlabel("Hour of day")
    panel(ax, "(b) Grid carbon intensity")

    ax = axes[1, 0]
    colors = {"Jan": C["classic"], "Apr": C["green"],
              "Jul": C["red"], "Oct": C["orange"]}
    for name, month in [("Jan", 1), ("Apr", 4), ("Jul", 7), ("Oct", 10)]:
        ax.plot(H, m.pv_profile_kw(month), color=colors[name],
                lw=1.6 if month == 7 else 1.2, label=name)
    ax.set_ylabel("PV (kW)")
    ax.set_xlabel("Hour of day")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    panel(ax, "(c) PVGIS regional mean-day profiles")

    ax = axes[1, 1]
    sc = ax.scatter(carbon, price, c=H, cmap=CMAP_HOUR, s=22,
                    edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, fraction=0.046, label="Hour")
    ax.set_xlabel("CI (gCO$_2$/kWh)")
    ax.set_ylabel("Price (\\$/kWh)")
    panel(ax, "(d) Price-carbon relation")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_context_profiles.png", bbox_inches="tight")
    plt.close(fig)


def fig04_reduced_qubo():
    """Quantum block surrogate only; no claim that it is the full SOC model."""
    from paths import REDUCED_QUBO_CSV, REDUCED_SPECTRUM_CSV
    qr_path = HERE / "output" / "reduced_qubo_matrix.npy"
    spec_path = HERE / "output" / "reduced_spectrum.npy"
    if qr_path.exists() and spec_path.exists():
        qr = np.load(qr_path)
        spec = np.load(spec_path)
    else:
        qr = pd.read_csv(REDUCED_QUBO_CSV, index_col=0).to_numpy(dtype=float)
        spec = pd.read_csv(REDUCED_SPECTRUM_CSV)["energy"].to_numpy(dtype=float)
    n = 16
    states = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    feasible = ((states[:, 0::2].sum(axis=1) == 5)
                & (states[:, 1::2].sum(axis=1) == 3))

    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.7))
    ax = axes[0]
    mat = qr + qr.T - np.diag(np.diag(qr))
    vmax = np.abs(mat).max()
    im = ax.imshow(mat, cmap="PuOr", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xlabel("Qubit index")
    ax.set_ylabel("Qubit index")
    plt.colorbar(im, ax=ax, fraction=0.046)
    panel(ax, "(a) 16-qubit block QUBO")

    ax = axes[1]
    shifted = spec - spec.min()
    bins = np.linspace(0, 2400, 60)
    ax.hist(shifted[~feasible], bins=bins, color=C["base"], alpha=0.7,
            label="Constraint-violating")
    ax.hist(shifted[feasible], bins=bins, color=C["green"], alpha=0.85,
            label="Feasible")
    ax.set_yscale("log")
    ax.set_xlabel("Energy above optimum\n(\\$-equivalent)")
    ax.set_ylabel("States (log)")
    ax.legend(frameon=False, fontsize=6.5)
    panel(ax, "(b) Feasibility separation")

    fig.tight_layout()
    fig.savefig(FIG / "fig04_qubo_encoding.png", bbox_inches="tight")
    plt.close(fig)


def fig08_dispatch():
    """Restore the six-panel dispatch anatomy used in the Plasma suite.

    Panel objects match the pre-review figure (tariff, level heatmap,
    throughput, site power balance, hourly cost, cumulative curves).
    Labels and series use the revised equal-workload schedule.
    """
    df = pd.read_csv(OUT / "hourly_dispatch.csv")
    price = df["price_usd_kwh"].to_numpy()
    carbon = df["carbon_g_kwh"].to_numpy()
    pv = df["pv_kw"].to_numpy()
    temp = df["temperature_c"].to_numpy()
    charge = df["charge"].to_numpy().astype(bool)
    discharge = df["discharge"].to_numpy().astype(bool)
    grid_b = df["grid_BAU_kw"].to_numpy()
    grid_r = df["grid_revised_kw"].to_numpy()
    levels_b = df["level_BAU"].to_numpy(dtype=int)
    levels_r = df["level_revised"].to_numpy(dtype=int)
    compute_only = solve_revised_dispatch(no_battery=True)
    levels_c = compute_only["levels"].astype(int)

    tok_b = m.throughput_mtok_h(m.U_LEVELS[levels_b])
    tok_r = m.throughput_mtok_h(m.U_LEVELS[levels_r])
    load_r = np.array([
        float(m.facility_power_kw(m.U_LEVELS[levels_r[t]], temp[t]))
        for t in range(m.T_SLOTS)
    ])
    demand = load_r + m.P_CH_KW * charge
    dis = m.P_DIS_KW * discharge
    pv_used = np.minimum(pv, np.maximum(demand - dis, 0.0))
    grid_s = np.maximum(demand - dis - pv_used, 0.0)
    tariff = m.effective_tariff(price, carbon)
    cost_b = price * grid_b * m.DT_H
    cost_r = price * grid_r * m.DT_H

    fig, axes = plt.subplots(6, 1, figsize=(7.0, 8.4), sharex=True)
    fig.subplots_adjust(hspace=0.38)

    ax = axes[0]
    ax.step(H, tariff, where="mid", color=C["quantum"], lw=1.4)
    ax.axvspan(16.5, 21.5, color=C["red"], alpha=0.08)
    ax.annotate("evening peak", xy=(19, tariff.max() * 0.55), ha="center",
                fontsize=7.5, color=C["red"])
    ax.set_ylabel("$\\tilde{\\pi}_t$\n(\\$/kWh)")
    panel(ax, "(a) Effective tariff (price + carbon)")

    ax = axes[1]
    mat = np.vstack([levels_b, levels_c, levels_r])
    ax.imshow(mat, aspect="auto", cmap=CMAP_LEVELS, vmin=0, vmax=2,
              extent=[-0.5, 23.5, -0.5, 2.5], origin="lower",
              interpolation="nearest")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["BAU", "Compute only", "Joint"], fontsize=7.5)
    panel(ax, "(b) Compute level (light = low, dark = high)")

    ax = axes[2]
    ax.step(H, tok_b, where="mid", color=C["base"], lw=1.4,
            label="BAU (demand-tracking)")
    ax.step(H, tok_r, where="mid", color=C["quantum"], lw=1.4,
            label="Revised schedule")
    ax.set_ylabel("Mtok/h")
    ax.set_ylim(15, 200)
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2)
    panel(ax, "(c) Token throughput (equal daily useful work)")

    ax = axes[3]
    ax.bar(H, grid_s, width=0.8, color=C["base"], label="Grid import")
    ax.bar(H, pv_used, width=0.8, bottom=grid_s, color=C["orange"],
           label="PV")
    ax.bar(H, dis, width=0.8, bottom=grid_s + pv_used, color=C["red"],
           label="Battery discharge")
    ax.bar(H, -m.P_CH_KW * charge, width=0.8, color=C["classic"],
           label="Battery charging")
    ax.step(H, demand, where="mid", color="k", lw=1.3,
            label="Load + charging")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("kW")
    ax.set_ylim(-m.P_CH_KW * 1.5, max(demand.max(), grid_s.max()) * 1.42)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=3)
    panel(ax, "(d) Site power balance (revised schedule)")

    ax = axes[4]
    w = 0.4
    ax.bar(H - w / 2, cost_b, width=w, color=C["base"], label="BAU")
    ax.bar(H + w / 2, cost_r, width=w, color=C["quantum"],
           label="Revised")
    ax.set_ylabel("\\$/h")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    panel(ax, "(e) Hourly electricity cost")

    ax = axes[5]
    ax.plot(H, np.cumsum(cost_b), color=C["base"], lw=1.5, label="Cost, BAU")
    ax.plot(H, np.cumsum(cost_r), color=C["quantum"], lw=1.5,
            label="Cost, revised")
    ax.set_ylabel("Cumulative \\$")
    ax2 = ax.twinx()
    ax2.plot(H, np.cumsum(tok_b), color=C["base"], lw=1.1, ls=":")
    ax2.plot(H, np.cumsum(tok_r), color=C["quantum"], lw=1.1, ls=":")
    ax2.set_ylabel("Cumulative Mtok", fontsize=8)
    # Keep the equal-work note in the lower white space, below all curves.
    tok_end = float(np.cumsum(tok_r)[-1])
    ax2.set_ylim(0.0, tok_end * 1.18)
    ax2.text(
        0.70, 0.12,
        "equal useful work\n3024 Mtok",
        transform=ax2.transAxes,
        ha="center", va="bottom", fontsize=7, color="#555555",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlabel("Hour of day")
    panel(ax, "(f) Cumulative cost (solid) and tokens (dotted)")
    ax.set_xlim(-0.5, 23.5)
    fig.savefig(FIG / "fig08_dispatch_anatomy.png", bbox_inches="tight")
    plt.close(fig)


def fig09_feasibility():
    """Single-row audits: backlog trajectory + deadline sensitivity."""
    df = pd.read_csv(OUT / "hourly_dispatch.csv")
    delay = pd.read_csv(OUT / "deadline_sensitivity.csv")
    backlog = df["batch_backlog_Mtok"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    ax.step(H, backlog, where="mid", color=C["orange"], lw=1.5)
    ax.fill_between(H, 0, backlog, step="mid", color=C["orange"], alpha=0.22)
    ax.axhline(0, color="k", lw=0.7)
    ax.annotate(f"peak {backlog.max():.0f} Mtok",
                xy=(float(np.argmax(backlog)), backlog.max()),
                xytext=(8, 8), textcoords="offset points", fontsize=7,
                color="#555555")
    ax.set_ylabel("Backlog (Mtoken)")
    ax.set_xlabel("Hour of day")
    ax.set_xlim(-0.5, 23.5)
    panel(ax, "(a) Batch backlog (clears by horizon end)")

    ax = axes[1]
    ax.bar(delay["max_batch_delay_h"], delay["total_cost_usd"],
           color=C["quantum"], width=1.2)
    for _, r in delay.iterrows():
        ax.annotate(f"{r['total_cost_usd']:.0f}",
                    (r["max_batch_delay_h"], r["total_cost_usd"]),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.set_xlabel("Maximum batch delay (h)")
    ax.set_xticks(delay["max_batch_delay_h"])
    ax.set_ylabel("Daily cost (\\$)")
    ax.set_ylim(delay["total_cost_usd"].min() * 0.97,
                delay["total_cost_usd"].max() * 1.04)
    panel(ax, "(b) Deadline sensitivity")
    fig.tight_layout()
    fig.savefig(FIG / "fig09_qos_design.png", bbox_inches="tight")
    plt.close(fig)


def fig10_kpis():
    """Three absolute KPI panels in the old ablation-bar style."""
    d = pd.read_csv(OUT / "headline_kpis.csv").set_index("scenario")
    scen = ["BAU", "compute_only", "joint_compute_storage"]
    labels = ["BAU\nbaseline", "Compute\nonly", "Compute +\nstorage"]
    # Keep the old bar palette roles: grey / amber / deep indigo.
    colors = [C["base"], C["orange"], C["quantum"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, col, title, fmt, unit in [
        (axes[0], "total_cost_usd", "(a) Absolute daily cost", "{:.0f}", "\\$"),
        (axes[1], "emissions_kg", "(b) Absolute daily emissions", "{:.0f}",
         "kgCO$_2$"),
        (axes[2], "evening_grid_kwh", "(c) Evening-peak import", "{:.0f}",
         "kWh"),
    ]:
        vals = [d.loc[s, col] for s in scen]
        ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_ylim(0, max(vals) * 1.16)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (i, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", labelsize=6.5)
        panel(ax, title)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_kpi_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig11_tradeoff():
    """Supported trade-off in the old frontier visual language.

    Grey connected curve + edged markers (formerly MILP line + annealing
    markers). Axes stay absolute, matching the revised objective.
    """
    d = pd.read_csv(OUT / "supported_tradeoff.csv").sort_values("emissions_kg")
    base = baseline_metrics()
    fig, ax = plt.subplots(figsize=(4.9, 3.6))
    ax.plot(d["emissions_kg"], d["total_cost_usd"], "-", color=C["base"],
            lw=1.6, zorder=2, label="Supported MILP curve")
    ax.scatter(d["emissions_kg"], d["total_cost_usd"], s=48,
               color=C["quantum"], edgecolors="k", linewidths=0.5, zorder=3,
               label="Weighted-sum points")
    ax.scatter([base["emissions_kg"]], [base["total_cost_usd"]],
               marker="s", s=55, color=C["base"], edgecolors="k",
               linewidths=0.5, zorder=4, label="BAU")
    offsets = {
        0: (6, -12, "left"),
        150: (6, -12, "left"),
        600: (-8, -4, "right"),
        1000: (-8, 6, "right"),
        5000: (-6, 6, "right"),
    }
    for lam, (dx, dy, ha) in offsets.items():
        r = d[d["lambda_usd_tco2"] == lam].iloc[0]
        ax.annotate(f"$\\lambda$={lam}",
                    (r["emissions_kg"], r["total_cost_usd"]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=7.5, color="#555555")
    ax.margins(x=0.08, y=0.12)
    ax.set_xlabel("Absolute daily emissions (kgCO$_2$)")
    ax.set_ylabel("Absolute daily cost (\\$)")
    ax.legend(frameon=False, fontsize=7.5, loc="center left",
              bbox_to_anchor=(0.02, 0.72))
    fig.tight_layout()
    fig.savefig(FIG / "fig11_pareto_frontier.png", bbox_inches="tight")
    plt.close(fig)


def fig12_tradeoff_mechanism():
    d = pd.read_csv(OUT / "supported_tradeoff.csv").sort_values(
        "lambda_usd_tco2")
    lambdas = d["lambda_usd_tco2"].to_numpy()
    selected = [0, 50, 150, 300, 600, 1000, 2500, 5000]
    levels = np.column_stack([
        solve_revised_dispatch(carbon_price=lam * 1e-6)["levels"]
        for lam in selected
    ])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axes[0]
    ax.plot(lambdas, d["total_cost_usd"], "o-", color=C["quantum"], lw=1.3)
    ax.set_xlabel("$\\lambda$ (\\$/tCO$_2$)")
    ax.set_ylabel("Daily cost (\\$)", color=C["quantum"])
    ax2 = ax.twinx()
    ax2.plot(lambdas, d["emissions_kg"], "s--", color=C["green"], lw=1.3)
    ax2.set_ylabel("Daily emissions (kgCO$_2$)", color=C["green"], fontsize=8)
    panel(ax, "(a) Supported trade-off sweep")

    ax = axes[1]
    im = ax.imshow(levels, aspect="auto", cmap=CMAP_LEVELS, vmin=0, vmax=2,
                   origin="lower", interpolation="nearest",
                   extent=[-0.5, len(selected) - 0.5, -0.5, 23.5])
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=45, fontsize=6)
    ax.set_xlabel("$\\lambda$ (\\$/tCO$_2$)")
    ax.set_ylabel("Hour of day")
    plt.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1, 2])
    panel(ax, "(b) Schedule migration")

    ax = axes[2]
    # MAC between adjacent lambda-ordered points after removing duplicate
    # (cost, emissions) solutions.  This follows the carbon-price sweep
    # rather than an emissions-sorted reconstruction of the front.
    unique = (d.sort_values("lambda_usd_tco2")
              .drop_duplicates(["total_cost_usd", "emissions_kg"])
              .reset_index(drop=True))
    x, y = [], []
    for i in range(1, len(unique)):
        dc = unique.loc[i, "total_cost_usd"] - unique.loc[i - 1, "total_cost_usd"]
        de = unique.loc[i - 1, "emissions_kg"] - unique.loc[i, "emissions_kg"]
        if de > 1e-9:
            x.append(unique.loc[i, "emissions_kg"])
            y.append(dc / (de / 1000.0))
    ax.step(x, y, where="post", color=C["quantum"], lw=1.4)
    ax.set_xlabel("Daily emissions (kgCO$_2$)")
    ax.set_ylabel("Marginal abatement cost\n(\\$/tCO$_2$)")
    ax.invert_xaxis()
    panel(ax, "(c) Marginal abatement cost")
    fig.tight_layout()
    fig.savefig(FIG / "fig12_pareto_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def fig14_seasonal():
    """Seasonal bars + seasonal supported curves, matching old fig14 objects."""
    d = pd.read_csv(OUT / "seasonal.csv")
    colmap = {"Jan": C["classic"], "Apr": C["green"], "Jul": C["red"],
              "Oct": C["orange"]}
    lams = [0, 50, 150, 300, 600, 1000, 2500, 5000]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))

    ax = axes[0]
    x = np.arange(len(d))
    w = 0.38
    ax.bar(x - w / 2, d["baseline_cost_usd"], width=w, color=C["base"],
           label="BAU baseline")
    ax.bar(x + w / 2, d["revised_cost_usd"], width=w, color=C["quantum"],
           label="Revised schedule")
    for xi, vb, vq, sv in zip(x, d["baseline_cost_usd"], d["revised_cost_usd"],
                              d["cost_saving_pct"]):
        ax.annotate(f"{vb:.0f}", (xi - w / 2, vb), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6)
        ax.annotate(f"{vq:.0f}", (xi + w / 2, vq), xytext=(0, 2),
                    textcoords="offset points", ha="center", fontsize=6)
        ax.annotate(f"$-${sv:.1f}%", (xi + w / 2, vq * 0.52), ha="center",
                    fontsize=5.5, color="white", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(d["season"], fontsize=7)
    ax.set_ylabel("Daily cost (\\$)")
    ax.set_ylim(0, d["baseline_cost_usd"].max() * 1.30)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    panel(ax, "(a) Daily cost across regional mean days")

    ax = axes[1]
    for name, month in [("Jan", 1), ("Apr", 4), ("Jul", 7), ("Oct", 10)]:
        pv_m = m.pv_profile_kw(month)
        pts = []
        for lam in lams:
            r = solve_revised_dispatch(carbon_price=lam * 1e-6, pv=pv_m)
            pts.append((r["emissions_kg"], r["total_cost_usd"]))
        pts = sorted(pts, key=lambda z: z[0])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", lw=1.3, markersize=4, color=colmap[name],
                label=name)
    ax.set_xlabel("Daily emissions (kgCO$_2$)")
    ax.set_ylabel("Daily cost (\\$)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.22)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2)
    panel(ax, "(b) Seasonal supported trade-offs")
    fig.tight_layout()
    fig.savefig(FIG / "fig14_seasonal.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig03_context_profiles()
    fig04_reduced_qubo()
    fig08_dispatch()
    fig10_kpis()
    fig11_tradeoff()
    fig12_tradeoff_mechanism()
    fig14_seasonal()
    print(f"Revised figures written to {FIG}")
