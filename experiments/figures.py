"""Génère toutes les figures du rapport à partir de results/*.csv.

    python -m experiments.figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from fusionrh.pipeline import EVIDENTIAL_METHODS, FUZZY_METHODS, METHOD_NAMES

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Palette de référence (slots catégoriels dans l'ordre fixe) et encre
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FAMILY_COLOR = {"évidentielle": SERIES[0], "floue": SERIES[1]}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


def fam(m):
    return "évidentielle" if m in EVIDENTIAL_METHODS else "floue"


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def family_legend(ax, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAMILY_COLOR["évidentielle"], label="Famille évidentielle (+TOPSIS)"),
                       Patch(color=FAMILY_COLOR["floue"], label="Famille floue")], loc=loc, fontsize=8)


# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_axis_off(); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    layers = [
        ("1. Agents évaluateurs", "RH · Manager technique · Test psychométrique · Analyse CV · LLM (simulé)"),
        ("2. Modélisation de l'incertitude", "masse m sur Θ={Faible,Moyen,Bon,Excellent} (+ m(Θ))  ⇄  nombre flou triangulaire"),
        ("3. Calibration et pondération", "fiabilité α (exactitude, Brier, ECE, audit d'équité) · poids AHP/BWM (CR < 0,1)"),
        ("4. Fusion (interchangeable)", "Dempster · Yager · Dubois-Prade · Murphy | moy. floue · OWA · Choquet · Mamdani"),
        ("5. Décision et explication", "TOPSIS / fuzzy TOPSIS · intervalle [Bel ; Pl] · règles activées · accords/conflits"),
        ("6. Supervision humaine", "conflit K ou ignorance > seuil  ⇒  renvoi au jury humain"),
    ]
    for k, (title, body) in enumerate(layers):
        y = 6.05 - k * 1.08
        col = SERIES[0] if k != 3 else SERIES[1]
        ax.add_patch(FancyBboxPatch((0.2, y - 0.38), 9.6, 0.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=SURFACE, ec=col, lw=1.6))
        ax.add_patch(FancyBboxPatch((0.2, y - 0.38), 0.12, 0.8, boxstyle="square,pad=0", fc=col, ec=col))
        ax.text(0.5, y + 0.12, title, fontsize=10, fontweight="bold", color=INK, va="center")
        ax.text(0.5, y - 0.18, body, fontsize=8.3, color=INK2, va="center")
        if k < len(layers) - 1:
            ax.annotate("", xy=(5, y - 0.66), xytext=(5, y - 0.42),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    save(fig, "fig_architecture.png")


def fig_kendall_scenarios(df):
    d = df[df["mode"] == "calibrées"]
    scen = sorted(d["scenario"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True)
    for ax, s in zip(axes.ravel(), scen):
        g = d[d["scenario"] == s].groupby("methode")["kendall"]
        mean, err = g.mean().reindex(METHOD_NAMES), g.apply(ci95).reindex(METHOD_NAMES)
        y = np.arange(len(METHOD_NAMES))[::-1]
        ax.barh(y, mean, xerr=err, height=0.62, color=[FAMILY_COLOR[fam(m)] for m in METHOD_NAMES],
                error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        best = mean.idxmax()
        for yi, m in zip(y, METHOD_NAMES):
            ax.text(mean[m] + err[m] + 0.01, yi, f"{mean[m]:.2f}", va="center", fontsize=7,
                    color=INK if m == best else MUTED, fontweight="bold" if m == best else "normal")
        ax.set_yticks(y); ax.set_yticklabels(METHOD_NAMES, fontsize=8)
        ax.set_title(s); ax.set_xlim(0.3, 1.0); ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("τ de Kendall (moy. ± IC95 %)")
    from matplotlib.patches import Patch
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=[Patch(color=FAMILY_COLOR["évidentielle"], label="Famille évidentielle (+TOPSIS)"),
                        Patch(color=FAMILY_COLOR["floue"], label="Famille floue")],
               loc="lower center", ncol=2, fontsize=8.5)
    save(fig, "fig_kendall_scenarios.png")


def fig_modes(df):
    modes = ["égales", "déclarées", "calibrées", "optimisées"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, sub) in zip(axes, [("Tous scénarios", df), ("S3 — agent surconfiant",
                                                                  df[df["scenario"].str.startswith("S3")])]):
        piv = sub.pivot_table(index="methode", columns="mode", values="kendall").reindex(METHOD_NAMES)
        y = np.arange(len(METHOD_NAMES))[::-1]
        for k, m in enumerate(modes):
            ax.scatter(piv[m], y + (k - 1.5) * 0.15, s=34, color=SERIES[k], label=f"fiabilités {m}",
                       edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.hlines(y, piv.min(axis=1), piv.max(axis=1), color=AXIS, lw=1, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(METHOD_NAMES); ax.set_title(title)
        ax.set_xlabel("τ de Kendall moyen"); ax.grid(axis="y", visible=False)
    h, l = axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5)
    save(fig, "fig_modes_fiabilite.png")


def fig_conflict_sweep(sw):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (title, ms) in zip(axes[:2], [("Famille évidentielle", EVIDENTIAL_METHODS), ("Famille floue", FUZZY_METHODS)]):
        for k, m in enumerate(ms):
            g = sw[sw["methode"] == m].groupby("extra_fault")["kendall"].mean()
            ax.plot(g.index, g.values, marker="o", color=SERIES[k], label=m)
        ax.set_title(title); ax.set_xlabel("probabilité de panne ajoutée"); ax.set_ylim(0.2, 0.9)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("τ de Kendall")
    for k, m in enumerate(["Dempster", "Moyenne floue"]):
        g = sw[sw["methode"] == m].groupby("extra_fault")["renvoi_f1"].mean()
        axes[2].plot(g.index, g.values, marker="o", color=SERIES[k], label=f"F1 renvoi — {m}")
    gk = sw[sw["methode"] == "Dempster"].groupby("extra_fault")["K_moyen"].mean()
    axes[2].plot(gk.index, gk.values, marker="s", color=SERIES[2], label="K moyen (max par critère)")
    axes[2].set_title("Conflit et renvoi au jury"); axes[2].set_xlabel("probabilité de panne ajoutée")
    axes[2].set_ylim(0, 1); axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_balayage_conflit.png")


def fig_zadeh(z):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x_of = lambda d: -np.log10(1 - d["K"])
    for k, m in enumerate(["Dempster", "Yager", "Dubois-Prade", "Murphy", "Moyenne floue"]):
        d = z[z["methode"] == m]
        axes[0].plot(x_of(d), d["E_utilite"], marker="o", color=SERIES[k], label=m)
        if m != "Moyenne floue":
            axes[1].plot(x_of(d), d["m_Moyen"], marker="o", color=SERIES[k], label=m)
    axes[0].set_title("Évaluation fusionnée (E[BetP] / centroïde)")
    axes[0].set_ylabel("utilité ∈ [0,1]")
    axes[1].set_title("Masse attribuée à « Moyen » (jugé quasi impossible)")
    for ax in axes:
        ax.set_xlabel("−log₁₀(1 − K)   (K → 1 vers la droite)")
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_zadeh.png")


def fig_robustness(rob, sens):
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), sharey=True)
    cols = [("stab_top3_perturbation", "Stabilité top-3\n(avis perturbés ±0,05)"),
            ("stab_top3_retrait_agent", "Stabilité top-3\n(retrait d'un agent)"),
            ("inversion_rang_pct", "Inversion de rang (%)\n(ajout d'un candidat)")]
    y = np.arange(len(METHOD_NAMES))[::-1]
    for ax, (c, t) in zip(axes[:3], cols):
        g = rob.groupby("methode")[c].mean().reindex(METHOD_NAMES)
        ax.barh(y, g, height=0.62, color=[FAMILY_COLOR[fam(m)] for m in METHOD_NAMES])
        for yi, m in zip(y, METHOD_NAMES):
            ax.text(g[m], yi, f" {g[m]:.2f}", va="center", fontsize=7, color=INK2)
        ax.set_title(t, fontsize=9); ax.grid(axis="y", visible=False)
    s = sens[sens["delta"].abs() == 0.2].groupby("methode")["kendall_vs_base"].mean().reindex(METHOD_NAMES)
    axes[3].barh(y, s, height=0.62, color=[FAMILY_COLOR[fam(m)] for m in METHOD_NAMES])
    for yi, m in zip(y, METHOD_NAMES):
        axes[3].text(s[m], yi, f" {s[m]:.3f}", va="center", fontsize=7, color=INK2)
    axes[3].set_title("Sensibilité aux poids ±20 %\n(τ vs classement de base)", fontsize=9)
    axes[3].set_xlim(0.8, 1.02); axes[3].grid(axis="y", visible=False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(METHOD_NAMES)
    family_legend(axes[2], "lower right")
    fig.tight_layout()
    save(fig, "fig_robustesse.png")


def fig_fairness(eq, pooled):
    d = eq[eq["scenario"].str.startswith("S5")]
    p = pooled[pooled["scenario"].str.startswith("S5")].copy()
    p["B_sur_A"] = p["taux_B"] / p["taux_A"]
    truth = float(p.loc[p["methode"] == "Vérité", "B_sur_A"].iloc[0])
    modes = ["déclarées", "calibrées", "calibrées+audit"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(METHOD_NAMES))[::-1]
    piv0 = p[p["methode"] != "Vérité"].pivot_table(index="methode", columns="mode", values="B_sur_A").reindex(METHOD_NAMES)
    piv1 = d.pivot_table(index="methode", columns="mode", values="cf_part_changee").reindex(METHOD_NAMES)
    for ax, piv, t in [(axes[0], piv0, "Taux de sélection B / A (cumulé sur 30 réplications)"),
                       (axes[1], piv1, "Contrefactuel : part des rangs modifiés")]:
        for k, m in enumerate(modes):
            ax.barh(y + (1 - k) * 0.26, piv[m], height=0.24, color=SERIES[k], label=f"fiabilités {m}")
        ax.set_title(t); ax.grid(axis="y", visible=False)
    axes[0].axvline(truth, color=INK, lw=1.2, ls="--")
    axes[0].text(truth + 0.01, y[-1] - 0.75, f"vérité (mérite) = {truth:.2f}".replace(".", ","), fontsize=7.5, color=INK)
    axes[0].axvline(0.8 * truth, color=MUTED, lw=1, ls=":")
    axes[0].text(0.8 * truth + 0.01, y[0] + 0.5, "4/5 du mérite", fontsize=7.5, color=MUTED)
    axes[0].set_xlim(0, max(1.4, truth + 0.2))
    axes[0].set_yticks(y); axes[0].set_yticklabels(METHOD_NAMES)
    h, l = axes[1].get_legend_handles_labels()
    fig.suptitle("Scénario S5 — agents biaisés contre le groupe B", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5)
    save(fig, "fig_equite.png")


def fig_reliability(diag, cal):
    d = diag[diag["scenario"].str.startswith("S3")]
    agents = list(dict.fromkeys(d["agent"]))
    fig, axes = plt.subplots(1, len(agents), figsize=(13, 2.9), sharey=True)
    for ax, a in zip(axes, agents):
        g = d[d["agent"] == a]
        ax.plot([0.25, 1], [0.25, 1], color=AXIS, lw=1, ls="--")
        ax.plot(g["confiance"], g["exactitude"], marker="o", color=SERIES[0])
        row = cal[(cal["agent"] == a) & cal["scenario"].str.startswith("S3")].iloc[0]
        ax.set_title(a.replace(" (", "\n("), fontsize=8.5)
        ax.text(0.27, 0.93, f"Brier {row.brier:.2f}\nECE {row.ece:.2f}", fontsize=7.5, color=INK2, va="top")
        if row.surconfiant:
            ax.text(0.97, 0.3, "SURCONFIANT", fontsize=7.5, color="#d03b3b", ha="right", fontweight="bold")
        ax.set_xlim(0.25, 1); ax.set_ylim(0, 1); ax.set_xlabel("confiance (max BetP)")
    axes[0].set_ylabel("exactitude observée")
    fig.tight_layout()
    save(fig, "fig_diagramme_fiabilite.png")


def fig_cost(cost):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    reps = ["Dempster", "Yager", "Moyenne floue", "Choquet", "Mamdani"]
    for k, m in enumerate(reps):
        g = cost[(cost["methode"] == m) & (cost["n_agents"] == 5)]
        axes[0].plot(g["n_candidats"], g["temps_ms"], marker="o", color=SERIES[k], label=m)
        g = cost[(cost["methode"] == m) & (cost["n_candidats"] == 80)]
        axes[1].plot(g["n_agents"], g["temps_ms"], marker="o", color=SERIES[k], label=m)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("nombre de candidats (5 agents)"); axes[0].set_ylabel("temps (ms)")
    axes[1].set_xlabel("nombre d'agents (80 candidats)"); axes[1].set_yscale("log")
    axes[0].set_title("Coût vs candidats"); axes[1].set_title("Coût vs agents")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_cout.png")


def _heat(ax, mat, rows, cols, fmt="{:.2f}", vmin=None, vmax=None):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    lo, hi = (vmin if vmin is not None else np.nanmin(mat)), (vmax if vmax is not None else np.nanmax(mat))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if (v - lo) / max(hi - lo, 1e-9) > 0.55 else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_explain_grid(summary):
    g = summary["grille_explicabilite"]
    mat = np.array([g["scores"][m] for m in METHOD_NAMES], float)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _heat(ax, mat, [f"{m} ({int(mat[i].sum())}/14)" for i, m in enumerate(METHOD_NAMES)], g["criteres"],
          fmt="{:.0f}", vmin=0, vmax=2)
    ax.set_title("Grille d'explicabilité (0 = absent, 1 = partiel, 2 = complet)")
    fig.tight_layout()
    save(fig, "fig_grille_explicabilite.png")


def fig_detection(df):
    d = df[df["mode"] == "calibrées"]
    piv = d.pivot_table(index="methode", columns="scenario", values="renvoi_f1").reindex(METHOD_NAMES)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _heat(ax, piv.values, METHOD_NAMES, [c.split(" ", 1)[1] if " " in c else c for c in piv.columns], vmin=0, vmax=1)
    ax.set_title("F1 du renvoi au jury (cas difficiles détectés) — fiabilités calibrées")
    fig.tight_layout()
    save(fig, "fig_renvoi_jury.png")


def main():
    df = pd.read_csv(RES / "principal.csv")
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig_architecture()
    fig_kendall_scenarios(df)
    fig_modes(df)
    fig_conflict_sweep(pd.read_csv(RES / "balayage_conflit.csv"))
    fig_zadeh(pd.read_csv(RES / "zadeh.csv"))
    fig_robustness(pd.read_csv(RES / "robustesse.csv"), pd.read_csv(RES / "sensibilite.csv"))
    fig_fairness(pd.read_csv(RES / "equite.csv"), pd.read_csv(RES / "equite_cumulee.csv"))
    fig_reliability(pd.read_csv(RES / "diagramme_fiabilite.csv"), pd.read_csv(RES / "calibration_agents.csv"))
    fig_cost(pd.read_csv(RES / "cout.csv"))
    fig_explain_grid(summary)
    fig_detection(df)
    print("figures écrites dans", FIG)


if __name__ == "__main__":
    main()
