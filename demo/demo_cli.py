"""Démonstration en ligne de commande (script de la soutenance).

    python demo/demo_cli.py [--scenario S2] [--n 15] [--seed 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from fusionrh.belief import MassFunction, conflict_degree, dempster, dubois_prade, murphy, yager
from fusionrh.calibration import calibrate_agents, reliabilities_for
from fusionrh.config import AHP_EXPERT_MATRIX, CRITERIA
from fusionrh.generator import generate_scenario, make_config
from fusionrh.metrics import kendall, top_k_precision
from fusionrh.pipeline import METHOD_NAMES, Supervisor, make_system
from fusionrh.weighting import ahp_weights


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S2")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    banner("1. Pondération des critères (AHP)")
    w = ahp_weights(AHP_EXPERT_MATRIX)
    print(", ".join(f"{c}={x:.3f}" for c, x in zip(CRITERIA, w.weights)), f"| CR = {w.consistency:.3f}",
          "(cohérent)" if w.consistent else "(INCOHÉRENT : réélicitation)")

    banner("2. Paradoxe de Zadeh : deux agents quasi certains et opposés")
    m1 = MassFunction.from_dict({"Excellent": 0.99, "Moyen": 0.01})
    m2 = MassFunction.from_dict({"Faible": 0.99, "Moyen": 0.01})
    print(f"Agent 1 : {m1}\nAgent 2 : {m2}\nConflit K = {conflict_degree([m1.m, m2.m]):.4f}")
    for name, f in [("Dempster", dempster), ("Yager", yager), ("Dubois-Prade", dubois_prade), ("Murphy", murphy)]:
        print(f"  {name:<13}-> {MassFunction(f([m1.m, m2.m]))}")

    cfg = make_config(a.scenario, n_candidates=a.n)
    banner(f"3. Scénario {cfg.name} — {cfg.description}")
    test = generate_scenario(cfg, seed=a.seed)
    cal = generate_scenario(make_config(a.scenario, n_candidates=120), seed=10_000 + a.seed)
    calib = calibrate_agents(cal)
    for c in calib.values():
        print(f"  {c.name:<26} exactitude={c.accuracy:.2f} Brier={c.brier:.2f} ECE={c.ece:.2f} "
              f"fiabilité calibrée={c.reliability:.2f}{'  [SURCONFIANT]' if c.overconfident else ''}")
    rel = reliabilities_for("calibrées", test, calib)

    banner("4. Comparaison des opérateurs (fiabilités calibrées)")
    for m in METHOD_NAMES:
        sys_ = make_system(m, w.weights, rel)  # Choquet sans capacité identifiée = somme pondérée
        rc = sys_.evaluate(cal)
        sys_.supervisor = Supervisor.fit(rc.conflicts, rc.ignorances, cal.hard_cases)
        r = sys_.evaluate(test)
        top = ", ".join(f"#{test.candidates[i].cid}" for i in r.order[:3])
        print(f"  {m:<18} τ={kendall(r.scores, test.utilities):.2f}  top-5={top_k_precision(r.scores, test.utilities, 5):.0%}"
              f"  top-3=[{top}]  renvoyés={int(r.flagged.sum())}")
    truth = ", ".join(f"#{test.candidates[i].cid}" for i in np.argsort(-test.utilities)[:3])
    print(f"  {'VÉRITÉ':<18} top-3=[{truth}]")

    banner("5. Explication détaillée (Yager) du 1er candidat et d'un cas renvoyé au jury")
    sys_ = make_system("Yager", w.weights, rel)
    rc = sys_.evaluate(cal)
    sys_.supervisor = Supervisor.fit(rc.conflicts, rc.ignorances, cal.hard_cases)
    r = sys_.evaluate(test)
    print(r.explain(int(r.order[0])))
    flagged = np.flatnonzero(r.flagged)
    if len(flagged):
        print()
        print(r.explain(int(flagged[0])))

    banner("6. Même candidat, explication Mamdani (règles activées)")
    rm = make_system("Mamdani", w.weights, rel).evaluate(test)
    print(rm.explain(int(r.order[0])))


if __name__ == "__main__":
    main()
