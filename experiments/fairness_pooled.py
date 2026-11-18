"""Ratio d'impact (règle des 4/5) calculé sur les sélections CUMULÉES de toutes les réplications.

Sur un top-10 parmi 40 candidats, le ratio min/max par réplication est très bruité et biaisé vers le bas
(même la vérité donne ≈ 0,5). On cumule donc les sélections par groupe sur les réplications.

    python -m experiments.fairness_pooled [--reps 30]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from experiments.run_experiments import RES, build_system, prepare
from fusionrh.calibration import fairness_adjusted
from fusionrh.config import AHP_EXPERT_MATRIX
from fusionrh.generator import make_config
from fusionrh.metrics import selection_mask
from fusionrh.pipeline import METHOD_NAMES
from fusionrh.weighting import ahp_weights

MODES = ["déclarées", "calibrées", "calibrées+audit"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    reps = ap.parse_args().reps
    w = ahp_weights(AHP_EXPERT_MATRIX).weights
    counts = {}
    for key in ("S1", "S5"):
        cfg = make_config(key)
        for rep in range(reps):
            cal, test, calib, rels = prepare(cfg, rep, w)
            rels["calibrées+audit"] = fairness_adjusted(rels["calibrées"], calib)
            g = test.groups
            entries = [("Vérité", "—", test.utilities)]
            for mode in MODES:
                for m in METHOD_NAMES:
                    entries.append((m, mode, build_system(m, w, rels[mode], cal).evaluate(test).scores))
            for m, mode, scores in entries:
                sel = selection_mask(scores, 10)
                c = counts.setdefault((cfg.name, m, mode), np.zeros(4))
                c += [sel[g == "A"].sum(), (g == "A").sum(), sel[g == "B"].sum(), (g == "B").sum()]
        print(cfg.name, "terminé", flush=True)
    rows = []
    for (scen, m, mode), (sa, na, sb, nb) in counts.items():
        ra, rb = sa / na, sb / nb
        rows.append({"scenario": scen, "methode": m, "mode": mode, "taux_A": ra, "taux_B": rb,
                     "ratio_impact_cumule": min(ra, rb) / max(ra, rb), "respecte_4_5": min(ra, rb) / max(ra, rb) >= 0.8})
    pd.DataFrame(rows).to_csv(RES / "equite_cumulee.csv", index=False)
    print("écrit :", RES / "equite_cumulee.csv")


if __name__ == "__main__":
    main()
