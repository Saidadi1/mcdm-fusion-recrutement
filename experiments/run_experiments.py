"""Campagne expérimentale complète.

    python -m experiments.run_experiments [--reps 30] [--quick]

Produit results/*.csv, results/summary.json (lu par le rapport) ; les figures sont produites par
experiments/figures.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from fusionrh.agents import AgentProfile
from fusionrh.aggregation import fit_interactions
from fusionrh.belief import MassFunction, conflict_degree, dempster, dubois_prade, murphy, yager
from fusionrh.calibration import (calibrate_agents, fairness_adjusted, llm_stability, optimize_alpha,
                                  reliabilities_for)
from fusionrh.config import (AHP_EXPERT_MATRIX, AHP_INCONSISTENT_MATRIX, BWM_BEST, BWM_BEST_TO_OTHERS,
                             BWM_OTHERS_TO_WORST, BWM_WORST, CRITERIA)
from fusionrh.explainability import GRID_CRITERIA, GRID_SCORES
from fusionrh.fusion.fuzzy_ops import WeightedFuzzyMean
from fusionrh.fuzzy import mass_to_tfn
from fusionrh.generator import SCENARIOS, generate_scenario, make_config
from fusionrh.metrics import (auc, detection_scores, discordant_pairs, four_fifths, holm, kendall,
                              selection_mask, spearman, top_k_overlap, top_k_precision)
from fusionrh.pipeline import METHOD_NAMES, DecisionSystem, Supervisor, make_system
from fusionrh.weighting import ahp_weights, bwm_weights, perturb_weights

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
MODES = ["égales", "déclarées", "calibrées", "optimisées"]
N_CAL = 120


def fit_capacity(cal, weights, rel):
    system = DecisionSystem("tmp", WeightedFuzzyMean(), None, weights, rel)
    fused = system.fuse_scenario(cal)
    x = np.array([[f.crisp for f in row] for row in fused])
    return fit_interactions(x, cal.utilities, weights)


def prepare(cfg, rep, weights):
    """Jeu de calibration + jeu de test + fiabilités pour les 4 modes."""
    cal = generate_scenario(cfg.__class__(**{**cfg.__dict__, "n_candidates": N_CAL}), seed=10_000 + rep)
    test = generate_scenario(cfg, seed=rep)
    calib = calibrate_agents(cal)
    opt = optimize_alpha(cal, "Dempster", passes=1, start={k: v.reliability for k, v in calib.items()})
    rels = {m: reliabilities_for(m, test, calib, opt) for m in MODES}
    return cal, test, calib, rels


def build_system(method, weights, rel, cal):
    cap = fit_capacity(cal, weights, rel) if method == "Choquet" else None
    sys_ = make_system(method, weights, rel, capacity=cap)
    rc = sys_.evaluate(cal)
    sys_.supervisor = Supervisor.fit(rc.conflicts, rc.ignorances, cal.hard_cases)
    return sys_


def evaluate_row(r, test, k_sel):
    u = test.utilities
    hard = test.hard_cases
    idx, expert = test.expert_ranking()
    det = detection_scores(r.flagged, hard)
    ff = four_fifths(selection_mask(r.scores, k_sel), test.groups)
    return {
        "kendall": kendall(r.scores, u), "spearman": spearman(r.scores, u),
        "top3": top_k_precision(r.scores, u, 3), "top5": top_k_precision(r.scores, u, 5),
        "top10": top_k_precision(r.scores, u, 10),
        "kendall_expert": kendall(r.scores[idx], expert),
        "K_moyen": float(r.conflicts.mean()), "ignorance_moy": float(r.ignorances.mean()),
        "renvoi_precision": det["precision"], "renvoi_rappel": det["rappel"], "renvoi_f1": det["f1"],
        "taux_renvoi": det["taux_renvoi"],
        "auc_conflit": auc(r.conflicts[~test.incomplete_cases], test.fault_cases[~test.incomplete_cases]),
        "auc_ignorance": auc(r.ignorances[~test.fault_cases], test.incomplete_cases[~test.fault_cases]),
        "ratio_impact": ff["ratio_impact"],
        "temps_ms": r.elapsed * 1000,
    }


# ---------------------------------------------------------------------------
def exp_main(reps, weights):
    rows = []
    for key in SCENARIOS:
        cfg = make_config(key)
        for rep in range(reps):
            cal, test, calib, rels = prepare(cfg, rep, weights)
            for mode in MODES:
                for method in METHOD_NAMES:
                    sys_ = build_system(method, weights, rels[mode], cal)
                    r = sys_.evaluate(test)
                    rows.append({"scenario": cfg.name, "rep": rep, "mode": mode, "methode": method,
                                 **evaluate_row(r, test, k_sel=10)})
        print(f"  {cfg.name} terminé", flush=True)
    return pd.DataFrame(rows)


def exp_conflict_sweep(reps, weights):
    rows = []
    for extra in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        cfg = make_config("S1", extra_fault=extra, name=f"panne+{extra:.1f}")
        for rep in range(reps):
            cal, test, calib, rels = prepare(cfg, rep, weights)
            for method in METHOD_NAMES:
                r = build_system(method, weights, rels["calibrées"], cal).evaluate(test)
                rows.append({"extra_fault": extra, "rep": rep, "methode": method,
                             **evaluate_row(r, test, 10)})
    return pd.DataFrame(rows)


def exp_zadeh():
    """Comportement quand K → 1 : deux agents quasi catégoriques en désaccord total."""
    rows = []
    for eps in [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001]:
        m1 = MassFunction.from_dict({"Excellent": 1 - eps, "Moyen": eps}).m
        m2 = MassFunction.from_dict({"Faible": 1 - eps, "Moyen": eps}).m
        k = conflict_degree([m1, m2])
        for name, f in [("Dempster", dempster), ("Yager", yager), ("Dubois-Prade", dubois_prade),
                        ("Murphy", murphy)]:
            mf = MassFunction(f([m1, m2]))
            rows.append({"eps": eps, "K": k, "methode": name, "m_Moyen": mf.m[2], "m_Theta": mf.ignorance,
                         "E_utilite": mf.expected_utility(), "bel_moyen": mf.bel("Moyen"),
                         "pl_moyen": mf.pl("Moyen")})
        t = mass_to_tfn(MassFunction((m1 + m2) / 2))
        rows.append({"eps": eps, "K": k, "methode": "Moyenne floue", "m_Moyen": np.nan, "m_Theta": np.nan,
                     "E_utilite": t.centroid, "bel_moyen": np.nan, "pl_moyen": np.nan})
    return pd.DataFrame(rows)


def exp_robustness(reps, weights):
    rows, sens = [], []
    for key in ("S1", "S2"):
        cfg = make_config(key)
        for rep in range(reps):
            cal, test, calib, rels = prepare(cfg, rep, weights)
            rel = rels["calibrées"]
            for method in METHOD_NAMES:
                sys_ = build_system(method, weights, rel, cal)
                base = sys_.evaluate(test).scores
                pert = [top_k_overlap(base, sys_.evaluate(test.regenerate(run_id=100 + p, extra_noise=0.05)).scores)
                        for p in range(3)]
                rem = []
                for a in range(len(test.agents)):
                    rem.append(top_k_overlap(base, sys_.evaluate(test.without_agent(a)).scores))
                rev = []
                for s in range(3):
                    sc2 = test.with_extra_candidates(1, seed_offset=4242 + s)
                    rev.append(discordant_pairs(base, sys_.evaluate(sc2).scores[:test.n]))
                rows.append({"scenario": cfg.name, "rep": rep, "methode": method,
                             "stab_top3_perturbation": np.mean(pert), "stab_top3_retrait_agent": np.mean(rem),
                             "pire_retrait_agent": np.min(rem), "inversion_rang_pct": 100 * np.mean(rev)})
                if key == "S1":
                    for delta in (-0.2, -0.1, 0.1, 0.2):
                        kt, ov = [], []
                        for j in range(len(CRITERIA)):
                            sys_.weights = perturb_weights(weights, j, delta)
                            if method == "Choquet":
                                sys_.aggregator.capacity.shapley = sys_.weights
                            s2 = sys_.evaluate(test).scores
                            kt.append(kendall(s2, base)); ov.append(top_k_overlap(base, s2))
                        sys_.weights = weights
                        if method == "Choquet":
                            sys_.aggregator.capacity.shapley = weights
                        sens.append({"rep": rep, "methode": method, "delta": delta,
                                     "kendall_vs_base": np.mean(kt), "stab_top3": np.mean(ov)})
    return pd.DataFrame(rows), pd.DataFrame(sens)


def exp_fairness(reps, weights):
    rows = []
    for key in ("S1", "S5"):
        cfg = make_config(key)
        for rep in range(reps):
            cal, test, calib, rels = prepare(cfg, rep, weights)
            rels["calibrées+audit"] = fairness_adjusted(rels["calibrées"], calib)
            for mode in ("déclarées", "calibrées", "calibrées+audit"):
                for method in METHOD_NAMES:
                    sys_ = build_system(method, weights, rels[mode], cal)
                    r = sys_.evaluate(test)
                    deltas = []
                    for i in range(10):
                        r2 = sys_.evaluate(test.with_flipped_group(i))
                        deltas.append(r2.ranks[i] - r.ranks[i])
                    deltas = np.array(deltas)
                    ff = four_fifths(selection_mask(r.scores, 10), test.groups)
                    truth_ff = four_fifths(selection_mask(test.utilities, 10), test.groups)
                    rows.append({"scenario": cfg.name, "rep": rep, "mode": mode, "methode": method,
                                 "ratio_impact": ff["ratio_impact"], "ratio_impact_verite": truth_ff["ratio_impact"],
                                 "cf_delta_rang_abs": float(np.abs(deltas).mean()),
                                 "cf_part_changee": float((deltas != 0).mean()),
                                 "kendall": kendall(r.scores, test.utilities)})
    return pd.DataFrame(rows)


def exp_cost(weights):
    rows = []
    base_agents = make_config("S1").agents
    for n_agents in (3, 4, 5, 7, 10):
        agents = [base_agents[k % 5].with_(name=f"{base_agents[k % 5].name}#{k}") for k in range(n_agents)]
        for n_cand in (20, 40, 80, 160, 320):
            cfg = make_config("S1", n_candidates=n_cand, agents=agents)
            sc = generate_scenario(cfg, seed=1)
            rel = {a.name: 0.8 for a in sc.agents}
            for method in METHOD_NAMES:
                sys_ = make_system(method, weights, rel)
                t0 = time.perf_counter(); sys_.evaluate(sc); dt = time.perf_counter() - t0
                rows.append({"n_agents": n_agents, "n_candidats": n_cand, "methode": method, "temps_ms": dt * 1000})
    return pd.DataFrame(rows)


def exp_calibration():
    rows, diag = [], []
    for key in ("S1", "S3", "S5"):
        cfg = make_config(key, n_candidates=200)
        sc = generate_scenario(cfg, seed=2024)
        for name, c in calibrate_agents(sc).items():
            rows.append({"scenario": cfg.name, "agent": name, "exactitude": c.accuracy, "brier": c.brier,
                         "ece": c.ece, "confiance_moy": c.mean_confidence, "surconfiance": c.overconfidence,
                         "surconfiant": c.overconfident, "fiabilite_calibree": c.reliability,
                         "ecart_groupes": c.group_gap, "biais_detecte": c.biased,
                         "fiabilite_declaree": next(a.profile.declared_reliability for a in sc.agents if a.name == name)})
            for conf, acc, n in c.bins:
                diag.append({"scenario": cfg.name, "agent": name, "confiance": conf, "exactitude": acc, "n": n})
    sc = generate_scenario(make_config("S1", n_candidates=60), seed=7)
    llm = llm_stability(sc, "LLM entretien (simulé)", runs=5)
    rule = llm_stability(sc, "Manager technique", runs=5)
    return pd.DataFrame(rows), pd.DataFrame(diag), {"LLM": llm, "Manager (règles)": rule}


def stats_tests(df):
    out = {}
    cal = df[df["mode"] == "calibrées"]
    for scen, g in cal.groupby("scenario"):
        piv = g.pivot(index="rep", columns="methode", values="kendall")[METHOD_NAMES]
        fr = friedmanchisquare(*[piv[m] for m in METHOD_NAMES])
        best = piv.mean().idxmax()
        others = [m for m in METHOD_NAMES if m != best]
        pv = []
        for m in others:
            d = piv[best] - piv[m]
            pv.append(1.0 if np.allclose(d, 0) else wilcoxon(piv[best], piv[m]).pvalue)
        adj = holm(pv)
        out[scen] = {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue), "meilleure": best,
                     "rangs_moyens": piv.rank(axis=1, ascending=False).mean().round(2).to_dict(),
                     "wilcoxon_holm_vs_meilleure": {m: float(p) for m, p in zip(others, adj)}}
    # effet du mode de fiabilité (toutes scénarios confondus)
    modes = {}
    for method in METHOD_NAMES:
        g = df[df["methode"] == method].pivot_table(index=["scenario", "rep"], columns="mode", values="kendall")
        res = {}
        for a, b in [("calibrées", "égales"), ("calibrées", "déclarées"), ("optimisées", "calibrées")]:
            d = g[a] - g[b]
            p = 1.0 if np.allclose(d, 0) else wilcoxon(g[a], g[b]).pvalue
            res[f"{a} vs {b}"] = {"diff_moy": float(d.mean()), "p": float(p)}
        modes[method] = res
    flat = [(m, k) for m in modes for k in modes[m]]
    adj = holm([modes[m][k]["p"] for m, k in flat])
    for (m, k), p in zip(flat, adj):
        modes[m][k]["p_holm"] = float(p)
    out["effet_mode_fiabilite"] = modes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    reps = 3 if args.quick else args.reps
    sub_reps = max(2, reps // 3)
    t0 = time.time()

    ahp = ahp_weights(AHP_EXPERT_MATRIX)
    bwm = bwm_weights(BWM_BEST, BWM_WORST, BWM_BEST_TO_OTHERS, BWM_OTHERS_TO_WORST)
    bad = ahp_weights(AHP_INCONSISTENT_MATRIX)
    weights = ahp.weights
    summary = {"poids": {"AHP": ahp.weights.tolist(), "AHP_CR": ahp.consistency,
                         "AHP_lambda": ahp.details["lambda_max"], "BWM": bwm.weights.tolist(),
                         "BWM_ratio": bwm.consistency, "BWM_xi": bwm.details["xi"],
                         "AHP_incoherent_CR": bad.consistency, "criteres": list(CRITERIA)},
               "reps": reps, "scenarios": {make_config(k).name: make_config(k).description for k in SCENARIOS}}

    print("1/7 expérience principale…", flush=True)
    main_df = exp_main(reps, weights); main_df.to_csv(RES / "principal.csv", index=False)
    print("2/7 balayage du conflit…", flush=True)
    exp_conflict_sweep(sub_reps, weights).to_csv(RES / "balayage_conflit.csv", index=False)
    print("3/7 paradoxe de Zadeh…", flush=True)
    exp_zadeh().to_csv(RES / "zadeh.csv", index=False)
    print("4/7 robustesse et sensibilité…", flush=True)
    rob, sens = exp_robustness(sub_reps, weights)
    rob.to_csv(RES / "robustesse.csv", index=False); sens.to_csv(RES / "sensibilite.csv", index=False)
    print("5/7 équité…", flush=True)
    exp_fairness(sub_reps, weights).to_csv(RES / "equite.csv", index=False)
    print("6/7 coût de calcul…", flush=True)
    exp_cost(weights).to_csv(RES / "cout.csv", index=False)
    print("7/7 calibration…", flush=True)
    cal_df, diag_df, stab = exp_calibration()
    cal_df.to_csv(RES / "calibration_agents.csv", index=False); diag_df.to_csv(RES / "diagramme_fiabilite.csv", index=False)
    summary["stabilite_llm"] = stab
    summary["tests_statistiques"] = stats_tests(main_df)
    summary["grille_explicabilite"] = {"criteres": [c for c, _ in GRID_CRITERIA], "scores": GRID_SCORES}
    summary["duree_s"] = time.time() - t0
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Terminé en {summary['duree_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
