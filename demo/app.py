"""Démonstration interactive.

    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from fusionrh.aggregation import fit_interactions
from fusionrh.belief import (MassFunction, conflict_degree, dempster, dubois_prade, label_of, murphy,
                             yager, TotalConflictError)
from fusionrh.calibration import (RELIABILITY_MODES, calibrate_agents, fairness_adjusted, reliabilities_for)
from fusionrh.config import AHP_EXPERT_MATRIX, CLASSES, CRITERIA_LABELS, CRITERIA
from fusionrh.fusion.fuzzy_ops import WeightedFuzzyMean
from fusionrh.fuzzy import mass_to_tfn
from fusionrh.generator import SCENARIOS, generate_scenario, make_config
from fusionrh.metrics import kendall, top_k_precision
from fusionrh.pipeline import METHOD_NAMES, DecisionSystem, Supervisor, make_system
from fusionrh.weighting import ahp_weights

st.set_page_config(page_title="Fusion d'avis — sélection de candidats", layout="wide")
st.title("Système multi-agents d'aide à la sélection de candidats (MCDM)")
st.caption("Outil d'AIDE à la décision sur données synthétiques : la décision finale appartient au jury humain.")

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.header("Scénario")
    scen_key = st.selectbox("Scénario synthétique", list(SCENARIOS),
                            format_func=lambda k: make_config(k).name)
    st.caption(make_config(scen_key).description)
    n_cand = st.slider("Nombre de candidats", 10, 80, 25, 5)
    seed = st.number_input("Graine aléatoire", 0, 10_000, 1)
    st.header("Système")
    method = st.selectbox("Opérateur de fusion", METHOD_NAMES, index=1)
    mode = st.selectbox("Fiabilités des agents", list(RELIABILITY_MODES) + ["calibrées+audit"], index=2)


@st.cache_resource(show_spinner="Génération du scénario et calibration…")
def setup(scen_key: str, n_cand: int, seed: int):
    cfg = make_config(scen_key, n_candidates=n_cand)
    test = generate_scenario(cfg, seed=int(seed))
    cal = generate_scenario(make_config(scen_key, n_candidates=120), seed=10_000 + int(seed))
    calib = calibrate_agents(cal)
    return cfg, test, cal, calib


cfg, test, cal, calib = setup(scen_key, n_cand, seed)
weights = ahp_weights(AHP_EXPERT_MATRIX).weights


def rel_for(mode):
    if mode == "calibrées+audit":
        return fairness_adjusted(reliabilities_for("calibrées", test, calib), calib)
    return reliabilities_for(mode, test, calib)


@st.cache_resource(show_spinner=False)
def build(method: str, mode: str, scen_key: str, n_cand: int, seed: int):
    rel = rel_for(mode)
    cap = None
    if method == "Choquet":
        fused = DecisionSystem("t", WeightedFuzzyMean(), None, weights, rel).fuse_scenario(cal)
        cap = fit_interactions(np.array([[f.crisp for f in r] for r in fused]), cal.utilities, weights)
    sys_ = make_system(method, weights, rel, capacity=cap)
    rc = sys_.evaluate(cal)
    sys_.supervisor = Supervisor.fit(rc.conflicts, rc.ignorances, cal.hard_cases)
    return sys_


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Classement & explications", "Comparer les opérateurs",
                                        "Atelier de fusion (Zadeh)", "Calibration des agents", "Éthique"])

# ---------------------------------------------------------------- 1. classement
with tab1:
    system = build(method, mode, scen_key, n_cand, seed)
    res = system.evaluate(test)
    rows = []
    for i in res.order:
        r = res.results[i]
        rows.append({"Rang": r.rank, "Candidat": f"#{r.cid}", "Score": round(r.score, 3),
                     "Croyance": round(r.low, 2), "Plausibilité": round(r.high, 2),
                     "Conflit": round(r.conflict, 2), "Ignorance": round(r.ignorance, 2),
                     "Jury": "⚠ renvoyé" if r.flagged else "",
                     "Rang vrai (caché en réel)": int((test.utilities > test.utilities[i]).sum() + 1)})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("τ de Kendall vs vérité", f"{kendall(res.scores, test.utilities):.2f}")
    c2.metric("Précision top-5", f"{top_k_precision(res.scores, test.utilities, 5):.0%}")
    c3.metric("Renvoyés au jury", f"{int(res.flagged.sum())}/{test.n}")
    c4.metric("Seuils (conflit / ignorance)", f"{system.supervisor.tau_conflict:.2f} / {system.supervisor.tau_ignorance:.2f}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    pick = st.selectbox("Explication détaillée du candidat", [f"#{test.candidates[i].cid}" for i in res.order])
    idx = next(i for i, c in enumerate(test.candidates) if f"#{c.cid}" == pick)
    st.code(res.explain(idx), language=None)
    with st.expander("Avis bruts des agents pour ce candidat"):
        names = test.agent_names()
        st.dataframe(pd.DataFrame([{"Agent": names[a], "Critère": CRITERIA_LABELS[CRITERIA[j]],
                                    "Masse": str(o.mass.to_dict(2)), "Nombre flou": str(o.tfn),
                                    "Classe BetP": "—" if o.abstained else o.label,
                                    "Abstention": o.abstained}
                                   for (a, j), o in sorted(test.opinions[idx].items())]),
                     hide_index=True, width="stretch")
    st.caption(f"Fiabilités utilisées ({mode}) : " +
               ", ".join(f"{k} = {v:.2f}" for k, v in system.reliabilities.items()))

# ---------------------------------------------------------------- 2. comparaison
with tab2:
    st.write("Tous les opérateurs, même scénario, mêmes fiabilités.")
    rows, tops = [], {}
    for m in METHOD_NAMES:
        r = build(m, mode, scen_key, n_cand, seed).evaluate(test)
        rows.append({"Méthode": m, "τ Kendall": round(kendall(r.scores, test.utilities), 3),
                     "Top-5": top_k_precision(r.scores, test.utilities, 5),
                     "Renvoyés": int(r.flagged.sum()), "Temps (ms)": round(r.elapsed * 1000, 1)})
        tops[m] = [f"#{test.candidates[i].cid}" for i in r.order[:5]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    truth = [f"#{test.candidates[i].cid}" for i in np.argsort(-test.utilities)[:5]]
    st.write("**Top-5 par méthode** (vérité : " + ", ".join(truth) + ")")
    st.dataframe(pd.DataFrame(tops, index=[f"{k + 1}" for k in range(5)]), width="stretch")

# ---------------------------------------------------------------- 3. atelier
with tab3:
    st.write("Saisissez les masses de deux agents sur Θ = {Faible, Moyen, Bon, Excellent} et observez "
             "la réaction de chaque règle. Le préréglage reproduit le **paradoxe de Zadeh**.")
    zadeh = st.checkbox("Préréglage Zadeh (deux agents quasi certains et opposés)", value=True)
    cols = st.columns(2)
    masses = []
    for k, col in enumerate(cols):
        with col:
            st.subheader(f"Agent {k + 1}")
            default = ([0.0, 0.01, 0.0, 0.99] if k == 0 else [0.99, 0.01, 0.0, 0.0]) if zadeh else [0.1, 0.2, 0.4, 0.2]
            vals = [st.slider(f"m({c})", 0.0, 1.0, float(default[i]), 0.01, key=f"m{k}{i}") for i, c in enumerate(CLASSES)]
            theta = max(0.0, 1.0 - sum(vals))
            st.caption(f"m(Θ) = {theta:.2f} (reste)")
            if sum(vals) > 1:
                st.error("La somme des masses dépasse 1.")
            m = np.zeros(16)
            for i, v in enumerate(vals):
                m[1 << i] = v
            m[15] = theta
            masses.append(m / m.sum())
    st.metric("Degré de conflit K", f"{conflict_degree(masses):.4f}")
    out = []
    for name, f in [("Dempster", dempster), ("Yager", yager), ("Dubois-Prade", dubois_prade), ("Murphy", murphy)]:
        try:
            mf = MassFunction(f(masses))
            out.append({"Règle": name, "Masse fusionnée": str(mf.to_dict(3)),
                        "BetP": ", ".join(f"{c}={p:.2f}" for c, p in zip(CLASSES, mf.betp())),
                        "Utilité E[BetP]": round(mf.expected_utility(), 3),
                        "m(Θ)": round(mf.ignorance, 3)})
        except TotalConflictError:
            out.append({"Règle": name, "Masse fusionnée": "non définie (K = 1)"})
    t = mass_to_tfn(MassFunction((masses[0] + masses[1]) / 2))
    out.append({"Règle": "Moyenne floue", "Masse fusionnée": f"TFN {t}", "Utilité E[BetP]": round(t.centroid, 3)})
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4. calibration
with tab4:
    st.write("Calibration mesurée sur un jeu **distinct** (120 candidats, autre graine).")
    st.dataframe(pd.DataFrame([{"Agent": c.name, "Exactitude": round(c.accuracy, 3), "Brier": round(c.brier, 3),
                                "ECE": round(c.ece, 3), "Confiance moy.": round(c.mean_confidence, 3),
                                "Surconfiant": c.overconfident, "Écart groupes B−A": round(c.group_gap, 3),
                                "Biais détecté": c.biased,
                                "Fiabilité déclarée": next(a.profile.declared_reliability for a in cal.agents if a.name == c.name),
                                "Fiabilité calibrée": round(c.reliability, 3)} for c in calib.values()]),
                 hide_index=True, width="stretch")
    agent = st.selectbox("Diagramme de fiabilité", list(calib))
    b = calib[agent].bins
    st.line_chart(pd.DataFrame({"exactitude observée": [x[1] for x in b], "calibration parfaite": [x[0] for x in b]},
                               index=[round(x[0], 2) for x in b]))

# ---------------------------------------------------------------- 5. éthique
with tab5:
    st.markdown("""
* **AI Act (UE 2024/1689)** : les systèmes d'IA de recrutement/sélection sont classés **à haut risque**
  (annexe III, point 4) : gestion des risques, gouvernance des données, documentation, journalisation,
  transparence, **contrôle humain** (art. 14), exactitude et robustesse.
* **Maroc — loi 09-08 (CNDP)** : déclaration/autorisation préalable du traitement, information et
  consentement des candidats, finalité déterminée, minimisation, droits d'accès, de rectification et d'opposition.
* Ce démonstrateur n'utilise **que des données synthétiques** ; l'attribut protégé (groupe A/B) n'est jamais
  vu par les opérateurs et ne sert qu'à l'audit d'équité.
* Les candidats à fort conflit ou forte ignorance sont **renvoyés au jury**, le système ne tranche pas seul.
""")
