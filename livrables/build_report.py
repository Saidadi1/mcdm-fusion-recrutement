"""Génère le rapport (≤ 30 pages) : livrables/Rapport_Fusion_Avis_MCDM.docx

Toutes les formules sont des équations Word natives (OMML) produites depuis LaTeX (voir omml.py) ;
tous les chiffres sont lus dans results/.

    python livrables/build_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from omml import add_equation, add_inline_math
from fusionrh.aggregation import DEFAULT_RULES, rule_text
from fusionrh.belief import MassFunction, conflict_degree, dempster, dubois_prade, label_of, murphy, yager
from fusionrh.calibration import calibrate_agents, reliabilities_for
from fusionrh.config import AHP_EXPERT_MATRIX, BWM_BEST_TO_OTHERS, BWM_OTHERS_TO_WORST, CLASSES, CRITERIA, CRITERIA_LABELS
from fusionrh.fuzzy import TFN, importance_owa_weights, mass_to_tfn, q_most, weighted_tfn_mean
from fusionrh.generator import generate_scenario, make_config
from fusionrh.pipeline import EVIDENTIAL_METHODS, FUZZY_METHODS, METHOD_NAMES, Supervisor, make_system
from fusionrh.weighting import ahp_weights

RES, FIG, OUT = ROOT / "results", ROOT / "figures", ROOT / "livrables"
BLUE = RGBColor(0x1C, 0x5C, 0xAB)
INK2 = RGBColor(0x52, 0x51, 0x4E)

# =============================================================================
# DONNÉES
# =============================================================================
main = pd.read_csv(RES / "principal.csv")
cal_main = main[main["mode"] == "calibrées"]
sweep = pd.read_csv(RES / "balayage_conflit.csv")
zadeh = pd.read_csv(RES / "zadeh.csv")
rob = pd.read_csv(RES / "robustesse.csv")
sens = pd.read_csv(RES / "sensibilite.csv")
eq = pd.read_csv(RES / "equite.csv")
pool = pd.read_csv(RES / "equite_cumulee.csv")
cost = pd.read_csv(RES / "cout.csv")
calag = pd.read_csv(RES / "calibration_agents.csv")
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
SCEN = sorted(main["scenario"].unique())
P = S["poids"]
stats = S["tests_statistiques"]


def f2(x):
    return f"{x:.2f}".replace(".", ",")


def f3(x):
    return f"{x:.3f}".replace(".", ",")


def pct(x):
    return f"{100 * x:.0f} %"


def scen(prefix):
    return next(s for s in SCEN if s.startswith(prefix))


kt = cal_main.pivot_table(index="methode", columns="scenario", values="kendall").reindex(METHOD_NAMES)
overall = cal_main.groupby("methode").mean(numeric_only=True).reindex(METHOD_NAMES)
best = overall["kendall"].idxmax()
f1_by = cal_main.groupby("methode")["renvoi_f1"].mean().reindex(METHOD_NAMES)
f1_sc = cal_main.pivot_table(index="methode", columns="scenario", values="renvoi_f1").reindex(METHOD_NAMES)
auc_c = cal_main.groupby("methode")["auc_conflit"].mean().reindex(METHOD_NAMES)
auc_i = cal_main.groupby("methode")["auc_ignorance"].mean().reindex(METHOD_NAMES)
evid_f1, fuzzy_f1 = f1_by[EVIDENTIAL_METHODS].mean(), f1_by[FUZZY_METHODS].mean()
modes = main.pivot_table(index="methode", columns="mode", values="kendall").reindex(METHOD_NAMES)
s3 = main[main["scenario"].str.startswith("S3")].pivot_table(index="methode", columns="mode", values="kendall").reindex(METHOD_NAMES)
gain_s3 = s3["calibrées"] - s3["déclarées"]
robm = rob.groupby("methode").mean(numeric_only=True).reindex(METHOD_NAMES)
rob_sc = rob.pivot_table(index="methode", columns="scenario", values="stab_top3_retrait_agent").reindex(METHOD_NAMES)
sens20 = sens[sens["delta"].abs() == 0.2].groupby("methode")["kendall_vs_base"].mean().reindex(METHOD_NAMES)
sens10 = sens[sens["delta"].abs() == 0.1].groupby("methode")["kendall_vs_base"].mean().reindex(METHOD_NAMES)
eq5 = eq[eq["scenario"].str.startswith("S5")]
cf_piv = eq5.pivot_table(index="methode", columns="mode", values="cf_part_changee").reindex(METHOD_NAMES)
dr_piv = eq5.pivot_table(index="methode", columns="mode", values="cf_delta_rang_abs").reindex(METHOD_NAMES)
kq_piv = eq5.pivot_table(index="methode", columns="mode", values="kendall").reindex(METHOD_NAMES)
truth_ratio = eq5["ratio_impact_verite"].mean()
pool["BA"] = pool["taux_B"] / pool["taux_A"]
truth_ba = float(pool.loc[(pool.methode == "Vérité") & pool.scenario.str.startswith("S5"), "BA"].iloc[0])
ba5 = pool[pool.scenario.str.startswith("S5") & (pool.methode != "Vérité")].pivot_table(index="methode", columns="mode", values="BA").reindex(METHOD_NAMES)
ba1 = pool[pool.scenario.str.startswith("S1") & (pool.methode != "Vérité")].pivot_table(index="methode", columns="mode", values="BA").reindex(METHOD_NAMES)
raw5 = pool[pool.scenario.str.startswith("S5") & (pool.methode != "Vérité")].pivot_table(index="methode", columns="mode", values="ratio_impact_cumule").reindex(METHOD_NAMES)
cost_big = cost[(cost.n_agents == 5) & (cost.n_candidats == 320)].set_index("methode")["temps_ms"].reindex(METHOD_NAMES)
sw = sweep.pivot_table(index="extra_fault", columns="methode", values="kendall")[METHOD_NAMES]
sw_f1 = sweep.pivot_table(index="extra_fault", columns="methode", values="renvoi_f1")[METHOD_NAMES]

# =============================================================================
# MISE EN PAGE
# =============================================================================
doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
st.paragraph_format.line_spacing = 1.08
for lvl, size in ((1, 15), (2, 12.5), (3, 11)):
    h = doc.styles[f"Heading {lvl}"]
    h.font.name = "Calibri"; h.font.size = Pt(size); h.font.color.rgb = BLUE; h.font.bold = True
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before = Pt(12 if lvl == 1 else 8); h.paragraph_format.space_after = Pt(4)

FIGN, TABN, EQN = [0], [0], [0]
TOKEN = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*|//[^/]+//|`[^`]+`)")


def _runs(p, text, size=None):
    """**gras**, //italique//, `code`, $LaTeX en ligne$."""
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            add_inline_math(p, tok[1:-1])
            continue
        if tok.startswith("**"):
            r = p.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("//") and tok.endswith("//") and len(tok) > 4:
            r = p.add_run(tok[2:-2]); r.italic = True
        elif tok.startswith("`"):
            r = p.add_run(tok[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        else:
            r = p.add_run(tok)
        if size:
            r.font.size = Pt(size)
    return p


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, color=None, space_after=None, keep=False):
    p = doc.add_paragraph(); p.alignment = align
    _runs(p, text, size)
    if color:
        for r in p.runs:
            r.font.color.rgb = color
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if keep:
        p.paragraph_format.keep_with_next = True
    return p


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style); p.paragraph_format.space_after = Pt(2)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _runs(p, it)


def E(latex):
    """Équation numérotée ; renvoie son numéro."""
    EQN[0] += 1
    add_equation(doc, latex, EQN[0])
    return EQN[0]


def definition(title, text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run(title + ". "); r.bold = True; r.font.color.rgb = BLUE
    _runs(p, text)


def H(text, lvl=1):
    doc.add_heading(text, lvl)


def shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_)
    tcPr.append(shd)


def table(header, rows, widths=None, size=8.5, caption=None):
    if caption:
        TABN[0] += 1
        c = para(f"Tableau {TABN[0]} — {caption}", align=WD_ALIGN_PARAGRAPH.LEFT, size=9, color=INK2, space_after=2, keep=True)
        c.runs[0].bold = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]; cell.text = ""
        _runs(cell.paragraphs[0], str(h), size)
        for r in cell.paragraphs[0].runs:
            r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255)
        shade(cell, "1C5CAB")
    for k, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            _runs(cells[i].paragraphs[0], str(v), size)
            if k % 2:
                shade(cells[i], "F0F4FA")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    for k, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if k == 0:
            th = OxmlElement("w:tblHeader"); trPr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return t


def figure(name, caption, width=16.0):
    FIGN[0] += 1
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(FIG / name), width=Cm(width))
    p.paragraph_format.keep_with_next = True
    c = para(f"Figure {FIGN[0]} — {caption}", align=WD_ALIGN_PARAGRAPH.CENTER, size=9, color=INK2)
    c.runs[0].bold = True
    return FIGN[0]


def code(lines, size=8):
    for line in lines.strip("\n").split("\n"):
        p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.6)
        r = p.add_run(line); r.font.name = "Consolas"; r.font.size = Pt(size)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def field(run, instr):
    for tag, txt in (("begin", None), (None, instr), ("separate", None)):
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
        run._r.append(el)


def end_field(run):
    el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), "end"); run._r.append(el)


def add_page_numbers():
    p = doc.sections[0].footer.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(); field(r, "PAGE"); end_field(r); r.font.size = Pt(9)


def toc():
    p = doc.add_paragraph(); r = p.add_run(); field(r, 'TOC \\o "1-2" \\h \\z \\u')
    r2 = p.add_run("Clic droit → « Mettre à jour les champs » pour afficher la table des matières.")
    r2.italic = True; end_field(r2)


# =============================================================================
# EXEMPLE DÉTAILLÉ (calculé)
# =============================================================================
EX_AGENTS = [
    ("Manager technique", {"Bon": 0.6, ("Bon", "Excellent"): 0.3, "Theta": 0.1}, 0.9),
    ("Test psychométrique", {"Moyen": 0.5, "Bon": 0.3, "Theta": 0.2}, 0.7),
    ("Analyse CV", {"Excellent": 0.8, "Bon": 0.15, "Theta": 0.05}, 0.4),
]
ex_raw = [MassFunction.from_dict(d) for _, d, _ in EX_AGENTS]
ex_alpha = [a for _, _, a in EX_AGENTS]
ex_disc = [m.discount(a) for m, a in zip(ex_raw, ex_alpha)]
ex_K = conflict_degree([m.m for m in ex_disc])
ex_res = {n: MassFunction(f([m.m for m in ex_disc])) for n, f in
          [("Dempster", dempster), ("Yager", yager), ("Dubois-Prade", dubois_prade), ("Murphy", murphy)]}
ex_tfns = [mass_to_tfn(m) for m in ex_raw]
ex_mean = weighted_tfn_mean(ex_tfns, ex_alpha)
_order = np.argsort([-t.centroid for t in ex_tfns])
ex_owa_w = importance_owa_weights([ex_alpha[k] for k in _order], q_most)
_arr = np.array([ex_tfns[k].as_tuple() for k in _order])
ex_owa = TFN(*(ex_owa_w[:, None] * _arr).sum(axis=0))


def mass_str(m: MassFunction, k=4):
    items = sorted(m.focal(), key=lambda t: -t[1])[:k]
    return " ; ".join(f"{label_of(a)} : {f2(v)}" for a, v in items if round(v, 2) > 0)


# =============================================================================
# PAGE DE GARDE
# =============================================================================
for _ in range(5):
    doc.add_paragraph()
para("Rapport de projet — Intelligence artificielle et aide à la décision", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, color=INK2)
p = para("Système multi-agents d'aide à la sélection de candidats fondé sur le MCDM", align=WD_ALIGN_PARAGRAPH.CENTER, size=22, color=BLUE)
p.runs[0].bold = True
para("Étude comparative des opérateurs de fusion des avis (Dempster-Shafer, Yager, fusion floue), avec calibration "
     "des agents et évaluation sur scénarios synthétiques", align=WD_ALIGN_PARAGRAPH.CENTER, size=12.5, color=INK2)
for _ in range(4):
    doc.add_paragraph()
para("Question de recherche : //quel opérateur de fusion produit le classement de candidats le plus fiable, le plus "
     "robuste au conflit, le plus équitable et le plus explicable, et dans quelles conditions ?//",
     align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
for _ in range(7):
    doc.add_paragraph()
para("Réalisé par : ……………………………………", align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
para("Encadré par : ……………………………………", align=WD_ALIGN_PARAGRAPH.CENTER, size=11)
para("Année universitaire 2026-2027", align=WD_ALIGN_PARAGRAPH.CENTER, size=11, color=INK2)
page_break()

# =============================================================================
# RÉSUMÉ
# =============================================================================
H("Résumé", 1)
para(f"Dans un recrutement, plusieurs évaluateurs — RH, manager technique, test psychométrique, analyse du CV, "
     f"assistant fondé sur un grand modèle de langage — produisent des avis **hétérogènes, incertains et "
     f"conflictuels**. Ce travail conçoit un système multi-agents en six couches dans lequel chaque avis est représenté "
     f"à la fois par une fonction de masse sur le cadre de discernement Θ = {{Faible, Moyen, Bon, Excellent}} et par un "
     f"nombre flou triangulaire, les deux représentations étant reliées par une table de correspondance explicite. "
     f"Neuf méthodes de fusion — quatre évidentielles (Dempster, Yager, Dubois-Prade, Murphy) et cinq floues (moyenne "
     f"pondérée floue, OWA « la plupart », OWA « au moins un », intégrale de Choquet 2-additive, inférence de Mamdani) — "
     f"sont placées derrière une interface unique et comparées sur six scénarios synthétiques contrôlés "
     f"({S['reps']} réplications chacun), avec des fiabilités d'agents égales, déclarées, calibrées ou optimisées.")
para(f"Aucun opérateur ne domine sur tous les critères. Pour l'**exactitude**, la famille floue l'emporte : "
     f"{best} atteint un τ de Kendall moyen de {f3(overall.loc[best, 'kendall'])}, contre environ "
     f"{f2(overall.loc[EVIDENTIAL_METHODS, 'kendall'].mean())} pour les règles évidentielles, dont la combinaison "
     f"conjonctive concentre la masse et perd la résolution intra-classe. Pour la **supervision**, c'est l'inverse : "
     f"la famille évidentielle repère bien mieux les cas à renvoyer au jury (F1 {f2(evid_f1)} contre {f2(fuzzy_f1)}) "
     f"grâce à la représentation explicite du conflit et de l'ignorance. L'intégrale de Choquet devient la meilleure "
     f"quand les critères interagissent ; la calibration des fiabilités est décisive face à un agent surconfiant ; "
     f"le TOPSIS classique provoque des inversions de rang ; aucun opérateur n'est équitable par construction et "
     f"la règle des quatre cinquièmes, appliquée seule, peut masquer un biais réel. Nous recommandons une "
     f"**architecture hybride** — un opérateur flou pour classer, un indicateur évidentiel pour superviser — "
     f"assortie d'un audit d'équité des agents, conformément aux exigences de l'AI Act européen et de la loi "
     f"marocaine 09-08.")
para("**Mots-clés** : théorie des fonctions de croyance, règle de Dempster, règle de Yager, OWA, intégrale de "
     "Choquet, inférence de Mamdani, TOPSIS, AHP, BWM, calibration, équité algorithmique, AI Act, loi 09-08.", size=9.5)
p = para("**Table des matières**", align=WD_ALIGN_PARAGRAPH.LEFT)
p.runs[0].font.size = Pt(15); p.runs[0].font.color.rgb = BLUE
toc()
page_break()
H("Abréviations et notations", 1)
table(["Symbole / sigle", "Signification"], [
    ["$\\Theta$", "cadre de discernement {Faible, Moyen, Bon, Excellent}"],
    ["$m,\\;\\mathrm{Bel},\\;\\mathrm{Pl}$", "fonction de masse, croyance, plausibilité"],
    ["$\\mathrm{BetP}$", "probabilité pignistique (décision)"],
    ["$K$", "degré de conflit entre sources (masse de l'ensemble vide)"],
    ["$\\alpha$", "fiabilité d'un agent, coefficient d'affaiblissement"],
    ["$\\tilde{a}=(a,b,c)$", "nombre flou triangulaire (TFN)"],
    ["$\\phi_i,\\;I_{ij}$", "indice de Shapley du critère i, indice d'interaction entre i et j"],
    ["$w_j$", "poids du critère j (AHP ou BWM)"],
    ["$\\tau,\\;\\rho$", "coefficients de corrélation de rang de Kendall et de Spearman"],
    ["AHP / BWM", "Analytic Hierarchy Process / Best-Worst Method"],
    ["CR, CI, RI", "ratio, indice de cohérence et indice aléatoire de Saaty"],
    ["TOPSIS", "Technique for Order Preference by Similarity to Ideal Solution"],
    ["OWA", "Ordered Weighted Averaging (moyenne pondérée ordonnée)"],
    ["ECE", "Expected Calibration Error (erreur de calibration attendue)"],
    ["LLM", "Large Language Model (grand modèle de langage)"],
    ["CNDP", "Commission nationale de contrôle de la protection des données à caractère personnel"],
    ["AI Act", "règlement (UE) 2024/1689 sur l'intelligence artificielle"],
], widths=[4.5, 12.0], caption="Abréviations et notations")
page_break()

# =============================================================================
# 1. INTRODUCTION
# =============================================================================
H("1. Introduction", 1)
H("1.1 Contexte", 2)
para("La sélection de candidats est une décision multicritère prise sous incertitude. Pour un même poste, l'entreprise "
     "mobilise plusieurs sources d'évaluation : l'entretien RH apprécie la personnalité et l'adéquation culturelle, le "
     "manager technique juge les compétences et l'expérience, un test psychométrique mesure des aptitudes, une analyse "
     "automatique du CV extrait le parcours, et de plus en plus souvent un assistant fondé sur un grand modèle de langage "
     "(LLM) résume un entretien. Ces avis sont :")
bullets([
    "**hétérogènes** : chaque évaluateur ne couvre qu'une partie des critères et s'exprime sur sa propre échelle ;",
    "**incertains** : l'information est incomplète, et l'évaluateur peut assumer une ignorance (« un CV ne permet pas de "
    "juger les soft skills ») qu'il serait faux de transformer en note moyenne ;",
    "**conflictuels** : le test psychométrique peut contredire l'entretien, un agent peut se tromper ou être biaisé.",
])
para("Le cœur du problème n'est donc pas l'algorithme de classement final, mais la **manière de fusionner ces avis**. "
     "Selon que l'on renormalise le conflit, qu'on le transforme en ignorance, qu'on fasse une moyenne ou qu'on "
     "retienne l'avis le plus favorable, le même candidat peut finir premier ou dixième. Comme l'IA appliquée au "
     "recrutement est classée « à haut risque » par le règlement européen sur l'IA, ce choix doit en outre être fiable, "
     "robuste, équitable et explicable.")
H("1.2 Question de recherche et objectifs", 2)
para("Ce projet répond à la question : //quel opérateur de fusion produit le classement de candidats le plus fiable, "
     "le plus robuste au conflit, le plus équitable et le plus explicable, et dans quelles conditions ?// Les objectifs "
     "sont de maîtriser la théorie des fonctions de croyance et les agrégations floues, de concevoir une architecture "
     "multi-agents où chaque agent exprime une opinion incertaine, de mener une comparaison expérimentale rigoureuse "
     "(vérité de référence, scénarios contrôlés, tests statistiques) et d'intégrer les exigences d'équité et "
     "d'explicabilité propres à une application à haut risque.")
H("1.3 Contributions et plan", 2)
bullets([
    "un **système multi-agents modulaire** en six couches où chaque opérateur de fusion est interchangeable ;",
    "une **double représentation** de chaque avis (masse ⇄ nombre flou) avec une table de correspondance explicite ;",
    "une **calibration** des agents (exactitude, score de Brier, diagramme de fiabilité, surconfiance, audit d'équité) "
    "et une pondération des critères par AHP et BWM avec contrôle de cohérence ;",
    "un **générateur de scénarios** paramétrable avec vérité de référence et variations « toutes choses égales par "
    "ailleurs » (contrefactuels, retrait d'agent, ajout de candidat) ;",
    f"une **campagne expérimentale** de {S['reps']} réplications × 6 scénarios × 4 modes de fiabilité × 9 méthodes, "
    "évaluée sur sept familles de critères avec tests de Friedman et de Wilcoxon-Holm ;",
    "une analyse **éthique et réglementaire** (AI Act, loi 09-08) traduite en mesures techniques.",
])
para("La section 2 situe le travail dans la littérature, la section 3 pose les fondements mathématiques, la section 4 "
     "décrit l'architecture, la section 5 déroule un exemple de fusion complet, la section 6 traite la calibration, la "
     "section 7 le protocole, la section 8 les résultats, la section 9 la discussion et la section 10 la dimension "
     "éthique et réglementaire.")

# =============================================================================
# 2. ÉTAT DE L'ART
# =============================================================================
H("2. État de l'art", 1)
H("2.1 Décision multicritère pour la sélection de personnel", 2)
para("La sélection de personnel est un domaine d'application classique de l'aide multicritère à la décision (MCDM). "
     "L'AHP de Saaty (1980) y est utilisé pour hiérarchiser les critères par comparaisons par paires, et le BWM de "
     "Rezaei (2015) réduit le nombre de comparaisons nécessaires tout en améliorant la cohérence. Le TOPSIS de Hwang et "
     "Yoon (1981) classe les alternatives selon leur proximité à une solution idéale ; sa version floue (Chen, 2000) "
     "est très utilisée lorsque les évaluations sont linguistiques. Ces travaux supposent cependant le plus souvent "
     "que la matrice de décision est donnée : la question de la **construction** de cette matrice à partir d'avis "
     "multiples, incomplets et contradictoires y est peu traitée — c'est précisément l'objet de ce projet.")
H("2.2 Fusion d'informations incertaines", 2)
para("La théorie des fonctions de croyance (Dempster, 1967 ; Shafer, 1976) distingue l'incertitude de l'ignorance en "
     "autorisant des masses sur des ensembles d'hypothèses. Le modèle des croyances transférables de Smets et Kennes "
     "(1994) sépare le niveau « crédal », où l'on combine, du niveau « pignistique », où l'on décide. La règle de "
     "Dempster a été vivement critiquée par Zadeh (1986) pour son comportement contre-intuitif en cas de fort conflit, "
     "ce qui a suscité de nombreuses alternatives : transfert du conflit vers l'ignorance (Yager, 1987), vers l'union "
     "des hypothèses en désaccord (Dubois et Prade, 1988), ou moyenne préalable des sources (Murphy, 2000). La prise "
     "en compte de la fiabilité des sources par affaiblissement remonte à Shafer ; Mercier, Quost et Denœux (2008) en "
     "ont proposé des versions contextuelles. Du côté flou, les opérateurs OWA de Yager (1988) et leur pilotage par "
     "quantificateurs linguistiques (Yager, 1996) permettent de modéliser une attitude (« la plupart des agents »), "
     "tandis que l'intégrale de Choquet, avec les capacités k-additives de Grabisch (1997), modélise les interactions "
     "entre critères. Les systèmes d'inférence de Mamdani (1975) offrent une représentation par règles lisibles.")
H("2.3 Calibration, équité et explicabilité", 2)
para("La fiabilité d'un évaluateur probabiliste se mesure par le score de Brier (1950) et les diagrammes de "
     "fiabilité ; Guo et al. (2017) ont montré que les modèles d'apprentissage modernes sont souvent surconfiants, ce "
     "qui concerne directement les agents LLM. En recrutement, l'équité est classiquement contrôlée par la règle des "
     "quatre cinquièmes (Uniform Guidelines, 1978) et, plus récemment, par l'équité contrefactuelle (Kusner et al., "
     "2017). Enfin, la comparaison statistique de méthodes sur des jeux multiples suit les recommandations de Demšar "
     "(2006) : test de Friedman puis comparaisons post hoc corrigées.")
para("**Positionnement.** À notre connaissance, peu de travaux comparent de façon contrôlée les opérateurs "
     "évidentiels et flous **sous une même interface**, avec la même calibration des sources, une vérité de référence "
     "connue et des critères d'équité et d'explicabilité. C'est l'apport principal de ce projet.")

# =============================================================================
# 3. FONDEMENTS THÉORIQUES
# =============================================================================
H("3. Fondements théoriques", 1)
H("3.1 Fonctions de croyance", 2)
definition("Définition 1 (cadre de discernement)", r"Le cadre $\Theta=\{\mathrm{Faible},\mathrm{Moyen},\mathrm{Bon},\mathrm{Excellent}\}$ "
           r"est l'ensemble exhaustif des niveaux possibles d'un candidat sur un critère ; $2^{\Theta}$ désigne l'ensemble "
           r"de ses $16$ sous-ensembles.")
definition("Définition 2 (fonction de masse)", r"Une fonction de masse est une application $m:2^{\Theta}\to[0,1]$ telle que :")
E(r"\sum_{A\subseteq\Theta} m(A)=1")
para(r"Un sous-ensemble $A$ tel que $m(A)>0$ est un **élément focal**. Une masse sur un singleton exprime un avis "
     r"précis, une masse sur un sous-ensemble comme $\{\mathrm{Bon},\mathrm{Excellent}\}$ un avis imprécis, et "
     r"$m(\Theta)$ l'**ignorance** assumée. La masse vide $m(\Theta)=1$ représente une abstention totale.")
definition("Définition 3 (croyance, plausibilité)", r"La croyance $\mathrm{Bel}(A)$ est la masse qui implique $A$ ; "
           r"la plausibilité $\mathrm{Pl}(A)$ est la masse compatible avec $A$ :")
E(r"\mathrm{Bel}(A)=\sum_{\emptyset\neq B\subseteq A} m(B)\text{,}\qquad \mathrm{Pl}(A)=\sum_{B\cap A\neq\emptyset} m(B)=1-\mathrm{Bel}(\bar{A})")
para(r"L'intervalle $[\mathrm{Bel}(A),\mathrm{Pl}(A)]$ encadre toute probabilité compatible avec l'avis ; sa largeur "
     r"mesure l'imprécision.")
definition("Définition 4 (probabilité pignistique)", r"Pour décider, la masse est transformée en probabilité "
           r"(Smets) en partageant chaque masse également entre les éléments de l'ensemble focal :")
n_betp = E(r"\mathrm{BetP}(\theta)=\frac{1}{1-m(\emptyset)}\sum_{A\ni\theta}\frac{m(A)}{|A|}")
para(r"Avec les utilités $u=(0,\,1/3,\,2/3,\,1)$ associées aux quatre classes, on obtient une évaluation "
     r"nette par critère et ses bornes inférieure et supérieure (espérances de Dempster), qui forment l'**intervalle "
     r"croyance–plausibilité** restitué au jury :")
E(r"\mathbb{E}[u]=\sum_{\theta\in\Theta}\mathrm{BetP}(\theta)\,u(\theta)")
E(r"u_{\mathrm{inf}}=\sum_{A\subseteq\Theta} m(A)\min_{\theta\in A}u(\theta)\text{,}\qquad u_{\mathrm{sup}}=\sum_{A\subseteq\Theta} m(A)\max_{\theta\in A}u(\theta)")
definition("Définition 5 (affaiblissement)", r"La fiabilité $\alpha\in[0,1]$ d'une source est intégrée par "
           r"l'affaiblissement de Shafer, qui transfère la part non fiable vers l'ignorance :")
n_disc = E(r"m^{\alpha}(A)=\alpha\, m(A)\;\;(A\neq\Theta)\text{,}\qquad m^{\alpha}(\Theta)=1-\alpha+\alpha\, m(\Theta)")
para(r"Une source totalement non fiable ($\alpha=0$) devient la masse vide, élément neutre de la combinaison.")

H("3.2 Règles de combinaison et conflit", 2)
para(r"La combinaison **conjonctive** de deux sources indépendantes affecte le produit des masses à l'intersection "
     r"des éléments focaux ; la masse affectée à l'ensemble vide est le **degré de conflit** $K$ :")
E(r"m_{\cap}(C)=\sum_{A\cap B=C} m_1(A)\,m_2(B)\text{,}\qquad K=m_{\cap}(\emptyset)=\sum_{A\cap B=\emptyset} m_1(A)\,m_2(B)")
para("Les quatre règles comparées diffèrent uniquement par le traitement de K :")
para("**Dempster** — le conflit est éliminé par renormalisation (règle associative et commutative) :", keep=True)
E(r"m_{D}(C)=\frac{m_{\cap}(C)}{1-K}\quad(C\neq\emptyset)\text{,}\qquad m_{D}(\emptyset)=0")
para("**Yager (1987)** — le conflit est interprété comme de l'ignorance et transféré vers Θ :", keep=True)
E(r"m_{Y}(C)=m_{\cap}(C)\quad(\emptyset\neq C\neq\Theta)\text{,}\qquad m_{Y}(\Theta)=m_{\cap}(\Theta)+K")
para("**Dubois-Prade (1988)** — le produit de deux avis disjoints est affecté à leur union (« l'un ou l'autre ») :", keep=True)
E(r"m_{DP}(C)=m_{\cap}(C)+\sum_{A\cup B=C,\;A\cap B=\emptyset} m_1(A)\,m_2(B)")
para(r"**Murphy (2000)** — les $n$ masses sont moyennées puis la moyenne est combinée $n-1$ fois avec elle-même par la "
     r"règle de Dempster :", keep=True)
E(r"\bar{m}=\frac{1}{n}\sum_{i=1}^{n} m_i\text{,}\qquad m_{M}=\bar{m}\oplus\bar{m}\oplus\cdots\oplus\bar{m}\quad(n\text{ termes})")
para(r"La règle de Dubois-Prade n'étant pas associative, elle est appliquée séquentiellement dans l'ordre des agents "
     r"pour $n>2$ sources. **Paradoxe de Zadeh** : si l'agent 1 donne $m_1(\mathrm{Excellent})=0{,}99$, "
     r"$m_1(\mathrm{Moyen})=0{,}01$ et l'agent 2 $m_2(\mathrm{Faible})=0{,}99$, $m_2(\mathrm{Moyen})=0{,}01$, alors "
     r"$K=0{,}9999$ et Dempster conclut $m_D(\mathrm{Moyen})=1$ : une certitude sur l'hypothèse que les deux agents "
     r"jugeaient quasi impossible.")

H("3.3 Agrégations floues", 2)
definition("Définition 6 (nombre flou triangulaire)", r"Un TFN $\tilde{a}=(a,b,c)$, $a\le b\le c$, a pour fonction "
           r"d'appartenance, centroïde et $\alpha$-coupe :")
E(r"\mu_{\tilde{a}}(x)=\max\left(0,\;\min\left(\frac{x-a}{b-a},\;\frac{c-x}{c-b}\right)\right)")
E(r"G(\tilde{a})=\frac{a+b+c}{3}\text{,}\qquad \tilde{a}_{\alpha}=\left[a+\alpha(b-a),\;c-\alpha(c-b)\right]")
para(r"**Moyenne pondérée floue.** Avec les fiabilités $\alpha_i$ comme poids : $\tilde{a}=\sum_i \alpha_i\tilde{a}_i/\sum_i\alpha_i$, "
     r"calculée composante par composante.")
para(r"**OWA de Yager.** Les arguments sont triés par ordre décroissant ($b_k$ est le $k$-ième plus grand) puis pondérés "
     r"selon leur **rang**, et non selon leur source. Les poids dérivent d'un quantificateur linguistique $Q$ ; avec "
     r"des importances $u_k$ (ici les fiabilités), Yager (1996) pose :")
E(r"\mathrm{OWA}(a_1,\ldots,a_n)=\sum_{k=1}^{n} w_k\,b_k")
E(r"w_k=Q\left(\frac{S_k}{T}\right)-Q\left(\frac{S_{k-1}}{T}\right)\text{,}\qquad S_k=\sum_{l\le k}u_l\text{,}\qquad T=S_n")
para(r"Le quantificateur « la plupart » est $Q(r)=\min(1,\max(0,(r-0{,}3)/0{,}5))$ ; « au moins un » est $Q(r)=1$ pour "
     r"$r>0$, ce qui réduit l'OWA au maximum (attitude optimiste).")
definition("Définition 7 (capacité et intégrale de Choquet)", r"Une capacité sur l'ensemble $N$ des critères est une "
           r"fonction $\mu:2^{N}\to[0,1]$ monotone avec $\mu(\emptyset)=0$ et $\mu(N)=1$. L'intégrale de Choquet de "
           r"$x$ s'écrit, avec $x_{(1)}\le\cdots\le x_{(n)}$ et $A_{(i)}=\{(i),\ldots,(n)\}$ :")
E(r"C_{\mu}(x)=\sum_{i=1}^{n}\left(x_{(i)}-x_{(i-1)}\right)\mu\left(A_{(i)}\right)")
para(r"Une capacité générale compte $2^n-2$ paramètres. Une capacité **2-additive** (Grabisch, 1997) n'en compte que "
     r"$n(n+1)/2-1$ et s'écrit avec les indices de Shapley $\phi_i$ et d'interaction $I_{ij}$ :")
n_cho = E(r"C_{\mu}(x)=\sum_{i=1}^{n}\phi_i x_i-\frac{1}{2}\sum_{i<j} I_{ij}\left|x_i-x_j\right|\text{,}\qquad \phi_i-\frac{1}{2}\sum_{j\neq i}\left|I_{ij}\right|\ge 0")
para(r"$I_{ij}>0$ traduit une **complémentarité** (les deux critères doivent être bons ensemble), $I_{ij}<0$ une "
     r"**redondance** (un bon score sur l'un compense l'autre, par exemple expérience et compétences techniques). La "
     r"contrainte de droite garantit la monotonie. Dans notre système, $\phi$ est fixé aux poids AHP et les six "
     r"interactions sont identifiées sur le jeu de calibration par moindres carrés régularisés, puis projetées sur "
     r"les contraintes.")
para(r"**Inférence de Mamdani.** Pour une règle $r$ « SI $x_1$ est $A_1^r$ ET … ALORS $y$ est $B^r$ », le degré "
     r"d'activation est le minimum des appartenances ; les conclusions écrêtées sont agrégées par le maximum puis "
     r"défuzzifiées par centroïde :")
E(r"\beta_r=\min_{j}\mu_{A_j^r}(x_j)\text{,}\qquad \mu(y)=\max_{r}\min\left(\beta_r,\mu_{B^r}(y)\right)\text{,}\qquad y^{*}=\frac{\int y\,\mu(y)\,dy}{\int\mu(y)\,dy}")

H("3.4 Pondération et classement multicritère", 2)
para(r"**AHP.** À partir d'une matrice de comparaisons par paires $A=(a_{ij})$ réciproque ($a_{ji}=1/a_{ij}$), les poids "
     r"sont le vecteur propre principal ; la cohérence est contrôlée par le ratio $\mathrm{CR}$, avec l'indice "
     r"aléatoire $\mathrm{RI}(4)=0{,}90$, et l'on exige $\mathrm{CR}<0{,}1$ :")
E(r"A\,w=\lambda_{\mathrm{max}}\,w\text{,}\qquad \mathrm{CI}=\frac{\lambda_{\mathrm{max}}-n}{n-1}\text{,}\qquad \mathrm{CR}=\frac{\mathrm{CI}}{\mathrm{RI}}")
para(r"**BWM** (modèle linéaire de Rezaei). Le décideur compare le meilleur critère $B$ à tous les autres ($a_{Bj}$) et "
     r"tous les autres au pire $W$ ($a_{jW}$) ; on résout le programme linéaire, puis $\xi^{*}/\mathrm{CI}$ mesure la cohérence :")
E(r"\min\;\xi\quad\text{s.c.}\quad\left|w_B-a_{Bj}w_j\right|\le\xi\text{,}\;\;\left|w_j-a_{jW}w_W\right|\le\xi\text{,}\;\;\sum_j w_j=1\text{,}\;\;w_j\ge 0")
para(r"**TOPSIS.** La matrice $x_{ij}$ (candidat $i$, critère $j$) est normalisée et pondérée, puis chaque candidat est "
     r"situé par rapport à la solution idéale $v^{+}$ et anti-idéale $v^{-}$ :")
E(r"v_{ij}=w_j\,\frac{x_{ij}}{\sqrt{\sum_k x_{kj}^{2}}}\text{,}\qquad v_j^{+}=\max_i v_{ij}\text{,}\qquad v_j^{-}=\min_i v_{ij}")
E(r"d_i^{\pm}=\sqrt{\sum_j\left(v_{ij}-v_j^{\pm}\right)^{2}}\text{,}\qquad CC_i=\frac{d_i^{-}}{d_i^{+}+d_i^{-}}")
para(r"Le **fuzzy TOPSIS** de Chen (2000) applique la même démarche à des TFN normalisés par le maximum des $c_{ij}$, "
     r"avec la distance « vertex » :")
E(r"d(\tilde{a},\tilde{b})=\sqrt{\frac{1}{3}\left[(a_1-a_2)^2+(b_1-b_2)^2+(c_1-c_2)^2\right]}")
para(r"Comme la normalisation et l'idéal dépendent de l'ensemble des candidats, TOPSIS est sujet à l'**inversion de rang**.")

H("3.5 Propriétés comparées des opérateurs", 2)
para("Le tableau ci-dessous résume les propriétés mathématiques qui conditionnent le comportement des opérateurs "
     "dans les expériences. L'**idempotence** (fusionner un avis avec lui-même le laisse inchangé) évite de "
     "renforcer artificiellement une information répétée ; l'**élément neutre** indique comment une abstention est "
     "absorbée ; le **renforcement** traduit le fait que deux avis concordants augmentent la certitude.")
table(["Opérateur", "Commut.", "Assoc.", "Idempot.", "Élément neutre", "Renforce", "Traitement du conflit"], [
    ["Dempster", "oui", "oui", "non", "masse vide", "oui", "renormalisé (masqué)"],
    ["Yager", "oui", "non*", "non", "masse vide", "oui", "transféré vers Θ"],
    ["Dubois-Prade", "oui", "non", "non", "masse vide", "oui", "transféré vers A ∪ B"],
    ["Murphy", "oui", "—", "oui", "—", "faible", "dilué par la moyenne"],
    ["Moyenne floue", "oui", "—", "oui", "aucun", "non", "moyenné (masqué)"],
    ["OWA (la plupart)", "oui", "—", "oui", "aucun", "non", "avis extrêmes écartés"],
    ["OWA (au moins un)", "oui", "oui", "oui", "aucun", "non", "avis maximal retenu"],
    ["Choquet", "—", "—", "oui", "—", "—", "inter-critères"],
    ["Mamdani", "—", "—", "—", "—", "—", "inter-critères (règles)"],
], widths=[3.1, 1.8, 1.8, 1.8, 2.3, 2.0, 3.7], size=8, caption="Propriétés des opérateurs de fusion")
para("(*) La règle de Yager devient associative si l'on combine d'abord toutes les sources par la règle conjonctive "
     "puis que l'on transfère le conflit global vers Θ, ce que fait notre implémentation. Les opérateurs évidentiels "
     "**renforcent** l'avis commun (deux agents qui disent « Bon » donnent plus de certitude qu'un seul), alors que "
     "les opérateurs flous, idempotents, **moyennent** : c'est l'origine de leurs comportements opposés observés en "
     "exactitude et en détection des cas difficiles.")

# =============================================================================
# 4. ARCHITECTURE
# =============================================================================
H("4. Architecture et conception du système", 1)
figure("fig_architecture.png", "Architecture en six couches du système multi-agents", 12.5)
H("4.1 Couche 1 — agents évaluateurs", 2)
para("Cinq agents (le sujet en demande trois à cinq) couvrent chacun leur domaine. Ils sont simulés par des règles "
     "bruitées dont les paramètres sont documentés dans `docs/GENERATEUR.md`. L'agent « LLM entretien (simulé) » "
     "reproduit deux défauts typiques d'un grand modèle de langage : une **variabilité d'une exécution à l'autre** et "
     "une **surconfiance** (masses très tranchées). Un agent LLM réel peut être branché en implémentant la même "
     "méthode `evaluate`.")
table(["Agent", "Critères couverts", "σ (bruit)", "γ", "u = m(Θ)", "α déclarée", "Particularité"], [
    ["RH (entretien)", "soft skills, adéquation", "0,10", "1,0", "0,15", "0,80", "—"],
    ["Manager technique", "technique, expérience, adéquation", "0,08", "1,5", "0,10", "0,90", "le plus précis"],
    ["Test psychométrique", "soft skills, technique", "0,12", "1,0", "0,20", "0,70", "ignorance élevée"],
    ["Analyse CV", "expérience, technique", "0,15", "1,2", "0,15", "0,60", "surconfiant (S3), biaisé (S5)"],
    ["LLM entretien (simulé)", "les 4 critères", "0,12", "2,5", "0,05", "0,85", "surconfiant, instable"],
], widths=[3.2, 3.9, 1.3, 0.9, 1.5, 1.6, 4.1], caption="Agents évaluateurs et leurs paramètres")
H("4.2 Couche 2 — construction des avis et table de correspondance", 2)
para(r"Un agent perçoit une valeur latente $o\in[0,1]$. Ses appartenances de Ruspini $\mu_k(o)$ aux deux classes "
     r"voisines $k$ et $k+1$ (qui somment à 1) sont accentuées par un exposant $\gamma$ (surconfiance si $\gamma>1$), "
     r"puis la masse est répartie entre singletons, paire adjacente et ignorance selon la précision $q$ et l'ignorance $u$ :")
E(r"p_k=\frac{\mu_k(o)^{\gamma}}{\mu_k(o)^{\gamma}+\mu_{k+1}(o)^{\gamma}}")
E(r"m(\{k\})=(1-u)\,q\,p_k\text{,}\qquad m(\{k,k+1\})=(1-u)(1-q)\text{,}\qquad m(\Theta)=u")
para("La forme floue est obtenue par la table de correspondance du tableau 3, bijective sur les singletons et sur Θ. "
     "Un sous-ensemble est d'autant plus large qu'il est imprécis, et l'ignorance totale est le nombre flou le plus "
     "large possible.")
table(["Élément focal", "Nombre flou associé", "Justification"], [
    ["Faible", "$(0,\\,0,\\,1/3)$", "partition de Ruspini de [0, 1]"],
    ["Moyen", "$(0,\\,1/3,\\,2/3)$", "idem"],
    ["Bon", "$(1/3,\\,2/3,\\,1)$", "idem"],
    ["Excellent", "$(2/3,\\,1,\\,1)$", "idem"],
    ["Sous-ensemble A", "$(\\min_{k\\in A}a_k,\\;\\bar{b}_A,\\;\\max_{k\\in A}c_k)$", "l'imprécision élargit le support"],
    ["Θ (ignorance)", "$(0,\\,0{,}5,\\,1)$", "nombre flou le plus large"],
    ["Masse m quelconque", "$\\sum_A m(A)\\,\\widetilde{T}(A)$", "« espérance floue » de la masse"],
    ["Inverse TFN → masse", "$m(\\Theta)$ = largeur excédentaire ; reste selon $\\mu(b)$", "approximation, testée par aller-retour"],
], widths=[3.4, 6.6, 6.5], caption="Table de correspondance masse ⇄ nombre flou")
H("4.3 Couches 3 à 6 — interface de fusion, décision et supervision", 2)
para("Tout opérateur hérite d'une classe abstraite commune ; une **méthode** associe un opérateur inter-agents (par "
     "critère) et un agrégateur inter-critères. Changer d'opérateur se fait en une ligne :")
code('''class FusionOperator(ABC):
    @abstractmethod
    def fuse(self, opinions, reliabilities) -> CriterionFusion:
        """valeur nette, bornes [Bel ; Pl], TFN, conflit, ignorance, contributions des agents"""

system  = make_system("Yager", poids_AHP, fiabilites_calibrees)   # ou "Choquet", "OWA (la plupart)"...
ranking = system.evaluate(scenario)          # classement + intervalle + renvoi au jury
print(ranking.explain(i))                    # justification lisible''')
table(["Méthode", "Fusion des agents (par critère)", "Agrégation des critères", "Signal de supervision"], [
    ["Dempster, Yager, Dubois-Prade, Murphy", "règle évidentielle après affaiblissement", "TOPSIS", "conflit K ; largeur Pl − Bel"],
    ["Moyenne floue", "moyenne pondérée de TFN", "fuzzy TOPSIS", "dispersion ; largeur de l'α-coupe"],
    ["OWA (la plupart / au moins un)", "OWA à importances", "fuzzy TOPSIS", "idem"],
    ["Choquet", "moyenne pondérée de TFN", "Choquet 2-additive", "idem"],
    ["Mamdani", "moyenne pondérée de TFN", "inférence de Mamdani", "idem"],
], widths=[4.6, 4.6, 3.3, 4.0], caption="Les neuf méthodes comparées")
para("Le traitement d'un candidat suit l'algorithme ci-dessous. La **supervision humaine** renvoie le candidat au "
     "jury si son conflit maximal par critère ou son ignorance pondérée dépasse un seuil ; les seuils sont **appris "
     "sur le jeu de calibration** (maximisation du F1 de détection des cas difficiles), séparément pour chaque méthode.")
code('''Pour chaque candidat i :
    Pour chaque critère j :
        avis    ← avis des agents qui couvrent j            (masses ou TFN)
        F[i,j]  ← opérateur.fuse(avis, fiabilités α)          (valeur, [Bel ; Pl], K, ignorance)
    score[i], [bas, haut] ← agrégateur(F[i,·], poids AHP)     (TOPSIS / Choquet / Mamdani)
    conflit[i]   ← max_j K[i,j] ;   ignorance[i] ← Σ_j w_j · ignorance[i,j]
    Si conflit[i] > τ_K  ou  ignorance[i] > τ_I  :  RENVOI AU JURY HUMAIN
Classer par score ; produire l'explication de chaque candidat''')

H("4.4 Choix de conception", 2)
bullets([
    "**Fusion par critère, puis agrégation des critères.** Les agents ne couvrent pas tous les critères ; fusionner "
    "critère par critère permet de comparer des avis de même nature et rend la matrice de décision de TOPSIS bien "
    "définie. Choquet et Mamdani, qui modélisent les relations entre critères, remplacent naturellement TOPSIS dans la "
    "seconde étape.",
    "**Affaiblissement avant combinaison.** Dans la famille évidentielle, la fiabilité agit par affaiblissement "
    "(équation " + str(n_disc) + "), ce qui est cohérent avec la sémantique des masses ; dans la famille floue, elle "
    "agit comme poids d'importance.",
    "**Abstentions.** Une abstention est la masse vide pour la famille évidentielle (élément neutre) ; elle est exclue "
    "de la moyenne pour la famille floue, faute d'élément neutre. Ce choix, favorable à l'exactitude floue, a un coût "
    "en supervision que les expériences mesurent (§8.3).",
    "**Seuils de supervision appris.** Les seuils de conflit et d'ignorance n'ont pas la même échelle selon les "
    "familles ; les apprendre sur la calibration pour chaque méthode évite de favoriser l'une d'elles.",
    "**Attribut protégé caché.** L'attribut protégé n'est jamais transmis aux opérateurs ; il n'est utilisé que par le "
    "générateur (pour simuler un biais) et par l'audit d'équité.",
])

# =============================================================================
# 5. EXEMPLE DÉTAILLÉ
# =============================================================================
H("5. Exemple de fusion pas à pas", 1)
para("Pour rendre les règles concrètes, considérons un candidat évalué sur les **compétences techniques** par trois "
     "agents. L'Analyse CV est très affirmative mais peu fiable (calibration α = 0,4).")
table(["Agent", "Masse exprimée", "Fiabilité α", "Masse affaiblie (équation " + str(n_disc) + ")"],
      [[n, mass_str(m), f2(a), mass_str(d)] for (n, _, a), m, d in zip(EX_AGENTS, ex_raw, ex_disc)],
      widths=[3.4, 5.0, 1.8, 6.3], caption="Avis des agents sur un critère")
para(f"La combinaison conjonctive des trois masses affaiblies donne un conflit $K = {f3(ex_K)}$ "
     f"(il serait de {f3(conflict_degree([m.m for m in ex_raw]))} sans affaiblissement). Les quatre règles "
     f"traitent ce conflit différemment :")
rows = []
for n, m in ex_res.items():
    b = m.betp()
    rows.append([n, mass_str(m, 4), " ; ".join(f2(x) for x in b), f3(m.expected_utility()),
                 f"[{f2(m.lower_expectation())} ; {f2(m.upper_expectation())}]"])
rows.append(["Moyenne floue", f"TFN ({f2(ex_mean.a)} ; {f2(ex_mean.b)} ; {f2(ex_mean.c)})", "—", f3(ex_mean.centroid),
             f"[{f2(ex_mean.alpha_cut(0.5)[0])} ; {f2(ex_mean.alpha_cut(0.5)[1])}]"])
rows.append(["OWA (la plupart)", f"TFN ({f2(ex_owa.a)} ; {f2(ex_owa.b)} ; {f2(ex_owa.c)})",
             "poids " + " ; ".join(f2(w) for w in ex_owa_w), f3(ex_owa.centroid),
             f"[{f2(ex_owa.alpha_cut(0.5)[0])} ; {f2(ex_owa.alpha_cut(0.5)[1])}]"])
table(["Règle", "Masse fusionnée (principaux éléments focaux)", "BetP (F ; M ; B ; E)", "Valeur nette", "Intervalle"],
      rows, widths=[2.8, 5.6, 3.3, 1.9, 2.9], size=8, caption="Résultat de chaque opérateur sur l'exemple")
d_, y_ = ex_res["Dempster"], ex_res["Yager"]
para(f"Dempster, en renormalisant, attribue {f2(d_.betp()[2])} de probabilité pignistique à « Bon » : le conflit "
     f"disparaît du résultat. Yager place {f2(y_.ignorance)} de masse sur Θ, ce qui élargit nettement l'intervalle "
     f"[{f2(y_.lower_expectation())} ; {f2(y_.upper_expectation())}] et signale au jury que les agents se contredisent. "
     f"Dubois-Prade conserve l'information « Moyen ou Excellent » sous forme disjonctive. Côté flou, la moyenne "
     f"pondérée donne une valeur proche de celle de Dempster mais sans mesure explicite du conflit, et l'OWA « la "
     f"plupart » ignore l'avis extrême le plus favorable (poids nul sur le premier rang). Cet exemple résume "
     f"l'enjeu : à données identiques, les opérateurs ne racontent pas la même histoire au décideur.")

# =============================================================================
# 6. CALIBRATION ET PONDÉRATION
# =============================================================================
H("6. Calibration des agents et pondération des critères", 1)
H("6.1 Poids des critères par AHP et BWM", 2)
names_short = ["Technique", "Soft skills", "Expérience", "Adéquation"]
frac = lambda v: {0.5: "1/2", 1.0: "1", 2.0: "2", 1 / 3: "1/3", 3.0: "3"}.get(round(v, 6), f2(v))
table(["AHP"] + names_short, [[names_short[i]] + [frac(v) for v in AHP_EXPERT_MATRIX[i]] for i in range(4)],
      widths=[3.0, 2.6, 2.6, 2.6, 2.6], caption="Matrice de comparaisons par paires (échelle de Saaty)")
table(["Critère", "Poids AHP", "Poids BWM", "Vecteur BWM (a_Bj ; a_jW)", "Poids vrais (générateur)"],
      [[CRITERIA_LABELS[c], f3(a), f3(b), f"{int(bo)} ; {int(ow)}", f3(t)] for c, a, b, bo, ow, t in
       zip(CRITERIA, P["AHP"], P["BWM"], BWM_BEST_TO_OTHERS, BWM_OTHERS_TO_WORST, [0.35, 0.25, 0.2, 0.2])] +
      [["Cohérence", f"CR = {f3(P['AHP_CR'])}", f"ξ/CI = {f3(P['BWM_ratio'])}", "—", "—"]],
      widths=[3.8, 2.6, 2.6, 3.8, 3.6], caption="Poids obtenus et contrôle de cohérence")
para(f"Les deux méthodes sont cohérentes ($\\lambda_{{\\mathrm{{max}}}}={f3(P['AHP_lambda'])}$, CR = {f3(P['AHP_CR'])} "
     f"< 0,1) et placent les compétences techniques en tête. Une matrice volontairement incohérente fournie dans "
     f"`config.py` donne CR = {f2(P['AHP_incoherent_CR'])} : le système la rejette et demande une **réélicitation**. "
     f"Les poids AHP sont utilisés dans toute la suite ; leur écart avec les poids vrais simule l'imperfection "
     f"d'une élicitation experte.")
H("6.2 Fiabilité et calibration des masses", 2)
para(r"La fiabilité de chaque agent est estimée sur un **jeu de calibration distinct** du jeu de test (120 candidats, "
     r"graine différente). Pour chaque avis non abstenu $n$, on compare la distribution $\mathrm{BetP}_n$ à la vraie "
     r"classe $y_n$ ; on calcule l'exactitude, le score de Brier, l'erreur de calibration ECE (sur des tranches $B_b$ de "
     r"confiance $\max_k\mathrm{BetP}_{nk}$) et la surconfiance $s$ :")
E(r"\mathrm{Brier}=\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{4}\left(\mathrm{BetP}_{nk}-\mathbb{1}[y_n=k]\right)^{2}")
E(r"\mathrm{ECE}=\sum_{b}\frac{|B_b|}{N}\left|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\right|")
para("La fiabilité utilisée pour l'affaiblissement corrige l'exactitude du hasard (1/4 pour quatre classes) et pénalise "
     "la surconfiance :")
E(r"\alpha=\mathrm{clip}\left(\frac{\mathrm{acc}-1/4}{3/4}\right)\times(1-s)\text{,}\qquad s=\max\left(0,\;\mathrm{conf}_{\mathrm{moy}}-\mathrm{acc}\right)")
para("Une variante **optimise directement** les α par descente par coordonnées sur une grille, en minimisant l'erreur "
     "quadratique entre l'évaluation fusionnée (Dempster) et le score vrai sur la calibration.")
c3 = calag[calag["scenario"].str.startswith("S3")]
table(["Agent (S3)", "Exactitude", "Brier", "ECE", "Confiance", "Surconfiant", "α déclarée", "α calibrée"],
      [[r.agent, f2(r.exactitude), f2(r.brier), f2(r.ece), f2(r.confiance_moy), "**oui**" if r.surconfiant else "non",
        f2(r.fiabilite_declaree), f2(r.fiabilite_calibree)] for r in c3.itertuples()],
      widths=[3.8, 1.8, 1.4, 1.3, 1.8, 1.8, 1.8, 1.8], caption="Calibration des agents dans le scénario S3 (200 candidats)")
figure("fig_diagramme_fiabilite.png", "Diagrammes de fiabilité (S3) : un agent bien calibré suit la diagonale ; "
       "l'Analyse CV, surconfiante, est nettement en dessous", 16.5)
llm = S["stabilite_llm"]
para(f"La calibration identifie correctement l'Analyse CV comme surconfiante : le recruteur lui déclarait une "
     f"fiabilité de 0,90, la calibration la ramène à {f2(c3.set_index('agent').loc['Analyse CV', 'fiabilite_calibree'])}. "
     f"**Agent LLM** : sur cinq exécutions, l'écart-type moyen de son évaluation nette est de "
     f"{f3(llm['LLM']['std_moyen_entre_executions'])} (contre {f3(llm['Manager (règles)']['std_moyen_entre_executions'])} "
     f"pour un agent à règles) et l'étiquette attribuée n'est identique sur les cinq exécutions que dans "
     f"{pct(llm['LLM']['accord_label_entre_executions'])} des cas. Son écart confiance–exactitude "
     f"({f2(llm['LLM']['ecart_confiance_exactitude'])}) conduit à un affaiblissement plus fort.")
c5 = calag[calag["scenario"].str.startswith("S5")]
para(r"**Audit d'équité des agents.** La calibration mesure aussi, pour chaque agent, l'écart moyen de ses avis entre "
     r"les groupes B et A à vraie valeur égale : avec $d=o-s$ l'écart entre avis perçu et score vrai, $g=\bar{d}_{B}-\bar{d}_{A}$. En S5, l'audit détecte "
     + ", ".join(f"{r.agent} ($g={f2(r.ecart_groupes)}$)" for r in c5.itertuples() if r.biais_detecte)
     + r". Le mode « calibrées+audit » affaiblit ces agents d'un facteur $\max(0{,}1;\,1-|g|/0{,}15)$. L'attribut "
       r"protégé n'est utilisé que pour cet audit, jamais pour évaluer un candidat.")

# =============================================================================
# 7. PROTOCOLE
# =============================================================================
H("7. Générateur de scénarios et protocole expérimental", 1)
H("7.1 Modèle génératif", 2)
para(r"Pour chaque candidat, quatre scores latents corrélés sont tirés ($\rho=0{,}5$ entre technique et expérience), "
     r"ainsi qu'un attribut protégé $g\in\{A,B\}$ **indépendant** des scores. L'utilité vraie, qui définit le vrai "
     r"classement, est une intégrale de Choquet 2-additive (équation " + str(n_cho) + r"), additive lorsque les "
     r"interactions sont nulles :")
E(r"z\sim\mathcal{N}(0,\Sigma)\text{,}\qquad s_j=\Phi(z_j)\text{,}\qquad U=\sum_j\phi^{*}_j s_j-\frac{1}{2}\sum_{j<l}I^{*}_{jl}\left|s_j-s_l\right|")
para(r"Un agent perçoit $o=s_j+\beta\,\mathbb{1}[g=B]+\varepsilon$, $\varepsilon\sim\mathcal{N}(0,\sigma^{2})$ ; avec "
     r"probabilité $p_{\mathrm{panne}}$ son avis est remplacé par une valeur aléatoire uniforme (avis erroné), et il "
     r"peut s'abstenir. Un candidat est un **cas difficile**, à renvoyer au jury, si son profil est incomplet ou s'il a "
     r"reçu au moins deux avis erronés et trompeurs (écart d'au moins deux classes). Un classement expert bruité "
     r"($U+\mathcal{N}(0,0{,}05^2)$) est disponible sur un sous-ensemble annoté de 12 candidats. Tous les tirages "
     r"sont indexés par (graine, candidat, agent, critère), ce qui permet de modifier un seul facteur à la fois.")
table(["Scénario", "Description"], [[k, v] for k, v in S["scenarios"].items()], widths=[3.8, 12.7],
      caption="Scénarios contrôlés")
H("7.2 Critères d'évaluation", 2)
para(r"Soit $n_c$ et $n_d$ les nombres de paires concordantes et discordantes entre le classement prédit et le vrai "
     r"classement, $d_i$ l'écart de rang du candidat $i$, $P_k$ et $T_k$ les top-$k$ prédit et vrai, et $r_A$, $r_B$ "
     r"les taux de sélection des deux groupes :")
E(r"\tau=\frac{n_c-n_d}{n(n-1)/2}\text{,}\qquad \rho=1-\frac{6\sum_i d_i^{2}}{n(n^{2}-1)}")
E(r"\mathrm{Prec}@k=\frac{|P_k\cap T_k|}{k}\text{,}\qquad \mathrm{RI}=\frac{\min(r_A,r_B)}{\max(r_A,r_B)}")
table(["Critère", "Mesure implémentée (fusionrh/metrics.py)"], [
    ["Exactitude", "τ de Kendall (version b), ρ de Spearman vs vrai classement ; précision top-3/5/10"],
    ["Accord humain", "τ de Kendall vs classement expert (sous-ensemble annoté)"],
    ["Conflit", "K moyen ; balayage K → 1 ; F1 / précision / rappel du renvoi au jury ; AUC du conflit et de l'ignorance"],
    ["Robustesse", "stabilité du top-3 sous perturbation (±0,05) et retrait d'un agent ; % de paires inversées à l'ajout d'un candidat"],
    ["Sensibilité", "τ entre classements avant / après variation de ±10 % et ±20 % de chaque poids AHP"],
    ["Équité", "ratio d'impact (règle des 4/5), taux B/A cumulés vs mérite ; contrefactuel : rangs modifiés quand seul le groupe change"],
    ["Explicabilité", "grille à 7 critères notés de 0 à 2"],
    ["Coût", "temps de calcul vs nombre d'agents (3–10) et de candidats (20–320)"],
], widths=[3.0, 13.5], caption="Critères d'évaluation")
H("7.3 Protocole et tests statistiques", 2)
para(f"Pour chaque scénario et chaque réplication $r=0,\\ldots,{S['reps'] - 1}$ : (1) génération d'un jeu de calibration "
     f"(120 candidats, graine 10 000 + r) et d'un jeu de test (40 candidats, graine r) ; (2) estimation des "
     f"fiabilités, des interactions de Choquet et des seuils de supervision **sur la calibration uniquement** ; "
     f"(3) évaluation des 9 méthodes × 4 modes de fiabilité sur le test, soit "
     f"{len(main):,}".replace(",", " ") + " classements évalués." +
     f" Les expériences de robustesse, de sensibilité et de contrefactuel utilisent {int(rob['rep'].nunique())} "
     f"réplications. Les méthodes sont comparées par un **test de Friedman** sur les τ appariés par réplication, puis "
     f"par des **tests de Wilcoxon signés** contre la meilleure méthode avec **correction de Holm** (Demšar, 2006).")

H("7.4 Implémentation et tests", 2)
para("Le système est écrit en Python (NumPy, SciPy, pandas). Les fonctions de masse sont des vecteurs de 16 "
     "composantes indexés par masque binaire ; la combinaison conjonctive est vectorisée par tables d'intersection, ce "
     "qui ramène une combinaison à quelques microsecondes. Le BWM est résolu par programmation linéaire (HiGHS), "
     "l'identification de Choquet par moindres carrés régularisés. **49 tests unitaires** vérifient les propriétés "
     "mathématiques attendues, notamment :")
bullets([
    "validité des masses (somme égale à 1, positivité) et encadrement Bel ≤ BetP ≤ Pl ;",
    "neutralité de la masse vide, commutativité et associativité de la règle de Dempster ;",
    "reproduction exacte du paradoxe de Zadeh pour Dempster, Yager et Dubois-Prade ; erreur levée quand K = 1 ;",
    "aller-retour de la table de correspondance sur les singletons et sur Θ ;",
    "cohérence AHP (CR nul pour une matrice parfaitement cohérente, CR > 0,1 pour la matrice incohérente) et BWM ;",
    "réduction de Choquet à la somme pondérée sans interaction, projection de monotonie, identification des "
    "interactions vraies à 0,02 près ;",
    "détection de l'agent surconfiant par la calibration ; contrefactuel neutre en l'absence de biais ;",
    "interface commune : les 7 opérateurs et les 9 méthodes produisent un classement valide et une explication.",
])

# =============================================================================
# 8. RÉSULTATS
# =============================================================================
H("8. Résultats", 1)
H("8.1 Exactitude du classement", 2)
figure("fig_kendall_scenarios.png", f"τ de Kendall par scénario et par méthode (fiabilités calibrées, "
       f"{S['reps']} réplications, IC à 95 %)", 16.5)
table(["Méthode", "τ Kendall", "ρ Spearman", "Top-3", "Top-5", "Top-10", "τ vs expert"],
      [[m, f3(r.kendall), f3(r.spearman), f2(r.top3), f2(r.top5), f2(r.top10), f3(r.kendall_expert)]
       for m, r in overall.iterrows()],
      widths=[3.6, 2.0, 2.1, 1.6, 1.6, 1.6, 2.0], caption="Exactitude moyenne sur les six scénarios (fiabilités calibrées)")
para(f"Le test de Friedman rejette l'égalité des méthodes dans les six scénarios (p < 10⁻²⁰). La famille floue domine : "
     f"**{best}** obtient le meilleur τ moyen ({f3(overall.loc[best, 'kendall'])}), suivi de la moyenne floue "
     f"({f3(overall.loc['Moyenne floue', 'kendall'])}) et de Choquet ({f3(overall.loc['Choquet', 'kendall'])}). Les "
     f"quatre règles évidentielles forment un groupe compact (τ entre {f3(overall.loc[EVIDENTIAL_METHODS, 'kendall'].min())} "
     f"et {f3(overall.loc[EVIDENTIAL_METHODS, 'kendall'].max())}). L'explication est structurelle : la combinaison "
     f"conjonctive de plusieurs sources concordantes **concentre la masse** sur une ou deux classes, et la probabilité "
     f"pignistique (équation {n_betp}) ne distingue plus deux candidats d'une même classe, alors qu'une moyenne de "
     f"nombres flous conserve la position fine de chaque avis. Mamdani est le moins exact "
     f"(τ = {f3(overall.loc['Mamdani', 'kendall'])}) : 14 règles à trois termes produisent une surface de décision en "
     f"paliers. L'OWA « au moins un » (le maximum) s'effondre en conflit (τ = {f3(kt.loc['OWA (au moins un)', scen('S2')])} "
     f"en S2) car un seul avis erroné optimiste suffit à promouvoir un candidat.")
table(["Méthode"] + [s.split(" ", 1)[0] for s in SCEN],
      [[m] + [("**" + f3(kt.loc[m, s]) + "**") if m == kt[s].idxmax() else f3(kt.loc[m, s]) for s in SCEN] for m in METHOD_NAMES],
      widths=[3.6] + [2.15] * len(SCEN), caption="τ de Kendall par scénario (meilleure méthode en gras)")
lines = []
for s in SCEN:
    d = stats[s]
    ns = [m for m, p in d["wilcoxon_holm_vs_meilleure"].items() if p >= 0.05]
    lines.append(f"**{s}** — meilleure : {d['meilleure']} ; Friedman χ² = {d['friedman_chi2']:.1f}".replace(".", ",")
                 + (f" ; non significativement différente(s) au seuil de 5 % après Holm : {', '.join(ns)}." if ns
                    else " ; toutes les autres méthodes sont significativement inférieures."))
bullets(lines)
para(f"**Lecture par scénario.** En S2 (conflit élevé), toutes les méthodes perdent environ 0,15 point de τ ; les plus "
     f"résistantes sont celles qui atténuent l'avis isolé. En S3 (agent surconfiant), l'écart entre familles se "
     f"maintient et l'effet de la calibration est maximal (§8.2). En S4 (ignorance forte), l'exactitude baisse pour "
     f"tous, mais les règles évidentielles exploitent l'abstention de façon neutre. En S6 (interactions), **Choquet "
     f"devient la meilleure méthode** (τ = {f3(kt.loc['Choquet', scen('S6')])}) grâce aux interactions identifiées "
     f"sur la calibration, qui retrouvent la redondance technique/expérience et la complémentarité soft "
     f"skills/adéquation. Le classement expert bruité est reproduit dans le même ordre, ce qui confirme la hiérarchie.")

H("8.2 Expérience clé : fiabilités égales, déclarées, calibrées", 2)
figure("fig_modes_fiabilite.png", "Effet du mode de fiabilité : ensemble des scénarios (gauche) et scénario S3 avec "
       "agent surconfiant (droite)", 16.5)
rows = []
for m in METHOD_NAMES:
    e = stats["effet_mode_fiabilite"][m]["calibrées vs déclarées"]
    rows.append([m, f3(modes.loc[m, "égales"]), f3(modes.loc[m, "déclarées"]), f3(modes.loc[m, "calibrées"]),
                 f3(modes.loc[m, "optimisées"]), f"{e['diff_moy']:+.3f}".replace(".", ","),
                 f"{e['p_holm']:.1e}" if e["p_holm"] < 0.01 else f2(e["p_holm"]), f"{gain_s3[m]:+.3f}".replace(".", ",")])
table(["Méthode", "Égales", "Déclarées", "Calibrées", "Optimisées", "Calibr. − décl.", "p (Holm)", "Gain en S3"], rows,
      widths=[3.3, 1.6, 1.7, 1.7, 1.8, 2.0, 1.8, 1.7], caption="τ de Kendall moyen selon le mode de fiabilité")
para(f"Toutes conditions confondues, l'effet du mode est faible mais significatif pour la plupart des méthodes : les "
     f"agents du scénario nominal sont de qualité voisine, donc des fiabilités égales sont déjà raisonnables. L'effet "
     f"devient net quand **la fiabilité déclarée est fausse** : en S3, où le recruteur accorde 0,9 à un agent "
     f"surconfiant et très bruité, la calibration améliore le τ de toutes les méthodes pondérées, jusqu'à "
     f"{gain_s3.max():+.3f}".replace(".", ",") + f" pour {gain_s3.idxmax()}." +
     " Yager en profite le plus : avec une fiabilité surestimée, l'agent surconfiant crée un conflit que Yager transforme "
     "en ignorance, ce qui écrase l'information des autres agents ; une fois l'agent affaibli, ce conflit disparaît. "
     "L'optimisation directe de α apporte un léger gain supplémentaire à la famille évidentielle, pour laquelle elle est "
     "optimisée, au prix d'environ cent fusions du jeu de calibration. L'OWA « au moins un » est insensible au mode : "
     "avec ce quantificateur, seul l'avis maximal compte.")

H("8.3 Gestion du conflit et renvoi au jury", 2)
figure("fig_zadeh.png", "Comportement quand K → 1 (paradoxe de Zadeh) : seule la règle de Dempster attribue une "
       "masse 1 à l'hypothèse « Moyen » que les deux agents jugeaient quasi impossible", 15.5)
z = zadeh[zadeh["eps"] == zadeh["eps"].min()].set_index("methode")
para(f"Quand K → 1, Dempster conclut « Moyen » avec m = {f2(z.loc['Dempster', 'm_Moyen'])}, Yager place "
     f"{f2(z.loc['Yager', 'm_Theta'])} sur Θ (« je ne sais pas »), Dubois-Prade place la masse sur "
     f"{{Faible, Excellent}} (« l'un ou l'autre, à trancher ») et Murphy partage entre Faible et Excellent. Seules "
     f"Yager et Dubois-Prade **signalent** l'incohérence dans le résultat lui-même ; Dempster masque le conflit "
     f"derrière une certitude fausse. Le degré K reste toutefois calculé et exposé pour toutes les règles, ce qui "
     f"permet à la supervision de le détecter.")
figure("fig_balayage_conflit.png", "Balayage du conflit : exactitude par famille et qualité du renvoi au jury quand la "
       "probabilité d'avis erroné augmente", 16.5)
k0, k6 = sw.index.min(), sw.index.max()
para(f"Quand la probabilité de panne ajoutée passe de {f2(k0)} à {f2(k6)}, le τ de Dempster passe de "
     f"{f2(sw.loc[k0, 'Dempster'])} à {f2(sw.loc[k6, 'Dempster'])}, celui de {best} de {f2(sw.loc[k0, best])} à "
     f"{f2(sw.loc[k6, best])} et celui de l'OWA « au moins un » de {f2(sw.loc[k0, 'OWA (au moins un)'])} à "
     f"{f2(sw.loc[k6, 'OWA (au moins un)'])}. Les opérateurs qui atténuent l'avis isolé se dégradent le moins vite ; "
     f"le maximum est le plus fragile. Au-delà d'environ 40 % d'avis erronés, toutes les méthodes convergent : "
     f"aucune fusion ne compense une majorité de sources défaillantes. Le conflit K moyen **diminue** quand les "
     f"pannes augmentent, ce qui peut surprendre : la calibration détecte des agents moins fiables et les affaiblit "
     f"davantage, ce qui transforme leur désaccord en ignorance avant la combinaison. Le conflit n'est donc pas un "
     f"indicateur absolu de qualité des sources, mais un indicateur relatif à leur fiabilité estimée.")
figure("fig_renvoi_jury.png", "F1 du renvoi au jury par méthode et par scénario", 13.5)
para(f"La famille évidentielle détecte bien mieux les cas à renvoyer (F1 moyen {f2(evid_f1)} contre {f2(fuzzy_f1)}). "
     f"La décomposition par type de cas difficile explique pourquoi : pour détecter les **avis erronés**, la dispersion "
     f"floue vaut le conflit K (AUC {f2(auc_c[FUZZY_METHODS].mean())} contre {f2(auc_c[EVIDENTIAL_METHODS].mean())}) ; "
     f"mais pour détecter les **profils incomplets**, l'ignorance évidentielle est quasi parfaite (AUC "
     f"{f2(auc_i[EVIDENTIAL_METHODS].mean())} contre {f2(auc_i[FUZZY_METHODS].mean())}). En effet, une abstention est "
     f"la masse vide, élément neutre qui laisse visible l'absence d'information, alors que la moyenne floue doit "
     f"exclure l'abstention et en perd la trace. Dubois-Prade obtient le meilleur F1 ({f2(f1_by['Dubois-Prade'])}) ; "
     f"Yager le moins bon de sa famille ({f2(f1_by['Yager'])}) car il **confond conflit et ignorance** dans m(Θ) : un "
     f"candidat aux avis contradictoires ressemble à un candidat sans avis.")

H("8.4 Robustesse et sensibilité aux poids", 2)
figure("fig_robustesse.png", "Robustesse (S1 et S2) et sensibilité aux poids (S1)", 16.5)
inv_e = robm.loc[EVIDENTIAL_METHODS, "inversion_rang_pct"].mean()
para(f"La stabilité du top-3 sous perturbation des avis est comparable entre méthodes (de "
     f"{f2(robm['stab_top3_perturbation'].min())} à {f2(robm['stab_top3_perturbation'].max())}). Le retrait d'un agent "
     f"est plus pénalisant (pire cas moyen entre {f2(robm['pire_retrait_agent'].min())} et "
     f"{f2(robm['pire_retrait_agent'].max())}) : le système dépend fortement du Manager, seul agent à couvrir trois "
     f"critères avec précision, ce qui plaide pour une redondance des évaluateurs. Le résultat le plus marquant "
     f"concerne l'**inversion de rang** : l'ajout d'un seul candidat inverse en moyenne {f2(inv_e)} % des paires avec "
     f"le TOPSIS classique, dont la normalisation dépend de l'ensemble des candidats, contre "
     f"{f2(robm.loc[FUZZY_METHODS, 'inversion_rang_pct'].mean())} % pour les méthodes floues. Pour un recrutement, "
     f"c'est un défaut d'équité procédurale : l'ordre entre deux candidats ne devrait pas dépendre d'un troisième.")
para(f"Sous une variation de ±20 % d'un poids, le τ entre classements reste supérieur à "
     f"{f2(sens20.drop('Mamdani').min())} pour toutes les méthodes pondérées (±10 % : supérieur à "
     f"{f2(sens10.drop('Mamdani').min())}) : les classements sont robustes aux imprécisions d'élicitation AHP/BWM. "
     f"Mamdani est insensible aux poids (τ = 1) car l'importance des critères est encodée dans les règles — c'est une "
     f"rigidité autant qu'une robustesse.")

H("8.5 Équité", 2)
figure("fig_equite.png", "Équité dans le scénario S5 (agents biaisés contre le groupe B)", 16.5)
table(["Méthode", "B/A S1", "B/A décl.", "B/A calibr.", "B/A audit", "4/5 brut", "|Δrang| calibr.", "|Δrang| audit", "τ audit"],
      [[m, f2(ba1.loc[m, "calibrées"]), f2(ba5.loc[m, "déclarées"]), f2(ba5.loc[m, "calibrées"]),
        f2(ba5.loc[m, "calibrées+audit"]), f2(raw5.loc[m, "calibrées"]), f2(dr_piv.loc[m, "calibrées"]),
        f2(dr_piv.loc[m, "calibrées+audit"]), f3(kq_piv.loc[m, "calibrées+audit"])] for m in METHOD_NAMES],
      size=7.5, widths=[3.0, 1.4, 1.5, 1.6, 1.5, 1.6, 1.9, 1.9, 1.4],
      caption=f"Équité : taux de sélection B/A cumulés (vérité = {f2(truth_ba)}) et déplacement contrefactuel (S5)")
para(f"**Mesure.** Sur un top-10 parmi 40 candidats, le ratio d'impact calculé réplication par réplication est très "
     f"bruité et biaisé vers le bas : le classement vrai lui-même n'obtient en moyenne que {f2(truth_ratio)}. Nous "
     f"cumulons donc les sélections par groupe sur les {S['reps']} réplications et comparons le rapport des taux B/A à "
     f"celui du **classement vrai** ({f2(truth_ba)} : dans ces échantillons, le groupe B est légèrement plus fort au "
     f"sommet, par le hasard du tirage).")
n_below = int((ba5["calibrées"] / truth_ba < 0.8).sum())
para(f"**Résultats.** Sans agent biaisé (S1), toutes les méthodes reproduisent le rapport du mérite (B/A entre "
     f"{f2(ba1['calibrées'].min())} et {f2(ba1['calibrées'].max())}) et le test contrefactuel ne modifie **aucun** "
     f"rang : les opérateurs n'ont pas accès à l'attribut protégé. En S5, tous les opérateurs **propagent** le biais "
     f"des agents : B/A tombe entre {f2(ba5['calibrées'].min())} et {f2(ba5['calibrées'].max())}, soit "
     f"{pct(ba5['calibrées'].min() / truth_ba)} à {pct(ba5['calibrées'].max() / truth_ba)} du rapport mérité — sous le "
     f"seuil des 4/5 relativement au mérite pour {n_below} méthodes sur 9. Pourtant la règle des 4/5 appliquée "
     f"brutalement (min/max des taux) donne {f2(raw5['calibrées'].min())} à {f2(raw5['calibrées'].max())} et **ne "
     f"détecte rien**, car le biais ne fait que réduire l'avantage réel du groupe B. C'est une limite importante de "
     f"cette règle, qui doit être complétée par des tests contrefactuels et, quand c'est possible, par une "
     f"comparaison au mérite. La calibration par exactitude ne corrige pas le biais : un décalage de 0,08 à 0,15 "
     f"dégrade peu l'exactitude par classe. L'**audit d'équité** ramène B/A entre "
     f"{f2(ba5['calibrées+audit'].min())} et {f2(ba5['calibrées+audit'].max())} et réduit le déplacement "
     f"contrefactuel moyen de {f2(dr_piv['calibrées'].mean())} à {f2(dr_piv['calibrées+audit'].mean())} rang, sans "
     f"perte d'exactitude notable. L'équité est donc une propriété de la **chaîne** (agents, calibration, "
     f"supervision), pas de l'opérateur.")

H("8.6 Explicabilité", 2)
figure("fig_grille_explicabilite.png", "Grille d'explicabilité (total sur 14)", 13.0)
para("Les règles évidentielles fournissent l'intervalle croyance–plausibilité, le conflit K par critère et les "
     "contributions de chaque agent ; Dempster perd des points de prévisibilité (paradoxe de Zadeh). Mamdani est le "
     "plus lisible pour un non-spécialiste grâce aux règles activées, mais il est le moins parcimonieux. Choquet est le "
     "plus difficile à expliquer (indices d'interaction). Exemple d'explication produite par le système (scénario S2, "
     "méthode Yager) pour un candidat renvoyé au jury :")
cfg = make_config("S2", n_candidates=15)
test_ex = generate_scenario(cfg, seed=3)
cal_ex = generate_scenario(make_config("S2", n_candidates=120), seed=10_003)
rel_ex = reliabilities_for("calibrées", test_ex, calibrate_agents(cal_ex))
w_ex = ahp_weights(AHP_EXPERT_MATRIX).weights
sys_ex = make_system("Yager", w_ex, rel_ex)
rc = sys_ex.evaluate(cal_ex)
sys_ex.supervisor = Supervisor.fit(rc.conflicts, rc.ignorances, cal_ex.hard_cases)
r_ex = sys_ex.evaluate(test_ex)
flag = np.flatnonzero(r_ex.flagged)
i_ex = int(flag[0]) if len(flag) else int(r_ex.order[0])
code(r_ex.explain(i_ex), size=7.5)

H("8.7 Coût de calcul", 2)
figure("fig_cout.png", "Temps de calcul en fonction du nombre de candidats et d'agents", 14.0)
para(f"Toutes les méthodes sont linéaires en nombre de candidats. Pour 320 candidats et 5 agents, le temps va de "
     f"{cost_big.min():.0f} ms ({cost_big.idxmin()}) à {cost_big.max():.0f} ms ({cost_big.idxmax()}). Les règles "
     r"évidentielles croissent avec le nombre d'agents (une combinaison par source sur $2^{|\Theta|}=16$ éléments), ce "
     f"qui reste négligeable pour quatre classes mais deviendrait exponentiel avec un cadre plus fin. Le coût n'est "
     f"donc pas un critère discriminant à cette échelle.")

# =============================================================================
# 9. DISCUSSION
# =============================================================================
H("9. Discussion", 1)
H("9.1 Quel opérateur, dans quelles conditions ?", 2)
table(["Condition / objectif", "Opérateur recommandé", "Justification"], [
    ["Classer avec précision (agents globalement fiables)", f"{best}, moyenne floue", "meilleurs τ, top-k et accord expert (§8.1)"],
    ["Critères en interaction (redondance, complémentarité)", "Choquet 2-additive (interactions apprises)", "meilleur en S6 ; 6 paramètres identifiables"],
    ["Conflit fréquent, avis aberrants", "OWA « la plupart », Murphy", "dégradation la plus lente (§8.3) ; éviter « au moins un »"],
    ["Détecter les cas à renvoyer au jury", "Dubois-Prade (ou Dempster) pour la supervision", "meilleur F1 ; K et m(Θ) explicites"],
    ["Conflit extrême K → 1", "Yager ou Dubois-Prade, jamais Dempster seul", "Dempster crée une certitude fausse"],
    ["Fiabilités déclarées douteuses, agent LLM", "toute règle + fiabilités calibrées", "gain maximal en S3 (§8.2)"],
    ["Stabilité procédurale (ajout de candidats)", "scores absolus (Choquet, Mamdani, flou)", "TOPSIS classique : inversions de rang"],
    ["Explication au candidat / non-expert", "Mamdani (règles) + intervalle évidentiel", "grille d'explicabilité (§8.6)"],
    ["Agents potentiellement biaisés", "audit d'équité en calibration", "aucun opérateur ne corrige seul (§8.5)"],
], widths=[5.0, 4.8, 6.7], caption="Synthèse : choix de l'opérateur selon les conditions")
para(f"**Recommandation.** Le meilleur compromis observé est une **architecture hybride** : un opérateur flou ({best}, "
     "ou Choquet si des interactions entre critères sont attestées) produit le classement, et un indicateur évidentiel "
     "(conflit K et ignorance issus de Dubois-Prade ou de Dempster) pilote la supervision et le renvoi au jury. Les "
     "fiabilités sont calibrées sur des données de référence et un audit d'équité des agents est réalisé à chaque "
     "recalibrage. Cette séparation exploite le point fort de chaque famille : la résolution fine des agrégations "
     "floues et la sémantique explicite de l'ignorance et du conflit des fonctions de croyance.")
H("9.2 Menaces sur la validité", 2)
bullets([
    "**Validité interne.** Les opérateurs évidentiels sont combinés à un TOPSIS appliqué à l'espérance pignistique, "
    "ce qui les pénalise en exactitude ; un TOPSIS par intervalles exploitant [Bel ; Pl] pourrait réduire l'écart. "
    "La règle de Dubois-Prade est appliquée séquentiellement, ce qui introduit une dépendance à l'ordre des agents.",
    "**Validité de construit.** La vérité de référence est une utilité latente simulée et le classement expert est "
    "lui-même simulé. La grille d'explicabilité est notée par les auteurs et devrait être validée par une étude "
    "auprès de recruteurs.",
    "**Validité externe.** Les données sont synthétiques ; les conclusions dépendent du modèle génératif, en "
    "particulier du lien entre le bruit et la forme des masses, et du fait que l'agent LLM est simulé. Une "
    "validation sur données réelles anonymisées est nécessaire avant tout déploiement.",
    "**Validité statistique.** Les ratios d'impact sur de petits effectifs sont bruités ; c'est pourquoi les "
    "conclusions d'équité reposent sur des taux cumulés et sur des tests contrefactuels appariés.",
])

H("9.3 Recommandations pour un déploiement", 2)
para("Les résultats conduisent aux recommandations opérationnelles suivantes pour une entreprise qui voudrait passer "
     "du prototype à un usage réel :")
bullets([
    "**Constituer un jeu de calibration** de dossiers passés dont l'issue est connue (évaluation de fin de période "
    "d'essai, performance à un an), anonymisé, pour estimer les fiabilités, les interactions de Choquet et les seuils "
    "de supervision ; le recalibrer périodiquement.",
    "**Ne jamais utiliser les fiabilités déclarées** sans les confronter à une calibration : c'est en S3, quand la "
    "fiabilité déclarée est fausse, que les écarts entre modes sont les plus forts.",
    "**Traiter un agent LLM comme une source surconfiante par défaut** : mesurer sa stabilité sur plusieurs "
    "exécutions, l'affaiblir en conséquence, et vérifier qu'il n'infère ni émotions ni attributs protégés.",
    "**Afficher systématiquement l'intervalle et le conflit** à côté du score, et rendre obligatoire l'examen humain "
    "des dossiers signalés, avec un taux de renvoi soutenable pour le jury (de l'ordre de 10 à 20 %).",
    "**Auditer l'équité des agents** à chaque recalibrage, avec des contrefactuels et une comparaison au mérite, et "
    "pas seulement avec la règle des quatre cinquièmes.",
    "**Éviter le TOPSIS classique** pour des classements mis à jour au fil des candidatures (inversions de rang) ; "
    "préférer un score absolu ou figer l'ensemble des candidats avant de classer.",
])

# =============================================================================
# 10. ÉTHIQUE
# =============================================================================
H("10. Dimension éthique et réglementaire", 1)
H("10.1 AI Act européen (règlement (UE) 2024/1689)", 2)
para("Les systèmes d'IA destinés au recrutement ou à la sélection de personnes physiques — notamment pour publier des "
     "offres ciblées, filtrer les candidatures et évaluer les candidats — sont classés **à haut risque** (annexe III, "
     "point 4 a). Le règlement est entré en vigueur le 1er août 2024 ; les interdictions s'appliquent depuis le "
     "2 février 2025 ; les obligations des systèmes à haut risque de l'annexe III sont applicables en principe à partir "
     "du 2 août 2026, la proposition « omnibus numérique » de la Commission (novembre 2025) envisageant un report : le "
     "calendrier exact doit être vérifié à la date de déploiement. Le **fournisseur** du système doit notamment assurer :")
bullets([
    "un **système de gestion des risques** (art. 9) et une **gouvernance des données** (art. 10) incluant l'examen des "
    "biais ; l'art. 10(5) encadre le traitement exceptionnel de catégories particulières de données à cette seule fin ;",
    "une **documentation technique** (art. 11) et la **journalisation** automatique des événements (art. 12) ;",
    "la **transparence** envers le déployeur (art. 13) et un **contrôle humain** effectif (art. 14) : la personne qui "
    "supervise doit pouvoir comprendre les capacités et limites du système, détecter les anomalies, ne pas suivre sa "
    "sortie ou l'annuler, et rester consciente du biais d'automatisation ;",
    "un niveau approprié d'**exactitude, de robustesse et de cybersécurité** (art. 15).",
])
para("Le **déployeur** — l'entreprise qui recrute — doit utiliser le système conformément à sa notice, confier la "
     "supervision à des personnes compétentes, informer les représentants des travailleurs et les personnes concernées "
     "(art. 26) ; les candidats disposent d'un droit à l'explication des décisions individuelles (art. 86). Enfin, "
     "l'art. 5(1)(f) **interdit la reconnaissance des émotions** sur le lieu de travail : un agent qui analyse une "
     "transcription ou une vidéo d'entretien ne doit pas en inférer les émotions du candidat. Pour des candidats situés "
     "dans l'UE, l'art. 22 du RGPD encadre en outre les décisions fondées exclusivement sur un traitement automatisé.")
H("10.2 Maroc : loi 09-08 et CNDP", 2)
para("Au Maroc, le traitement des données des candidats relève de la **loi n° 09-08** relative à la protection des "
     "personnes physiques à l'égard du traitement des données à caractère personnel, dont le respect est contrôlé par "
     "la **Commission nationale de contrôle de la protection des données à caractère personnel (CNDP)**. Ses exigences "
     "principales pour un tel système sont :")
bullets([
    "la **déclaration préalable** ou la **demande d'autorisation** du traitement auprès de la CNDP avant sa mise en œuvre ;",
    "une **finalité** déterminée, explicite et légitime, des données **adéquates, pertinentes et non excessives**, et "
    "une durée de conservation limitée ;",
    "l'**information** des candidats et, selon les cas, leur **consentement** ; le respect de leurs droits d'**accès**, "
    "de **rectification** et d'**opposition** ;",
    "la **sécurité et la confidentialité** des données, y compris chez les sous-traitants ;",
    "l'encadrement des **transferts vers l'étranger**, point critique si un LLM hébergé hors du Maroc analyse les CV ;",
    "l'interdiction qu'une décision produisant des effets juridiques soit prise **sur le seul fondement d'un traitement "
    "automatisé** destiné à évaluer certains aspects de la personnalité (art. 11).",
])
para("Le **Code du travail** marocain (art. 9) interdit par ailleurs toute discrimination en matière d'emploi, y compris "
     "à l'embauche, fondée notamment sur le sexe, l'origine, la religion ou le handicap.")
H("10.3 Analyse des risques et mesures implémentées", 2)
table(["Risque", "Conséquence", "Mesure dans le système"], [
    ["Décision entièrement automatisée", "atteinte aux droits (AI Act art. 14, loi 09-08 art. 11)", "le système classe et explique ; renvoi au jury des cas à fort conflit ou forte ignorance"],
    ["Agent biaisé", "discrimination indirecte (Code du travail art. 9)", "attribut protégé jamais fourni aux opérateurs ; audit d'équité des agents ; contrefactuels ; taux B/A vs mérite"],
    ["Agent surconfiant ou LLM instable", "classement erroné, fausse assurance", "calibration (Brier, ECE), affaiblissement, mesure de stabilité entre exécutions"],
    ["Conflit masqué (paradoxe de Zadeh)", "certitude fausse présentée au jury", "conflit K exposé ; Yager / Dubois-Prade en conflit extrême"],
    ["Opacité", "contestation impossible (art. 13, 86)", "explication par candidat : intervalle [Bel ; Pl], accords et désaccords, règles activées"],
    ["Instabilité procédurale", "rang dépendant des autres candidats", "mesure des inversions de rang ; préférence pour les scores absolus"],
    ["Données excessives, transferts", "non-conformité loi 09-08", "uniquement des scores par critère ; données synthétiques ; pas d'envoi à un service externe"],
    ["Inférence d'émotions", "pratique interdite (art. 5(1)(f))", "aucun agent n'analyse les émotions"],
], widths=[3.6, 4.4, 8.5], size=8, caption="Risques, conséquences et mesures")

H("10.4 Gouvernance et suivi", 2)
para("La conformité ne s'arrête pas à la conception. Un déploiement responsable suppose : un **registre des "
     "traitements** et la déclaration à la CNDP ; une **analyse d'impact** documentant les risques du tableau "
     "précédent ; une **journalisation** de chaque classement (avis, fiabilités, opérateur, version du modèle, "
     "décision du jury) pour permettre l'audit et la contestation ; une **information claire des candidats** sur "
     "l'usage d'un outil d'aide à la décision et sur leurs droits ; un **suivi en production** des indicateurs "
     "mesurés ici (taux de renvoi, dérive de calibration des agents, taux de sélection par groupe) ; enfin une "
     "**formation** des recruteurs à la lecture des intervalles et du conflit, afin de limiter le biais "
     "d'automatisation, conformément à l'obligation de maîtrise de l'IA (AI Act, art. 4).")

# =============================================================================
# 11. CONCLUSION
# =============================================================================
H("11. Conclusion et perspectives", 1)
para(f"Nous avons conçu et évalué un système multi-agents d'aide à la sélection de candidats dans lequel neuf méthodes "
     f"de fusion sont interchangeables derrière une interface unique. La réponse à la question de recherche est "
     f"**conditionnelle**. La famille floue ({best}, moyenne floue, et Choquet en présence d'interactions) produit les "
     f"classements les plus **exacts**. La famille évidentielle (Dubois-Prade, Dempster) fournit le meilleur signal "
     f"pour **renvoyer les cas douteux au jury** et l'explication la plus riche de l'incertitude, à condition de ne "
     f"jamais utiliser Dempster seul en conflit extrême. La **calibration** des fiabilités est décisive dès qu'un agent "
     f"est surconfiant, ce qui est le cas typique d'un agent LLM. Enfin, **aucun opérateur n'est équitable par "
     f"construction**, et la règle des quatre cinquièmes peut à elle seule masquer un biais réel.")
para("Les perspectives sont : un TOPSIS par intervalles exploitant directement [Bel ; Pl] ; l'affaiblissement "
     "contextuel (une fiabilité par agent et par critère) ; un agent LLM réel évalué sur des CV anonymisés ; une étude "
     "utilisateur de l'explicabilité avec des recruteurs ; et une validation sur données réelles après déclaration ou "
     "autorisation auprès de la CNDP.")

# =============================================================================
# RÉFÉRENCES
# =============================================================================
H("Références", 1)
refs = [
    "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1–3.",
    "Chen, C.-T. (2000). Extensions of the TOPSIS for group decision-making under fuzzy environment. Fuzzy Sets and Systems, 114(1), 1–9.",
    "Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.",
    "Dempster, A. P. (1967). Upper and lower probabilities induced by a multivalued mapping. Annals of Mathematical Statistics, 38(2), 325–339.",
    "Dubois, D., & Prade, H. (1988). Representation and combination of uncertainty with belief functions and possibility measures. Computational Intelligence, 4(3), 244–264.",
    "Grabisch, M. (1997). k-order additive discrete fuzzy measures and their representation. Fuzzy Sets and Systems, 92(2), 167–189.",
    "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. Proceedings of ICML.",
    "Hwang, C.-L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications. Springer.",
    "Kusner, M., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual fairness. Advances in NeurIPS.",
    "Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller. International Journal of Man-Machine Studies, 7(1), 1–13.",
    "Mercier, D., Quost, B., & Denœux, T. (2008). Refined modeling of sensor reliability in the belief function framework using contextual discounting. Information Fusion, 9(2), 246–258.",
    "Murphy, C. K. (2000). Combining belief functions when evidence conflicts. Decision Support Systems, 29(1), 1–9.",
    "Rezaei, J. (2015). Best-worst multi-criteria decision-making method. Omega, 53, 49–57.",
    "Rezaei, J. (2016). Best-worst multi-criteria decision-making method: Some properties and a linear model. Omega, 64, 126–130.",
    "Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.",
    "Shafer, G. (1976). A Mathematical Theory of Evidence. Princeton University Press.",
    "Smets, P., & Kennes, R. (1994). The transferable belief model. Artificial Intelligence, 66(2), 191–234.",
    "Yager, R. R. (1987). On the Dempster-Shafer framework and new combination rules. Information Sciences, 41(2), 93–137.",
    "Yager, R. R. (1988). On ordered weighted averaging aggregation operators in multicriteria decision making. IEEE Transactions on Systems, Man, and Cybernetics, 18(1), 183–190.",
    "Yager, R. R. (1996). Quantifier guided aggregation using OWA operators. International Journal of Intelligent Systems, 11(1), 49–73.",
    "Zadeh, L. A. (1986). A simple view of the Dempster-Shafer theory of evidence and its implication for the rule of combination. AI Magazine, 7(2), 85–90.",
    "Règlement (UE) 2024/1689 du Parlement européen et du Conseil du 13 juin 2024 établissant des règles harmonisées concernant l'intelligence artificielle.",
    "Loi n° 09-08 relative à la protection des personnes physiques à l'égard du traitement des données à caractère personnel, promulguée par le dahir n° 1-09-15 du 18 février 2009 (Maroc).",
    "Loi n° 65-99 relative au Code du travail, promulguée par le dahir n° 1-03-194 du 11 septembre 2003 (Maroc).",
    "Uniform Guidelines on Employee Selection Procedures (1978), 29 CFR § 1607.4(D).",
]
for r in refs:
    p = doc.add_paragraph(style="List Number"); p.paragraph_format.space_after = Pt(1)
    run = p.add_run(r); run.font.size = Pt(9)

# =============================================================================
# ANNEXES
# =============================================================================
H("Annexe A — Organisation du code et reproduction", 1)
table(["Fichier / dossier", "Contenu"], [
    ["fusionrh/belief.py", "fonctions de masse, Bel, Pl, BetP, affaiblissement ; règles de Dempster, Yager, Dubois-Prade, Murphy"],
    ["fusionrh/fuzzy.py", "nombres flous, table de correspondance, quantificateurs et poids OWA"],
    ["fusionrh/fusion/", "interface FusionOperator et les 7 opérateurs inter-agents (registre OPERATORS)"],
    ["fusionrh/aggregation.py", "TOPSIS, fuzzy TOPSIS, Choquet 2-additive et son identification, Mamdani"],
    ["fusionrh/weighting.py", "AHP (CR) et BWM linéaire (ξ/CI)"],
    ["fusionrh/agents.py, generator.py", "agents évaluateurs et générateur de scénarios"],
    ["fusionrh/calibration.py", "exactitude, Brier, ECE, surconfiance, audit d'équité, optimisation de α, stabilité LLM"],
    ["fusionrh/pipeline.py", "DecisionSystem, supervision humaine, explications"],
    ["fusionrh/metrics.py, explainability.py", "critères d'évaluation, grille d'explicabilité"],
    ["tests/ (49 tests)", "propriétés mathématiques : Zadeh, neutralité de la masse vide, associativité, monotonie de Choquet…"],
    ["experiments/, demo/", "campagne expérimentale, figures, application Streamlit et démonstration console"],
], widths=[5.2, 11.3], caption="Organisation du code")
para("**Reproduction** (depuis la racine du projet) :", align=WD_ALIGN_PARAGRAPH.LEFT)
code("""pip install -r requirements.txt
pytest                                      # 49 tests unitaires
python -m experiments.run_experiments       # campagne complète (~1 h)
python -m experiments.fairness_pooled       # équité cumulée
python -m experiments.figures               # figures
python livrables/build_report.py            # ce rapport
streamlit run demo/app.py                   # démonstration""")
H("Annexe B — Résultats détaillés", 1)
sp = cal_main.pivot_table(index="methode", columns="scenario", values="spearman").reindex(METHOD_NAMES)
t5 = cal_main.pivot_table(index="methode", columns="scenario", values="top5").reindex(METHOD_NAMES)
table(["Méthode"] + [x.split(" ", 1)[0] for x in SCEN],
      [[m] + [f3(sp.loc[m, x]) for x in SCEN] for m in METHOD_NAMES],
      widths=[3.6] + [2.15] * len(SCEN), size=8, caption="ρ de Spearman par scénario (fiabilités calibrées)")
table(["Méthode"] + [x.split(" ", 1)[0] for x in SCEN],
      [[m] + [f2(t5.loc[m, x]) for x in SCEN] for m in METHOD_NAMES],
      widths=[3.6] + [2.15] * len(SCEN), size=8, caption="Précision du top-5 par scénario (fiabilités calibrées)")
det = cal_main.groupby("methode")[["renvoi_precision", "renvoi_rappel", "renvoi_f1", "taux_renvoi", "auc_conflit",
                                   "auc_ignorance"]].mean().reindex(METHOD_NAMES)
table(["Méthode", "Précision", "Rappel", "F1", "Taux de renvoi", "AUC conflit", "AUC ignorance"],
      [[m, f2(r.renvoi_precision), f2(r.renvoi_rappel), f2(r.renvoi_f1), pct(r.taux_renvoi), f2(r.auc_conflit),
        f2(r.auc_ignorance)] for m, r in det.iterrows()],
      widths=[3.6, 1.9, 1.9, 1.6, 2.4, 2.4, 2.6], size=8, caption="Renvoi au jury : moyennes sur les six scénarios")
H("Annexe C — Base de règles de Mamdani", 1)
para("Termes d'entrée : Bas (0 ; 0 ; 0,5), Moyen (0,2 ; 0,5 ; 0,8), Élevé (0,5 ; 1 ; 1). Termes de sortie : les quatre "
     "classes de la table de correspondance.", size=9.5)
for k, r in enumerate(DEFAULT_RULES):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(0)
    run = p.add_run(f"R{k + 1}. {rule_text(r)}"); run.font.size = Pt(9)

add_page_numbers()
out = OUT / "Rapport_Fusion_Avis_MCDM.docx"
doc.save(out)
print("rapport écrit :", out, f"({EQN[0]} équations, {FIGN[0]} figures, {TABN[0]} tableaux)")
