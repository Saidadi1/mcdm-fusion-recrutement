"""Génère la présentation : livrables/Presentation_Fusion_Avis_MCDM.pptx

    python livrables/build_slides.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm, Pt

from fusionrh.pipeline import EVIDENTIAL_METHODS, FUZZY_METHODS, METHOD_NAMES

RES, FIG, OUT = ROOT / "results", ROOT / "figures", ROOT / "livrables"
BLUE, INK, INK2, MUTED, LIGHT = (RGBColor(0x1C, 0x5C, 0xAB), RGBColor(0x0B, 0x0B, 0x0B), RGBColor(0x52, 0x51, 0x4E),
                                 RGBColor(0x89, 0x87, 0x81), RGBColor(0xF0, 0xF4, 0xFA))
ORANGE = RGBColor(0xEB, 0x68, 0x34)

main = pd.read_csv(RES / "principal.csv")
cal = main[main["mode"] == "calibrées"]
overall = cal.groupby("methode").mean(numeric_only=True).reindex(METHOD_NAMES)
best = overall["kendall"].idxmax()
f1 = cal.groupby("methode")["renvoi_f1"].mean().reindex(METHOD_NAMES)
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
s3 = main[main["scenario"].str.startswith("S3")].pivot_table(index="methode", columns="mode", values="kendall").reindex(METHOD_NAMES)
gain = (s3["calibrées"] - s3["déclarées"])
rob = pd.read_csv(RES / "robustesse.csv").groupby("methode").mean(numeric_only=True).reindex(METHOD_NAMES)
kt = cal.pivot_table(index="methode", columns="scenario", values="kendall").reindex(METHOD_NAMES)
s6 = [c for c in kt.columns if c.startswith("S6")][0]


def f2(x):
    return f"{x:.2f}".replace(".", ",")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EQ_DIR = FIG / "equations"
EQ_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
_EQ = [0]


def formula(latex, size=26):
    """Rend une formule LaTeX (mathtext) en PNG transparent haute résolution ; renvoie le chemin."""
    _EQ[0] += 1
    path = EQ_DIR / f"eq_{_EQ[0]:02d}.png"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=size, color="#0b0b0b")
    fig.savefig(path, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return path


def eq(slide, latex, x, y, h=1.2, size=26, w_max=14.5):
    """Place une formule à la hauteur h (cm), réduite si elle dépasse la largeur w_max (cm)."""
    pic_ = slide.shapes.add_picture(str(formula(latex, size)), Cm(x), Cm(y), height=Cm(h))
    if pic_.width > Cm(w_max):
        ratio = Cm(w_max) / pic_.width
        pic_.width = Cm(w_max)
        pic_.height = int(pic_.height * ratio)
    return pic_


prs = Presentation()
prs.slide_width, prs.slide_height = Cm(33.867), Cm(19.05)
BLANK = prs.slide_layouts[6]
N = [0]


def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, bullets=False):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame; tf.word_wrap = True
    items = content if isinstance(content, list) else [content]
    for k, it in enumerate(items):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        segs = it.split("**")
        first = True
        for j, seg in enumerate(segs):
            if not seg:
                continue
            r = p.add_run(); r.text = (("• " if bullets and first and it.strip() else "") + seg)
            first = False
            r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold or (j % 2 == 1)
            r.font.name = "Calibri"
    return tb


def slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Cm(0.5), prs.slide_height)
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background()
    text(s, 1.3, 0.7, 30, 1.6, title, size=28, color=BLUE, bold=True)
    if subtitle:
        text(s, 1.3, 2.25, 30, 1.0, subtitle, size=14, color=INK2)
    text(s, 31.0, 17.9, 2.5, 0.8, str(N[0]), size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def pic(s, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Cm(w)
    if h: kw["height"] = Cm(h)
    return s.shapes.add_picture(str(FIG / name), Cm(x), Cm(y), **kw)


def box(s, x, y, w, h, title, body, color=BLUE, size=13):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = LIGHT; sh.line.color.rgb = color; sh.line.width = Pt(1.5)
    sh.adjustments[0] = 0.08
    text(s, x + 0.4, y + 0.25, w - 0.8, 1.0, title, size=size + 2, color=color, bold=True)
    text(s, x + 0.4, y + 1.3, w - 0.8, h - 1.5, body, size=size, color=INK, bullets=isinstance(body, list))


def table(s, x, y, w, rows, col_w, size=11, row_h=0.85):
    t = s.shapes.add_table(len(rows), len(rows[0]), Cm(x), Cm(y), Cm(w), Cm(row_h * len(rows))).table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Cm(cw)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            p = c.text_frame.paragraphs[0]
            for r in p.runs:
                r.font.size = Pt(size); r.font.bold = i == 0
                r.font.color.rgb = RGBColor(255, 255, 255) if i == 0 else INK
            c.fill.solid(); c.fill.fore_color.rgb = BLUE if i == 0 else (LIGHT if i % 2 else RGBColor(255, 255, 255))
    return t


# 1. Titre ---------------------------------------------------------------------
s = prs.slides.add_slide(BLANK); N[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = BLUE; bg.line.fill.background()
text(s, 2.5, 4.2, 29, 4, "Système multi-agents d'aide à la sélection de candidats fondé sur le MCDM",
     size=38, color=RGBColor(255, 255, 255), bold=True)
text(s, 2.5, 9.3, 29, 2.5, "Étude comparative des opérateurs de fusion des avis (Dempster-Shafer, Yager, fusion floue), "
     "avec calibration des agents et évaluation sur scénarios synthétiques", size=18, color=RGBColor(0xCD, 0xE2, 0xFB))
text(s, 2.5, 14.5, 29, 1.5, "Soutenance de projet — 2026-2027", size=16, color=RGBColor(255, 255, 255))

# 2. Problématique ---------------------------------------------------------------
s = slide("Problématique", "Le cœur du sujet n'est pas le classement, mais la fusion des avis")
box(s, 1.3, 3.8, 10, 6.5, "Hétérogènes", "Chaque évaluateur couvre des critères différents : RH, manager, test psychométrique, CV, LLM.")
box(s, 11.8, 3.8, 10, 6.5, "Incertains", "Information incomplète, ignorance assumée : « je ne peux pas juger les soft skills sur un CV ».")
box(s, 22.3, 3.8, 10, 6.5, "Conflictuels", "Le test contredit l'entretien : selon l'opérateur, le même candidat finit 1er ou 10e.", color=ORANGE)
q = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(1.3), Cm(11.5), Cm(31), Cm(4.2))
q.fill.solid(); q.fill.fore_color.rgb = LIGHT; q.line.fill.background()
text(s, 2, 11.9, 29.5, 3.5, ["Question de recherche",
                             "Quel opérateur de fusion produit le classement le plus **fiable**, le plus **robuste au conflit**, "
                             "le plus **équitable** et le plus **explicable** — et dans quelles conditions ?"], size=18)

# 3. Architecture ----------------------------------------------------------------
s = slide("Architecture en six couches", "Chaque opérateur de fusion est interchangeable derrière une interface unique")
pic(s, "fig_architecture.png", 1.3, 3.3, h=13.8)
text(s, 25.6, 4.0, 7.9, 13, ["**Interface commune**", "FusionOperator.fuse(avis, fiabilités) → CriterionFusion",
                            "make_system(\"Yager\", poids, fiabilités)", "",
                            "**5 agents** simulés dont un LLM (instable, surconfiant)", "",
                            "**Supervision** : seuils appris sur la calibration"], size=13)

# 4. Incertitude -----------------------------------------------------------------
s = slide("Double représentation de chaque avis", "Fonction de masse sur Θ = {Faible, Moyen, Bon, Excellent}  ⇄  nombre flou triangulaire")
table(s, 1.3, 3.8, 18, [["Élément focal", "Nombre flou"], ["Faible", "(0 ; 0 ; 1/3)"], ["Moyen", "(0 ; 1/3 ; 2/3)"],
                       ["Bon", "(1/3 ; 2/3 ; 1)"], ["Excellent", "(2/3 ; 1 ; 1)"],
                       ["Sous-ensemble A", "(min a ; moy. b ; max c)"], ["Θ (ignorance)", "(0 ; 0,5 ; 1)"],
                       ["Masse m", "Σ m(A) · TFN(A)"]], [7, 11], size=14, row_h=1.1)
text(s, 20.3, 3.6, 13, 1, "**Construction d'un avis** (valeur perçue o, Ruspini μ)", size=14)
eq(s, r"m(\{k\})=(1-u)\,q\,p_k,\quad m(\{k,k\!+\!1\})=(1-u)(1-q),\quad m(\Theta)=u", 20.5, 4.7, h=0.85, w_max=12.8)
text(s, 20.3, 6.0, 13, 1, "**Affaiblissement** par la fiabilité α", size=14)
eq(s, r"m^{\alpha}(A)=\alpha\,m(A),\qquad m^{\alpha}(\Theta)=1-\alpha+\alpha\,m(\Theta)", 20.5, 7.1, h=0.85, w_max=12.8)
text(s, 20.3, 8.4, 13, 1, "**Croyance et plausibilité**", size=14)
eq(s, r"\mathrm{Bel}(A)=\sum_{\emptyset\neq B\subseteq A}m(B),\qquad \mathrm{Pl}(A)=\sum_{B\cap A\neq\emptyset}m(B)", 20.5, 9.5, h=1.5, w_max=12.8)
text(s, 20.3, 11.4, 13, 1, "**Décision** : probabilité pignistique", size=14)
eq(s, r"\mathrm{BetP}(\theta)=\frac{1}{1-m(\emptyset)}\sum_{A\ni\theta}\frac{m(A)}{|A|}", 20.5, 12.5, h=1.7, w_max=12.8)

# 5. Opérateurs ------------------------------------------------------------------
s = slide("Neuf méthodes de fusion comparées")
box(s, 1.3, 3.0, 15.3, 9.5, "Famille évidentielle (+ TOPSIS)",
    ["**Dempster** : renormalise le conflit K (paradoxe de Zadeh)", "**Yager** : conflit → ignorance Θ (prudente)",
     "**Dubois-Prade** : conflit → union A ∪ B", "**Murphy** : moyenne des masses puis combinaison",
     "Sortie : masse, BetP, [Bel ; Pl], conflit K"], size=15)
box(s, 17.3, 3.0, 15.3, 9.5, "Famille floue",
    ["**Moyenne pondérée floue** (+ fuzzy TOPSIS)", "**OWA** « la plupart » / « au moins un » (Yager 1996)",
     "**Choquet 2-additive** : interactions entre critères apprises (6 paramètres)",
     "**Mamdani** : 14 règles « SI technique Élevé ET … ALORS Bon »", "Sortie : nombre flou, α-coupe, règles activées"],
    color=ORANGE, size=15)
text(s, 1.3, 13.4, 31, 3, ["Toutes derrière la même interface : **FusionOperator.fuse(avis, fiabilités)** — changer d'opérateur = une ligne",
                           "Fiabilités : affaiblissement (évidentiel) ou poids d'importance (flou)"], size=15, bullets=True)

# 5 bis. Règles évidentielles ---------------------------------------------------------
s = slide("Règles de combinaison évidentielles", "Elles ne diffèrent que par le traitement du conflit K")
eq(s, r"m_{\cap}(C)=\sum_{A\cap B=C}m_1(A)\,m_2(B),\qquad K=m_{\cap}(\emptyset)", 2.0, 3.6, h=1.6, w_max=20)
rules_ = [
    ("Dempster", "renormalise : le conflit disparaît", r"m_D(C)=\frac{m_{\cap}(C)}{1-K}"),
    ("Yager", "le conflit devient de l'ignorance", r"m_Y(\Theta)=m_{\cap}(\Theta)+K"),
    ("Dubois-Prade", "le conflit va à l'union A ∪ B", r"m_{DP}(C)=m_{\cap}(C)+\sum_{A\cup B=C,\ A\cap B=\emptyset}m_1(A)\,m_2(B)"),
    ("Murphy", "moyenne puis n − 1 combinaisons", r"\bar{m}=\frac{1}{n}\sum_i m_i,\qquad m_M=\bar{m}\oplus\cdots\oplus\bar{m}"),
]
for k, (nm, desc, ltx) in enumerate(rules_):
    y = 6.0 + k * 2.55
    text(s, 1.3, y + 0.15, 6.5, 1.0, f"**{nm}**", size=16, color=BLUE)
    text(s, 1.3, y + 0.95, 7.5, 1.0, desc, size=12, color=INK2)
    eq(s, ltx, 9.0, y, h=1.55 if "sum" in ltx or "frac" in ltx else 0.8, w_max=13.3)
box(s, 23.0, 6.0, 10.0, 9.8, "Paradoxe de Zadeh",
    ["Agent 1 : Excellent 0,99 ; Moyen 0,01", "Agent 2 : Faible 0,99 ; Moyen 0,01", "K = 0,9999",
     "**Dempster** : Moyen = 1 (certitude fausse)", "**Yager** : Θ = 1 (« je ne sais pas »)",
     "**Dubois-Prade** : {Faible, Excellent}"], size=13, color=ORANGE)

# 5 ter. Agrégations floues ------------------------------------------------------------
s = slide("Agrégations floues", "OWA guidé par quantificateur, intégrale de Choquet 2-additive, inférence de Mamdani")
text(s, 1.3, 3.5, 15, 1, "**OWA de Yager** (arguments triés, bₖ = k-ième plus grand)", size=15)
eq(s, r"\mathrm{OWA}(a)=\sum_{k=1}^{n}w_k\,b_k,\qquad w_k=Q\left(\frac{S_k}{T}\right)-Q\left(\frac{S_{k-1}}{T}\right)", 1.6, 4.6, h=1.7, w_max=15.3)
text(s, 1.3, 6.6, 15.5, 1.6, ["« la plupart » : Q(r) = min(1, max(0, (r − 0,3)/0,5))", "« au moins un » : Q(r) = 1 si r > 0  →  maximum"],
     size=13, bullets=True)
text(s, 1.3, 9.2, 15, 1, "**Choquet 2-additive** (Shapley φ, interactions I)", size=15)
eq(s, r"C_{\mu}(x)=\sum_{i}\phi_i\,x_i-\frac{1}{2}\sum_{i<j}I_{ij}\,|x_i-x_j|", 1.6, 10.3, h=1.6, w_max=15.3)
text(s, 1.3, 12.2, 15.5, 2.2, ["I < 0 : redondance (technique ↔ expérience)", "I > 0 : complémentarité (soft skills ↔ adéquation)",
                               "monotonie : φᵢ − ½ Σⱼ |Iᵢⱼ| ≥ 0 ; I appris sur la calibration"], size=13, bullets=True)
text(s, 17.8, 3.5, 15, 1, "**Mamdani** (ET = min, agrégation = max)", size=15)
eq(s, r"\beta_r=\min_j\,\mu_{A_j^r}(x_j),\qquad y^{*}=\frac{\int y\,\mu(y)\,dy}{\int \mu(y)\,dy}", 18.1, 4.6, h=1.7, w_max=15)
text(s, 17.8, 6.6, 15, 3.5, ["14 règles expertes, par exemple :", "SI technique est Élevé ET soft skills est Moyen ALORS candidat est Bon",
                             "→ explication directement lisible par un recruteur"], size=13, bullets=True)
text(s, 17.8, 9.2, 15, 1, "**TOPSIS** (proximité à l'idéal)", size=15)
eq(s, r"CC_i=\frac{d_i^{-}}{d_i^{+}+d_i^{-}},\qquad d_i^{\pm}=\sqrt{\sum_j\left(v_{ij}-v_j^{\pm}\right)^2}", 18.1, 10.3, h=1.7, w_max=15)
text(s, 17.8, 12.2, 15, 2.2, ["normalisation dépendante de l'ensemble des candidats", "→ risque d'inversion de rang (mesuré)"],
     size=13, bullets=True)

# 5 quater. Exemple ---------------------------------------------------------------------
from fusionrh.belief import MassFunction, conflict_degree, dempster, dubois_prade, label_of, murphy, yager
_ag = [("Manager technique", {"Bon": 0.6, ("Bon", "Excellent"): 0.3, "Theta": 0.1}, 0.9),
       ("Test psychométrique", {"Moyen": 0.5, "Bon": 0.3, "Theta": 0.2}, 0.7),
       ("Analyse CV (surconfiant)", {"Excellent": 0.8, "Bon": 0.15, "Theta": 0.05}, 0.4)]
_raw = [MassFunction.from_dict(d) for _, d, _ in _ag]
_disc = [m.discount(a).m for m, (_, _, a) in zip(_raw, _ag)]


def _ms(m, k=3):
    return " ; ".join(f"{label_of(a)} {f2(v)}" for a, v in sorted(m.focal(), key=lambda t: -t[1])[:k] if round(v, 2) > 0)


s = slide("Exemple pas à pas : un critère, trois agents",
          f"Conflit après affaiblissement K = {f2(conflict_degree(_disc))} (sans affaiblissement : {f2(conflict_degree([m.m for m in _raw]))})")
table(s, 1.3, 3.5, 31.2, [["Agent", "Masse exprimée", "Fiabilité α"]] +
      [[n, _ms(m), f2(a)] for (n, _, a), m in zip(_ag, _raw)], [8, 18.2, 5], size=13, row_h=0.9)
_rows = [["Règle", "Masse fusionnée (éléments principaux)", "Valeur nette", "Intervalle [Bel ; Pl]"]]
for nm, fn in [("Dempster", dempster), ("Yager", yager), ("Dubois-Prade", dubois_prade), ("Murphy", murphy)]:
    mf = MassFunction(fn(_disc))
    _rows.append([nm, _ms(mf), f2(mf.expected_utility()), f"[{f2(mf.lower_expectation())} ; {f2(mf.upper_expectation())}]"])
table(s, 1.3, 8.0, 31.2, _rows, [5, 15.2, 4.5, 6.5], size=13, row_h=0.9)
text(s, 1.3, 13.2, 31, 4, ["Dempster renormalise : « Bon » domine, le conflit a disparu du résultat",
                           "Yager place la moitié de la masse sur Θ : intervalle large → signal pour le jury",
                           "Même données, quatre « histoires » différentes racontées au décideur"], size=14, bullets=True)

# 6. Calibration -----------------------------------------------------------------
P = S["poids"]
s = slide("Calibration des agents et pondération", "Sur un jeu de calibration distinct du jeu de test")
pic(s, "fig_diagramme_fiabilite.png", 1.3, 3.4, w=31)
text(s, 1.3, 11.0, 15, 7, ["**Poids des critères**",
                           f"AHP : technique {f2(P['AHP'][0])}, soft {f2(P['AHP'][1])}, expérience {f2(P['AHP'][2])}, adéquation {f2(P['AHP'][3])}",
                           f"CR = {P['AHP_CR']:.3f} < 0,1  ·  BWM ξ/CI = {P['BWM_ratio']:.3f}".replace(".", ","),
                           f"Matrice incohérente : CR = {f2(P['AHP_incoherent_CR'])} → réélicitation"], size=14)
text(s, 17.5, 11.0, 15.5, 1, "**Fiabilité α** (exactitude corrigée du hasard, pénalité de surconfiance s)", size=14)
eq(s, r"\alpha=\mathrm{clip}\left(\frac{\mathrm{acc}-1/4}{3/4}\right)\times(1-s)", 17.8, 12.1, h=1.5)
text(s, 17.5, 14.0, 15.5, 3, ["Brier, ECE, diagramme de fiabilité", "variante : α optimisé (erreur de fusion)",
                              "audit d'équité : écart entre groupes par agent"], size=13, bullets=True)

# 7. Protocole -------------------------------------------------------------------
s = slide("Générateur de scénarios et protocole", f"{S['reps']} réplications × 6 scénarios × 4 modes de fiabilité × 9 méthodes")
table(s, 1.3, 3.6, 18.5, [["Scénario", "Ce qui change"], ["S1 Nominal", "référence"], ["S2 Conflit", "+25 pts d'avis erronés"],
                         ["S3 Surconfiant", "agent CV bruité, masses catégoriques, α déclarée 0,9"],
                         ["S4 Ignorance", "30 % profils incomplets"], ["S5 Biais", "2 agents défavorisent le groupe B"],
                         ["S6 Interactions", "utilité vraie non additive"]], [5.5, 13], size=13, row_h=1.15)
text(s, 21, 3.6, 12, 14, ["**Vérité de référence** connue (utilité latente)", "",
                          "**Mesures** : τ Kendall, ρ Spearman, top-k, τ vs expert, F1 renvoi au jury, stabilité top-3, "
                          "inversion de rang, sensibilité ±10/20 %, règle des 4/5, contrefactuel, coût", "",
                          "**Tests** : Friedman + Wilcoxon-Holm"], size=14)

# 8. Exactitude ------------------------------------------------------------------
s = slide("Résultat 1 — Exactitude du classement",
          f"La famille floue l'emporte : {best} τ = {f2(overall.loc[best, 'kendall'])} ; évidentielle ≈ {f2(overall.loc[EVIDENTIAL_METHODS, 'kendall'].mean())}")
pic(s, "fig_kendall_scenarios.png", 1.3, 3.3, h=14.6)
text(s, 27.0, 4.0, 6.5, 14, ["Combinaison conjonctive = masse concentrée → perte de résolution intra-classe", "",
                             f"Choquet meilleur quand les critères interagissent (S6 : τ = {f2(kt.loc['Choquet', s6])})", "",
                             "Mamdani : surface en paliers", "",
                             "« Au moins un » s'effondre en conflit"], size=12, bullets=True)

# 9. Fiabilités ------------------------------------------------------------------
s = slide("Résultat 2 — Fiabilités égales, déclarées, calibrées",
          f"La calibration compte quand la fiabilité déclarée est fausse : gain max en S3 = {gain.max():+.3f} ({gain.idxmax()})".replace(".", ",") + f"")
pic(s, "fig_modes_fiabilite.png", 1.3, 3.4, w=31)

# 10. Zadeh ----------------------------------------------------------------------
s = slide("Résultat 3 — Comportement quand K → 1", "Paradoxe de Zadeh : deux agents quasi certains et opposés")
pic(s, "fig_zadeh.png", 1.3, 3.3, w=22)
text(s, 24.0, 3.8, 9.3, 14, ["**Dempster** : « Moyen » certain — certitude fausse", "",
                             "**Yager** : tout sur Θ — « je ne sais pas »", "",
                             "**Dubois-Prade** : {Faible, Excellent} — « à trancher »", "",
                             "**Murphy** : partage Faible / Excellent"], size=14)

# 11. Renvoi jury ---------------------------------------------------------------
s = slide("Résultat 3 bis — Renvoi au jury humain",
          f"Évidentielle : F1 = {f2(f1[EVIDENTIAL_METHODS].mean())}  vs  floue : F1 = {f2(f1[FUZZY_METHODS].mean())}")
pic(s, "fig_renvoi_jury.png", 1.3, 3.3, w=20.5)
text(s, 22.5, 4.0, 10.8, 14, ["Une abstention = masse vide m(Θ)=1 : l'ignorance reste visible", "",
                            "La moyenne floue doit exclure l'abstention → l'information manquante disparaît", "",
                            f"Meilleur : Dubois-Prade (F1 = {f2(f1['Dubois-Prade'])})", "",
                            f"Yager confond conflit et ignorance (F1 = {f2(f1['Yager'])})"], size=14, bullets=True)

# 12. Robustesse -----------------------------------------------------------------
s = slide("Résultat 4 — Robustesse et sensibilité",
          f"TOPSIS classique : {f2(rob.loc[EVIDENTIAL_METHODS, 'inversion_rang_pct'].mean())} % de paires inversées à l'ajout d'un candidat")
pic(s, "fig_robustesse.png", 1.3, 3.4, w=31)
text(s, 1.3, 13.5, 31, 4, ["Top-3 stable sous perturbation pour toutes les méthodes ; le retrait d'un agent est plus pénalisant",
                           "Poids ±20 % : τ > 0,9 pour toutes les méthodes pondérées → robustes à l'élicitation AHP/BWM",
                           "L'ordre entre deux candidats ne devrait pas dépendre d'un troisième : préférer des scores absolus"],
     size=14, bullets=True)

# 13. Équité ---------------------------------------------------------------------
s = slide("Résultat 5 — Équité", "Aucun opérateur ne corrige un agent biaisé : seul l'audit d'équité en calibration réduit le biais")
pic(s, "fig_equite.png", 1.3, 3.3, h=12.2)
text(s, 1.3, 15.6, 31, 3.2, ["Sans biais : contrefactuel = 0 rang modifié (l'attribut protégé n'est jamais vu)",
                             "Avec biais (S5) : B/A passe sous 4/5 du mérite, mais la règle des 4/5 brute ne détecte rien → "
                             "compléter par contrefactuels ; l'audit d'équité restaure B/A"], size=13, bullets=True)

# 14. Explicabilité --------------------------------------------------------------
s = slide("Explicabilité", "Grille à 7 critères + explication générée pour chaque candidat")
pic(s, "fig_grille_explicabilite.png", 1.3, 3.3, h=10.5)
import numpy as np
from fusionrh.calibration import calibrate_agents, reliabilities_for
from fusionrh.config import AHP_EXPERT_MATRIX
from fusionrh.generator import generate_scenario, make_config
from fusionrh.pipeline import Supervisor, make_system
from fusionrh.weighting import ahp_weights
_t = generate_scenario(make_config("S2", n_candidates=15), seed=3)
_c = generate_scenario(make_config("S2", n_candidates=120), seed=10_003)
_rel = reliabilities_for("calibrées", _t, calibrate_agents(_c))
_w = ahp_weights(AHP_EXPERT_MATRIX).weights
_sys = make_system("Yager", _w, _rel)
_rc = _sys.evaluate(_c)
_sys.supervisor = Supervisor.fit(_rc.conflicts, _rc.ignorances, _c.hard_cases)
_r = _sys.evaluate(_t)
_fl = np.flatnonzero(_r.flagged)
_i = int(_fl[0]) if len(_fl) else int(_r.order[0])
_lines = [l.strip() for l in _r.explain(_i).split("\n")]
_mam = make_system("Mamdani", _w, _rel).evaluate(_t).explain(int(_r.order[0])).split("\n")
_rules = [l.strip() for l in _mam if l.strip().startswith("R") and "(degré" in l][:2]
text(s, 22.5, 3.3, 10.8, 15, ["**Exemple (Yager, S2)**"] + _lines[:3] + [_lines[-1], "",
                              "**Mamdani — règles activées**"] + _rules, size=11)

# 15. Synthèse -------------------------------------------------------------------
s = slide("Synthèse — quel opérateur, dans quelles conditions ?")
table(s, 1.3, 2.8, 31.2, [["Objectif / condition", "Recommandation"],
                          ["Classer avec précision", f"{best}, moyenne floue"],
                          ["Critères en interaction", "Choquet 2-additive (interactions apprises)"],
                          ["Conflit fréquent", "OWA « la plupart », Murphy — éviter « au moins un »"],
                          ["Renvoyer les cas douteux au jury", "Dubois-Prade / Dempster (K et m(Θ) explicites)"],
                          ["Conflit extrême K → 1", "Yager ou Dubois-Prade, jamais Dempster seul"],
                          ["Fiabilités douteuses, agent LLM", "fiabilités calibrées (Brier, ECE)"],
                          ["Agents biaisés", "audit d'équité en calibration (aucun opérateur seul)"]],
      [12, 19.2], size=15, row_h=1.4)
text(s, 1.3, 15.2, 31, 3, ["**Architecture hybride recommandée** : opérateur flou pour classer + indicateur évidentiel pour superviser"],
     size=17, color=BLUE)

# 16. Éthique --------------------------------------------------------------------
s = slide("Dimension éthique et réglementaire", "Recrutement = application à haut risque")
box(s, 1.3, 3.5, 15.3, 9.0, "AI Act (UE 2024/1689)",
    ["Annexe III pt 4 : recrutement = **haut risque**", "Gestion des risques, données et biais (art. 9-10)",
     "Journalisation, transparence (art. 12-13)", "**Contrôle humain** effectif (art. 14)",
     "Interdiction de la reconnaissance des émotions au travail (art. 5)"], size=14)
box(s, 17.3, 3.5, 15.3, 9.0, "Maroc — loi 09-08 / CNDP",
    ["Déclaration / autorisation préalable CNDP", "Information, finalité, minimisation",
     "Droits d'accès, rectification, opposition", "Pas de décision **sur le seul fondement** d'un traitement automatisé",
     "Transferts à l'étranger encadrés (LLM hébergés)"], color=ORANGE, size=14)
text(s, 1.3, 13.6, 31, 2, "Dans le système : renvoi au jury · explications · audit d'équité · données synthétiques uniquement",
     size=15, color=BLUE, bold=True)

# 17. Démonstration --------------------------------------------------------------
s = slide("Démonstration")
text(s, 1.3, 3.2, 31, 14, ["**streamlit run demo/app.py**",
                           "   classement + intervalle [Bel ; Pl] + renvoi au jury + explication par candidat",
                           "   comparaison des 9 opérateurs sur le même scénario",
                           "   atelier de fusion : saisir deux masses et observer Dempster / Yager / Dubois-Prade / Murphy (Zadeh)",
                           "   calibration des agents : Brier, ECE, surconfiance, biais", "",
                           "**python demo/demo_cli.py --scenario S2**  (version ligne de commande)", "",
                           "**pytest**  → 49 tests unitaires  ·  **python -m experiments.run_experiments**  → campagne complète"],
     size=17)

# 18. Conclusion -----------------------------------------------------------------
s = slide("Conclusion et perspectives")
text(s, 1.3, 3.2, 31, 14, ["La réponse est **conditionnelle** : pas d'opérateur universel",
                           f"Flou ({best}, Choquet) pour l'**exactitude** ; évidentiel (Dubois-Prade) pour la **supervision** et l'explication de l'incertitude",
                           "La **calibration** des fiabilités est décisive face à un agent surconfiant",
                           "L'**équité** est une propriété de la chaîne, pas de l'opérateur",
                           "", "**Perspectives** : TOPSIS par intervalles [Bel ; Pl], affaiblissement contextuel, agent LLM réel sur CV anonymisés, "
                           "étude utilisateur avec des recruteurs, validation sur données réelles (autorisation CNDP)"],
     size=18, bullets=True)

out = OUT / "Presentation_Fusion_Avis_MCDM.pptx"
prs.save(out)
print("présentation écrite :", out)
