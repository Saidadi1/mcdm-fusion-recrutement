"""Grille d'évaluation de l'explicabilité (0 = absent, 1 = partiel, 2 = complet).

Les notes sont attribuées par analyse des sorties produites par chaque méthode (voir rapport §6.7) ;
certaines sont vérifiées automatiquement par `audit_outputs`.
"""
from __future__ import annotations

GRID_CRITERIA = [
    ("Intervalle d'incertitude", "La sortie fournit-elle un intervalle croyance–plausibilité ou une α-coupe ?"),
    ("Traçabilité des agents", "Peut-on dire quel agent a pesé et dans quel sens ?"),
    ("Exposition du conflit", "Le conflit entre agents est-il mesuré et restitué ?"),
    ("Règles lisibles", "La décision s'énonce-t-elle en règles linguistiques compréhensibles ?"),
    ("Comportement prévisible", "Absence d'effets contre-intuitifs (paradoxe de Zadeh, saut brutal) ?"),
    ("Parcimonie", "Nombre réduit de paramètres à justifier devant un candidat / un auditeur."),
    ("Compréhension non-expert", "Un recruteur non spécialiste peut-il comprendre la justification ?"),
]

GRID_SCORES: dict[str, list[int]] = {
    #                    Interv. Trac. Confl. Règles Prév. Parc. NonExp.
    "Dempster":          [2,     1,    2,     0,     0,    2,    1],
    "Yager":             [2,     1,    2,     0,     2,    2,    1],
    "Dubois-Prade":      [2,     1,    2,     0,     2,    2,    1],
    "Murphy":            [2,     1,    2,     0,     1,    2,    1],
    "Moyenne floue":     [1,     2,    1,     0,     2,    2,    2],
    "OWA (la plupart)":  [1,     1,    1,     1,     2,    1,    1],
    "OWA (au moins un)": [1,     1,    0,     1,     1,    2,    1],
    "Choquet":           [1,     2,    1,     0,     2,    0,    0],
    "Mamdani":           [1,     1,    1,     2,     1,    0,    2],
}


def grid_total(method: str) -> int:
    return sum(GRID_SCORES[method])


def audit_outputs(ranking) -> dict[str, bool]:
    """Vérifications automatiques sur une sortie de `DecisionSystem.evaluate`."""
    r = ranking.results[0]
    return {
        "intervalle": all(x.low <= x.score + 1e-6 or x.low <= x.high for x in ranking.results),
        "contributions_agents": any(f.contributions for f in r.criteria),
        "conflit_mesure": any(x.conflict > 0 for x in ranking.results),
        "regles_activees": "regles" in r.details,
        "masses_disponibles": r.criteria[0].mass is not None,
    }
