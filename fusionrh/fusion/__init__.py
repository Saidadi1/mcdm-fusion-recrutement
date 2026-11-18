"""Registre des opérateurs de fusion et des méthodes complètes (opérateur + agrégateur).

Méthode = (opérateur inter-agents, agrégateur inter-critères). Les 9 méthodes comparées :

| Méthode            | Fusion des agents (par critère) | Agrégation des critères |
|--------------------|---------------------------------|-------------------------|
| Dempster           | règle de Dempster               | TOPSIS                  |
| Yager              | règle de Yager                  | TOPSIS                  |
| Dubois-Prade       | règle de Dubois-Prade           | TOPSIS                  |
| Murphy             | moyenne de Murphy               | TOPSIS                  |
| Moyenne floue      | moyenne pondérée floue          | fuzzy TOPSIS            |
| OWA (la plupart)   | OWA, quantificateur « la plupart » | fuzzy TOPSIS         |
| OWA (au moins un)  | OWA, quantificateur « au moins un » | fuzzy TOPSIS        |
| Choquet            | moyenne pondérée floue          | Choquet 2-additive      |
| Mamdani            | moyenne pondérée floue          | inférence de Mamdani    |
"""
from __future__ import annotations

from .base import CriterionFusion, FusionOperator
from .evidential import DempsterOperator, DuboisPradeOperator, MurphyOperator, YagerOperator
from .fuzzy_ops import OWAOperator, WeightedFuzzyMean

OPERATORS = {
    "Dempster": DempsterOperator,
    "Yager": YagerOperator,
    "Dubois-Prade": DuboisPradeOperator,
    "Murphy": MurphyOperator,
    "Moyenne floue": WeightedFuzzyMean,
    "OWA (la plupart)": lambda: OWAOperator("most"),
    "OWA (au moins un)": lambda: OWAOperator("at_least_one"),
}


def get_operator(name: str) -> FusionOperator:
    return OPERATORS[name]()


__all__ = ["CriterionFusion", "FusionOperator", "DempsterOperator", "YagerOperator",
           "DuboisPradeOperator", "MurphyOperator", "WeightedFuzzyMean", "OWAOperator",
           "OPERATORS", "get_operator"]
