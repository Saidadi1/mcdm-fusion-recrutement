"""Opérateurs de la famille évidentielle : Dempster, Yager, Dubois-Prade, moyenne de Murphy.

Chaque source est d'abord affaiblie (discounting) par sa fiabilité α_i, puis combinée.
Le degré de conflit K est celui de la combinaison conjonctive des sources affaiblies (identique
pour les quatre règles, ce qui permet de comparer leur *traitement* du conflit à conflit égal).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..agents import Opinion
from ..belief import (MassFunction, TotalConflictError, conflict_degree, dempster,
                      dubois_prade, murphy, yager)
from ..fuzzy import mass_to_tfn
from .base import CriterionFusion, FusionOperator


class EvidentialOperator(FusionOperator):
    family = "évidentielle"

    def combine(self, masses: list[np.ndarray], reliabilities: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def fuse(self, opinions: Sequence[Opinion], reliabilities: Sequence[float]) -> CriterionFusion:
        if len(opinions) == 0:
            mf = MassFunction.vacuous()
            return CriterionFusion(0.5, 0.0, 1.0, mass_to_tfn(mf), 0.0, 1.0, mf, [], "aucun avis")
        disc = [o.mass.discount(r).m for o, r in zip(opinions, reliabilities)]
        k = conflict_degree(disc)
        note = ""
        try:
            fused = self.combine(disc, reliabilities)
        except TotalConflictError:
            fused = MassFunction.vacuous().m
            note = "conflit total (K=1) : règle non définie, masse vide retournée"
        mf = MassFunction(fused)
        lo, hi = mf.lower_expectation(), mf.upper_expectation()
        contrib = [(o.agent, o.crisp, float(r)) for o, r in zip(opinions, reliabilities) if not o.abstained]
        return CriterionFusion(mf.expected_utility(), lo, hi, mass_to_tfn(mf), k, hi - lo, mf, contrib, note)


class DempsterOperator(EvidentialOperator):
    name = "Dempster"

    def combine(self, masses, reliabilities):
        return dempster(masses)


class YagerOperator(EvidentialOperator):
    name = "Yager"

    def combine(self, masses, reliabilities):
        return yager(masses)


class DuboisPradeOperator(EvidentialOperator):
    name = "Dubois-Prade"

    def combine(self, masses, reliabilities):
        return dubois_prade(masses)


class MurphyOperator(EvidentialOperator):
    name = "Murphy"

    def combine(self, masses, reliabilities):
        return murphy(masses)
