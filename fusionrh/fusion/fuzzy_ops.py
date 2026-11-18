"""Opérateurs de la famille floue (fusion inter-agents) : moyenne pondérée floue et OWA de Yager.

Les abstentions explicites (masse vide) sont exclues ; l'ignorance partielle d'un agent se traduit
par un nombre flou plus large (table de correspondance). La fiabilité sert de poids d'importance.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..agents import Opinion
from ..fuzzy import QUANTIFIERS, TFN, importance_owa_weights, weighted_tfn_mean
from .base import CriterionFusion, FusionOperator, dispersion

_IGNORANT = TFN(0.0, 0.5, 1.0)


def _finish(tfn: TFN, active: Sequence[Opinion], rel: Sequence[float], note: str = "") -> CriterionFusion:
    lo, hi = tfn.alpha_cut(0.5)
    conf = dispersion([o.tfn.centroid for o in active], rel)
    contrib = [(o.agent, o.tfn.centroid, float(r)) for o, r in zip(active, rel)]
    return CriterionFusion(tfn.centroid, lo, hi, tfn, conf, hi - lo, None, contrib, note)


def _active(opinions, reliabilities):
    pairs = [(o, float(r)) for o, r in zip(opinions, reliabilities) if not o.abstained]
    return [p[0] for p in pairs], [p[1] for p in pairs]


class WeightedFuzzyMean(FusionOperator):
    name = "Moyenne floue"
    family = "floue"

    def fuse(self, opinions, reliabilities):
        act, rel = _active(opinions, reliabilities)
        if not act:
            return CriterionFusion(0.5, 0.25, 0.75, _IGNORANT, 0.0, 0.5, None, [], "aucun avis exprimé")
        return _finish(weighted_tfn_mean([o.tfn for o in act], rel), act, rel)


class OWAOperator(FusionOperator):
    """OWA de Yager pondéré par importance, poids issus d'un quantificateur linguistique."""

    family = "floue"
    LABELS = {"most": "la plupart", "at_least_one": "au moins un", "all": "tous"}

    def __init__(self, quantifier: str = "most"):
        self.quantifier = quantifier
        self.name = f"OWA ({self.LABELS[quantifier]})"

    def fuse(self, opinions, reliabilities):
        act, rel = _active(opinions, reliabilities)
        if not act:
            return CriterionFusion(0.5, 0.25, 0.75, _IGNORANT, 0.0, 0.5, None, [], "aucun avis exprimé")
        order = np.argsort([-o.tfn.centroid for o in act], kind="stable")
        w = importance_owa_weights([rel[k] for k in order], QUANTIFIERS[self.quantifier])
        arr = np.array([act[k].tfn.as_tuple() for k in order])
        a, b, c = (w[:, None] * arr).sum(axis=0)
        return _finish(TFN(float(a), float(b), float(c)), act, rel, f"poids OWA={np.round(w, 2).tolist()}")

    def __repr__(self) -> str:
        return f"OWAOperator({self.quantifier!r})"
