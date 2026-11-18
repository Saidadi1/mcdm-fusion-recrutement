"""Interface commune des opérateurs de fusion (couche 4).

Un opérateur fusionne, pour UN candidat et UN critère, les avis des agents qui couvrent ce critère,
en tenant compte de leur fiabilité. Tous les opérateurs renvoient un `CriterionFusion` homogène,
ce qui les rend interchangeables dans le pipeline de décision.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..agents import Opinion
from ..belief import MassFunction
from ..fuzzy import TFN


@dataclass
class CriterionFusion:
    crisp: float                       # évaluation nette ∈ [0,1] (E[BetP] ou centroïde)
    low: float                         # borne basse (espérance inférieure / α-coupe 0,5)
    high: float                        # borne haute
    tfn: TFN                           # évaluation floue fusionnée
    conflict: float                    # degré de conflit (K évidentiel ou dispersion floue)
    ignorance: float                   # ignorance résiduelle
    mass: Optional[MassFunction] = None
    contributions: list[tuple[str, float, float]] = field(default_factory=list)  # (agent, valeur, poids)
    note: str = ""


class FusionOperator(ABC):
    """Opérateur de fusion interchangeable."""

    name: str = "operator"
    family: str = "?"   # « évidentielle » ou « floue »

    @abstractmethod
    def fuse(self, opinions: Sequence[Opinion], reliabilities: Sequence[float]) -> CriterionFusion:
        """Fusionne les avis (un par agent couvrant le critère) ; reliabilities ∈ [0,1]."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def dispersion(values: Sequence[float], weights: Sequence[float]) -> float:
    """Indicateur de conflit pour la famille floue : écart absolu moyen pondéré, normalisé sur [0,1]."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if len(v) < 2 or w.sum() <= 0:
        return 0.0
    mean = (w * v).sum() / w.sum()
    return float(np.clip(2.0 * (w * np.abs(v - mean)).sum() / w.sum(), 0.0, 1.0))
