"""Agrégation multicritère (couche 5) : TOPSIS, fuzzy TOPSIS, intégrale de Choquet 2-additive,
système d'inférence floue de Mamdani. Tous implémentent la même interface `Aggregator`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np

from .config import CRITERIA, CRITERIA_LABELS, N_CRITERIA
from .fuzzy import TFN, tfn_distance, tri_membership


# ---------------------------------------------------------------------------
# Intégrale de Choquet 2-additive
# ---------------------------------------------------------------------------

@dataclass
class TwoAdditiveCapacity:
    """Capacité 2-additive représentée par les indices de Shapley φ_i et d'interaction I_ij.

    C_μ(x) = Σ_i φ_i x_i − ½ Σ_{i<j} I_ij |x_i − x_j|
    I_ij > 0 : complémentarité (les deux critères doivent être bons ensemble) ;
    I_ij < 0 : redondance (un bon score sur l'un compense l'autre).
    Monotonie : Σ_{j≠i} |I_ij| ≤ 2 φ_i pour tout i.
    """

    shapley: np.ndarray
    interactions: dict[tuple[int, int], float] = field(default_factory=dict)

    def is_monotone(self, tol: float = 1e-9) -> bool:
        s = np.zeros(len(self.shapley))
        for (i, j), v in self.interactions.items():
            s[i] += abs(v); s[j] += abs(v)
        return bool(np.all(s <= 2 * self.shapley + tol)) and abs(self.shapley.sum() - 1) < 1e-6

    def project_monotone(self) -> "TwoAdditiveCapacity":
        s = np.zeros(len(self.shapley))
        for (i, j), v in self.interactions.items():
            s[i] += abs(v); s[j] += abs(v)
        factor = np.where(s > 2 * self.shapley, 2 * self.shapley / np.maximum(s, 1e-12), 1.0)
        inter = {(i, j): v * min(factor[i], factor[j]) for (i, j), v in self.interactions.items()}
        return TwoAdditiveCapacity(self.shapley.copy(), inter)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = x @ self.shapley
        for (i, j), v in self.interactions.items():
            out = out - 0.5 * v * np.abs(x[..., i] - x[..., j])
        return out

    def mobius_singletons(self) -> np.ndarray:
        s = np.zeros(len(self.shapley))
        for (i, j), v in self.interactions.items():
            s[i] += v; s[j] += v
        return self.shapley - 0.5 * s


def choquet_2additive(x, shapley, interactions) -> np.ndarray:
    return TwoAdditiveCapacity(np.asarray(shapley, float), dict(interactions))(x)


def fit_interactions(x: np.ndarray, y: np.ndarray, shapley: np.ndarray,
                     ridge: float = 1e-3) -> TwoAdditiveCapacity:
    """Identification des interactions d'une capacité 2-additive (φ fixé, p. ex. poids AHP).

    Régression y ≈ β0 + β1 Σφ_i x_i + Σ γ_ij |x_i − x_j| (moindres carrés régularisés),
    puis I_ij = −2 γ_ij / β1 et projection sur les contraintes de monotonie.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pairs = list(combinations(range(x.shape[1]), 2))
    feats = [np.ones(len(x)), x @ shapley] + [np.abs(x[:, i] - x[:, j]) for i, j in pairs]
    a = np.column_stack(feats)
    reg = ridge * np.eye(a.shape[1]); reg[0, 0] = 0
    coef = np.linalg.solve(a.T @ a + reg, a.T @ y)
    beta1 = coef[1] if abs(coef[1]) > 1e-6 else 1e-6
    inter = {p: float(-2 * g / beta1) for p, g in zip(pairs, coef[2:])}
    return TwoAdditiveCapacity(np.asarray(shapley, float), inter).project_monotone()


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

@dataclass
class AggregationOutput:
    scores: np.ndarray                 # score global par candidat
    lower: np.ndarray                  # borne basse (croyance / α-coupe)
    upper: np.ndarray                  # borne haute (plausibilité / α-coupe)
    details: list[dict]                # informations d'explication par candidat


class Aggregator(ABC):
    name: str = "aggregator"

    @abstractmethod
    def aggregate(self, crisp: np.ndarray, low: np.ndarray, high: np.ndarray,
                  tfns: Sequence[Sequence[TFN]], weights: np.ndarray) -> AggregationOutput:
        """crisp/low/high : matrices (n candidats × m critères) ; tfns[i][j] : TFN fusionné."""


def topsis_closeness(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    norm = np.sqrt((x ** 2).sum(axis=0))
    v = x / np.where(norm > 0, norm, 1.0) * weights
    best, worst = v.max(axis=0), v.min(axis=0)
    d_pos = np.sqrt(((v - best) ** 2).sum(axis=1))
    d_neg = np.sqrt(((v - worst) ** 2).sum(axis=1))
    denom = d_pos + d_neg
    return np.where(denom > 0, d_neg / np.where(denom > 0, denom, 1), 0.5)


class TopsisAggregator(Aggregator):
    """TOPSIS classique (critères tous « bénéfice »), normalisation vectorielle."""

    name = "TOPSIS"

    def aggregate(self, crisp, low, high, tfns, weights):
        cc = topsis_closeness(crisp, weights)
        lo = low @ weights
        hi = high @ weights
        det = [{"utilite_ponderee": float(crisp[i] @ weights)} for i in range(len(crisp))]
        return AggregationOutput(cc, lo, hi, det)


class FuzzyTopsisAggregator(Aggregator):
    """Fuzzy TOPSIS (Chen 2000) : normalisation par max c_j, distance vertex, FPIS/FNIS empiriques."""

    name = "Fuzzy TOPSIS"

    def aggregate(self, crisp, low, high, tfns, weights):
        n, m = len(tfns), len(weights)
        arr = np.array([[t.as_tuple() for t in row] for row in tfns])  # n × m × 3
        cmax = arr[:, :, 2].max(axis=0)
        v = arr / np.where(cmax > 0, cmax, 1.0)[None, :, None] * weights[None, :, None]
        fpis = v[:, :, 2].max(axis=0)
        fnis = v[:, :, 0].min(axis=0)
        d_pos = np.zeros(n); d_neg = np.zeros(n)
        for j in range(m):
            d_pos += np.sqrt(((v[:, j, :] - fpis[j]) ** 2).mean(axis=1))
            d_neg += np.sqrt(((v[:, j, :] - fnis[j]) ** 2).mean(axis=1))
        cc = d_neg / np.maximum(d_pos + d_neg, 1e-12)
        det = [{"utilite_ponderee": float(crisp[i] @ weights)} for i in range(n)]
        return AggregationOutput(cc, low @ weights, high @ weights, det)


class ChoquetAggregator(Aggregator):
    """Intégrale de Choquet 2-additive (interactions entre critères)."""

    name = "Choquet"

    def __init__(self, capacity: TwoAdditiveCapacity | None = None):
        self.capacity = capacity

    def aggregate(self, crisp, low, high, tfns, weights):
        cap = self.capacity or TwoAdditiveCapacity(np.asarray(weights, float), {})
        s = cap(crisp)
        det = [{"interactions": {f"{CRITERIA[i]}×{CRITERIA[j]}": round(v, 3)
                                 for (i, j), v in cap.interactions.items() if abs(v) > 1e-3}}
               for _ in range(len(crisp))]
        # la Choquet est monotone : appliquée aux bornes, elle donne des bornes valides
        return AggregationOutput(s, cap(low), cap(high), det)


# ---------------------------------------------------------------------------
# Mamdani
# ---------------------------------------------------------------------------

INPUT_TERMS = {"Bas": (0.0, 0.0, 0.5), "Moyen": (0.2, 0.5, 0.8), "Élevé": (0.5, 1.0, 1.0)}
OUTPUT_TERMS = {"Faible": (0.0, 0.0, 1 / 3), "Moyen": (0.0, 1 / 3, 2 / 3),
                "Bon": (1 / 3, 2 / 3, 1.0), "Excellent": (2 / 3, 1.0, 1.0)}

# (antécédents {critère: terme}, conséquent)
DEFAULT_RULES: list[tuple[dict[str, str], str]] = [
    ({"technique": "Élevé", "adequation": "Élevé"}, "Excellent"),
    ({"technique": "Élevé", "soft_skills": "Élevé"}, "Excellent"),
    ({"technique": "Élevé", "soft_skills": "Moyen"}, "Bon"),
    ({"technique": "Moyen", "soft_skills": "Élevé"}, "Bon"),
    ({"technique": "Élevé", "experience": "Bas"}, "Bon"),
    ({"technique": "Moyen", "experience": "Élevé"}, "Bon"),
    ({"experience": "Élevé", "adequation": "Élevé"}, "Bon"),
    ({"technique": "Moyen", "soft_skills": "Moyen"}, "Moyen"),
    ({"adequation": "Moyen", "experience": "Moyen"}, "Moyen"),
    ({"soft_skills": "Bas"}, "Moyen"),
    ({"adequation": "Bas"}, "Moyen"),
    ({"technique": "Moyen", "soft_skills": "Bas"}, "Faible"),
    ({"technique": "Bas"}, "Faible"),
    ({"soft_skills": "Bas", "adequation": "Bas"}, "Faible"),
]


def rule_text(rule: tuple[dict[str, str], str]) -> str:
    ante, cons = rule
    cond = " ET ".join(f"{CRITERIA_LABELS[c].lower()} est {t}" for c, t in ante.items())
    return f"SI {cond} ALORS candidat est {cons}"


class MamdaniAggregator(Aggregator):
    """Inférence de Mamdani : ET = min, implication = min, agrégation = max, défuzzification = centroïde."""

    name = "Mamdani"

    def __init__(self, rules=None, resolution: int = 201):
        self.rules = rules or DEFAULT_RULES
        self.y = np.linspace(0, 1, resolution)
        self.out_mf = {k: tri_membership(self.y, *v) for k, v in OUTPUT_TERMS.items()}

    def infer(self, x: np.ndarray) -> tuple[float, list[tuple[int, float]]]:
        agg = np.zeros_like(self.y)
        acts = []
        for r_idx, (ante, cons) in enumerate(self.rules):
            strength = min(float(tri_membership(x[CRITERIA.index(c)], *INPUT_TERMS[t]))
                           for c, t in ante.items())
            if strength > 0:
                acts.append((r_idx, strength))
                agg = np.maximum(agg, np.minimum(strength, self.out_mf[cons]))
        if agg.sum() <= 0:
            return 0.5, acts
        return float((agg * self.y).sum() / agg.sum()), acts

    def aggregate(self, crisp, low, high, tfns, weights):
        n = len(crisp)
        s = np.zeros(n); lo = np.zeros(n); hi = np.zeros(n)
        det = []
        for i in range(n):
            s[i], acts = self.infer(crisp[i])
            a, _ = self.infer(low[i])
            b, _ = self.infer(high[i])
            lo[i], hi[i] = min(a, b, s[i]), max(a, b, s[i])
            acts.sort(key=lambda t: -t[1])
            det.append({"regles": [(r, round(v, 3)) for r, v in acts]})
        return AggregationOutput(s, lo, hi, det)
