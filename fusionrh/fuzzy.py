"""Nombres flous triangulaires, table de correspondance masse <-> flou, quantificateurs OWA.

Table de correspondance (documentée, section 3 du rapport)
-----------------------------------------------------------
* Classe singleton k  ->  TFN de la partition de Ruspini (config.CLASS_TFN).
* Sous-ensemble A     ->  TFN(A) = (min_{k∈A} a_k, moyenne_{k∈A} b_k, max_{k∈A} c_k).
  En particulier Θ -> (0, 0.5, 1) : l'ignorance totale est le nombre flou le plus large.
* Masse m             ->  TFN(m) = Σ_A m(A) · TFN(A)   (« espérance floue »).
* TFN (a, b, c)       ->  masse : m(Θ) = largeur excédentaire, le reste réparti sur les
  classes adjacentes selon l'appartenance de Ruspini du mode b (transformation inverse approchée).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .belief import N_SUBSETS, THETA, MassFunction
from .config import CLASS_TFN, N_CLASSES


@dataclass(frozen=True)
class TFN:
    """Nombre flou triangulaire (a ≤ b ≤ c)."""

    a: float
    b: float
    c: float

    def __post_init__(self):
        if not (self.a <= self.b + 1e-9 and self.b <= self.c + 1e-9):
            raise ValueError(f"TFN invalide : {self}")

    @property
    def centroid(self) -> float:
        return (self.a + self.b + self.c) / 3.0

    @property
    def width(self) -> float:
        return self.c - self.a

    def alpha_cut(self, alpha: float) -> tuple[float, float]:
        return self.a + alpha * (self.b - self.a), self.c - alpha * (self.c - self.b)

    def membership(self, x):
        return tri_membership(x, self.a, self.b, self.c)

    def __add__(self, other: "TFN") -> "TFN":
        return TFN(self.a + other.a, self.b + other.b, self.c + other.c)

    def scale(self, k: float) -> "TFN":
        return TFN(k * self.a, k * self.b, k * self.c)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)

    def __repr__(self) -> str:
        return f"TFN({self.a:.3f}, {self.b:.3f}, {self.c:.3f})"


def tri_membership(x, a: float, b: float, c: float):
    """Fonction d'appartenance triangulaire (gère les « épaules » a = b ou b = c)."""
    x = np.asarray(x, dtype=float)
    left = (x - a) / (b - a) if b > a else np.where(x >= a, 1.0, 0.0)
    right = (c - x) / (c - b) if c > b else np.where(x <= c, 1.0, 0.0)
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def ruspini_membership(x: float) -> np.ndarray:
    """Appartenance d'une valeur x ∈ [0,1] aux 4 classes (somme = 1, deux classes adjacentes)."""
    x = float(np.clip(x, 0.0, 1.0))
    pos = x * (N_CLASSES - 1)
    low = min(int(math.floor(pos)), N_CLASSES - 2)
    f = pos - low
    mu = np.zeros(N_CLASSES)
    mu[low] = 1.0 - f
    mu[low + 1] = f
    return mu


def nearest_class(x: float) -> int:
    return int(np.argmax(ruspini_membership(x)))


# --- table de correspondance --------------------------------------------------

def _subset_tfn(mask: int) -> TFN:
    if mask == 0:
        return TFN(0.0, 0.5, 1.0)
    ks = [k for k in range(N_CLASSES) if (mask >> k) & 1]
    if mask == THETA:
        return TFN(0.0, 0.5, 1.0)
    return TFN(min(CLASS_TFN[k][0] for k in ks),
               float(np.mean([CLASS_TFN[k][1] for k in ks])),
               max(CLASS_TFN[k][2] for k in ks))


SUBSET_TFN: list[TFN] = [_subset_tfn(a) for a in range(N_SUBSETS)]
_SUBSET_TFN_ARR = np.array([t.as_tuple() for t in SUBSET_TFN])


def mass_to_tfn(m) -> TFN:
    arr = m.m if isinstance(m, MassFunction) else np.asarray(m, dtype=float)
    a, b, c = arr @ _SUBSET_TFN_ARR / max(arr.sum(), 1e-12)
    return TFN(float(a), float(b), float(c))


def tfn_to_mass(t: TFN) -> MassFunction:
    """Transformation inverse approchée TFN -> masse."""
    base_width = 2.0 / 3.0  # largeur d'une classe intérieure (Moyen, Bon)
    ign = float(np.clip((t.width - base_width) / (1.0 - base_width), 0.0, 1.0))
    mu = ruspini_membership(t.b)
    m = np.zeros(N_SUBSETS)
    for k in range(N_CLASSES):
        m[1 << k] += (1.0 - ign) * mu[k]
    m[THETA] += ign
    return MassFunction(m)


def tfn_distance(t1: TFN, t2: TFN) -> float:
    """Distance « vertex » (Chen 2000)."""
    return math.sqrt(((t1.a - t2.a) ** 2 + (t1.b - t2.b) ** 2 + (t1.c - t2.c) ** 2) / 3.0)


def weighted_tfn_mean(tfns: Sequence[TFN], weights: Sequence[float]) -> TFN:
    w = np.asarray(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(tfns))
    arr = np.array([t.as_tuple() for t in tfns])
    a, b, c = (w[:, None] * arr).sum(axis=0) / w.sum()
    return TFN(float(a), float(b), float(c))


# --- quantificateurs linguistiques pour l'OWA ---------------------------------

def q_most(r: float) -> float:
    """« La plupart » (Yager) : Q(r) = 0 si r ≤ 0.3, 1 si r ≥ 0.8, linéaire entre les deux."""
    return float(np.clip((r - 0.3) / 0.5, 0.0, 1.0))


def q_at_least_one(r: float) -> float:
    """« Au moins un » : Q(r) = 1 dès que r > 0 (OWA = max, attitude optimiste)."""
    return 1.0 if r > 1e-12 else 0.0


def q_all(r: float) -> float:
    """« Tous » : Q(r) = 1 seulement si r = 1 (OWA = min, attitude pessimiste)."""
    return 1.0 if r >= 1.0 - 1e-12 else 0.0


QUANTIFIERS: dict[str, Callable[[float], float]] = {
    "most": q_most, "at_least_one": q_at_least_one, "all": q_all,
}


def owa_weights(n: int, quantifier: Callable[[float], float]) -> np.ndarray:
    """Poids OWA w_k = Q(k/n) − Q((k−1)/n)."""
    return np.array([quantifier(k / n) - quantifier((k - 1) / n) for k in range(1, n + 1)])


def importance_owa_weights(importances_sorted: Sequence[float],
                           quantifier: Callable[[float], float]) -> np.ndarray:
    """OWA pondéré par importance (Yager 1996) : w_k = Q(S_k/T) − Q(S_{k−1}/T), arguments triés
    par valeur décroissante, S_k = somme des importances des k premiers arguments."""
    u = np.asarray(importances_sorted, dtype=float)
    total = u.sum()
    if total <= 0:
        u = np.ones_like(u)
        total = u.sum()
    s = np.concatenate([[0.0], np.cumsum(u)]) / total
    return np.array([quantifier(s[k]) - quantifier(s[k - 1]) for k in range(1, len(u) + 1)])
