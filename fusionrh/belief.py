"""Théorie des fonctions de croyance (Dempster-Shafer) sur le cadre Θ = {Faible, Moyen, Bon, Excellent}.

Une fonction de masse est un vecteur numpy de taille 2^|Θ| = 16, indexé par masque binaire.
Toutes les règles de combinaison travaillent directement sur ces vecteurs (rapide, vectorisé).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import CLASSES, CLASS_UTILITY, N_CLASSES

N_SUBSETS = 1 << N_CLASSES
EMPTY = 0
THETA = N_SUBSETS - 1

_IDX = np.arange(N_SUBSETS)
AND_TABLE = _IDX[:, None] & _IDX[None, :]
OR_TABLE = _IDX[:, None] | _IDX[None, :]
CARD = np.array([bin(i).count("1") for i in range(N_SUBSETS)])
# SUBSET[b, a] = True  <=>  b ⊆ a
SUBSET = AND_TABLE == _IDX[:, None]
# INTERSECTS[b, a] = True <=> b ∩ a ≠ ∅
INTERSECTS = AND_TABLE != 0
# CONTAINS[k, A] = 1 si la classe k appartient à A
CONTAINS = np.array([[(a >> k) & 1 for a in range(N_SUBSETS)] for k in range(N_CLASSES)], dtype=float)
# min / max d'utilité sur chaque sous-ensemble (pour les espérances inférieure / supérieure)
_MIN_U = np.array([CLASS_UTILITY[[k for k in range(N_CLASSES) if (a >> k) & 1]].min() if a else 0.0
                   for a in range(N_SUBSETS)])
_MAX_U = np.array([CLASS_UTILITY[[k for k in range(N_CLASSES) if (a >> k) & 1]].max() if a else 0.0
                   for a in range(N_SUBSETS)])


class TotalConflictError(ValueError):
    """Levée quand le conflit vaut 1 : la règle de Dempster n'est pas définie."""


def mask_of(labels: str | Iterable[str] | int) -> int:
    """Convertit un libellé ('Bon'), une liste de libellés, 'Theta' ou un masque en masque binaire."""
    if isinstance(labels, (int, np.integer)):
        return int(labels)
    if isinstance(labels, str):
        if labels in ("Theta", "Θ", "Ω"):
            return THETA
        labels = [labels]
    mask = 0
    for lab in labels:
        mask |= 1 << CLASSES.index(lab)
    return mask


def label_of(mask: int) -> str:
    if mask == THETA:
        return "Θ"
    if mask == EMPTY:
        return "∅"
    names = [CLASSES[k] for k in range(N_CLASSES) if (mask >> k) & 1]
    return names[0] if len(names) == 1 else "{" + ",".join(names) + "}"


class MassFunction:
    """Fonction de masse m : 2^Θ -> [0, 1]."""

    __slots__ = ("m",)

    def __init__(self, m: Sequence[float] | np.ndarray):
        arr = np.asarray(m, dtype=float)
        if arr.shape != (N_SUBSETS,):
            raise ValueError(f"une masse doit avoir {N_SUBSETS} composantes")
        self.m = arr

    # ---- constructeurs -------------------------------------------------
    @classmethod
    def vacuous(cls) -> "MassFunction":
        m = np.zeros(N_SUBSETS)
        m[THETA] = 1.0
        return cls(m)

    @classmethod
    def categorical(cls, labels) -> "MassFunction":
        m = np.zeros(N_SUBSETS)
        m[mask_of(labels)] = 1.0
        return cls(m)

    @classmethod
    def from_dict(cls, d: Mapping) -> "MassFunction":
        m = np.zeros(N_SUBSETS)
        for key, val in d.items():
            m[mask_of(key)] += float(val)
        return cls(m)

    # ---- propriétés ----------------------------------------------------
    def is_valid(self, tol: float = 1e-9) -> bool:
        return bool(np.all(self.m >= -tol) and abs(self.m.sum() - 1.0) < 1e-6)

    @property
    def conflict(self) -> float:
        return float(self.m[EMPTY])

    @property
    def ignorance(self) -> float:
        return float(self.m[THETA])

    def bel(self, labels) -> float:
        a = mask_of(labels)
        return float(self.m[1:][SUBSET[1:, a]].sum())

    def pl(self, labels) -> float:
        a = mask_of(labels)
        return float(self.m[INTERSECTS[:, a]].sum())

    def betp(self) -> np.ndarray:
        """Probabilité pignistique BetP (Smets)."""
        denom = 1.0 - self.m[EMPTY]
        if denom <= 1e-12:
            return np.full(N_CLASSES, 1.0 / N_CLASSES)
        share = np.where(CARD > 0, self.m / np.maximum(CARD, 1), 0.0)
        return CONTAINS @ share / denom

    def expected_utility(self) -> float:
        return float(self.betp() @ CLASS_UTILITY)

    def lower_expectation(self) -> float:
        """Espérance inférieure (au sens de Dempster) — borne « croyance »."""
        denom = 1.0 - self.m[EMPTY]
        return float((self.m[1:] @ _MIN_U[1:]) / denom) if denom > 1e-12 else 0.0

    def upper_expectation(self) -> float:
        """Espérance supérieure — borne « plausibilité »."""
        denom = 1.0 - self.m[EMPTY]
        return float((self.m[1:] @ _MAX_U[1:]) / denom) if denom > 1e-12 else 1.0

    def discount(self, alpha: float) -> "MassFunction":
        """Affaiblissement de Shafer : m^α(A) = α m(A), m^α(Θ) = 1 − α + α m(Θ)."""
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out = alpha * self.m
        out[THETA] += 1.0 - alpha
        return MassFunction(out)

    def focal(self, tol: float = 1e-9) -> list[tuple[int, float]]:
        return [(int(a), float(v)) for a, v in enumerate(self.m) if v > tol]

    def to_dict(self, digits: int = 3) -> dict[str, float]:
        return {label_of(a): round(v, digits) for a, v in self.focal() if round(v, digits) > 0}

    def __repr__(self) -> str:
        return f"MassFunction({self.to_dict()})"


# ---------------------------------------------------------------------------
# Règles de combinaison (opèrent sur des tableaux numpy de taille 16)
# ---------------------------------------------------------------------------

def _arr(m) -> np.ndarray:
    return m.m if isinstance(m, MassFunction) else np.asarray(m, dtype=float)


def conjunctive(m1, m2) -> np.ndarray:
    """Combinaison conjonctive non normalisée (règle de Smets) : la masse du conflit va sur ∅."""
    prod = np.outer(_arr(m1), _arr(m2)).ravel()
    return np.bincount(AND_TABLE.ravel(), weights=prod, minlength=N_SUBSETS)


def conjunctive_all(masses: Sequence) -> np.ndarray:
    out = _arr(masses[0]).copy()
    for m in masses[1:]:
        out = conjunctive(out, m)
    return out


def conflict_degree(masses: Sequence) -> float:
    """Degré de conflit K = m_∩(∅) de la combinaison conjonctive de toutes les sources."""
    if len(masses) < 2:
        return 0.0
    return float(conjunctive_all(masses)[EMPTY])


def dempster(masses: Sequence) -> np.ndarray:
    """Règle de Dempster (n sources) : combinaison conjonctive puis renormalisation par 1 − K."""
    r = conjunctive_all(masses)
    k = r[EMPTY]
    if k >= 1.0 - 1e-12:
        raise TotalConflictError("conflit total (K = 1)")
    r[EMPTY] = 0.0
    return r / (1.0 - k)


def yager(masses: Sequence) -> np.ndarray:
    """Règle de Yager (1987) : le conflit est transféré à l'ignorance Θ."""
    r = conjunctive_all(masses)
    r[THETA] += r[EMPTY]
    r[EMPTY] = 0.0
    return r


def dubois_prade_pair(m1, m2) -> np.ndarray:
    """Règle de Dubois-Prade : m(A∩B) si non vide, sinon la masse va à A∪B."""
    prod = np.outer(_arr(m1), _arr(m2)).ravel()
    target = np.where(AND_TABLE == 0, OR_TABLE, AND_TABLE).ravel()
    return np.bincount(target, weights=prod, minlength=N_SUBSETS)


def dubois_prade(masses: Sequence) -> np.ndarray:
    """Application séquentielle de la règle de Dubois-Prade (approximation usuelle à n sources :
    la règle n'étant pas associative, l'ordre des sources est fixé par l'ordre des agents)."""
    out = _arr(masses[0]).copy()
    for m in masses[1:]:
        out = dubois_prade_pair(out, m)
    return out


def murphy(masses: Sequence, weights: Sequence[float] | None = None) -> np.ndarray:
    """Moyenne de Murphy (2000) : on moyenne les masses puis on combine la moyenne n−1 fois (Dempster)."""
    arrs = np.array([_arr(m) for m in masses])
    w = np.ones(len(arrs)) if weights is None else np.asarray(weights, dtype=float)
    avg = (w[:, None] * arrs).sum(axis=0) / w.sum()
    return dempster([avg] * len(arrs)) if len(arrs) > 1 else avg
