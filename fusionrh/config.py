"""Constantes globales du système : cadre de discernement, critères, table de correspondance.

Cadre de discernement  Θ = {Faible, Moyen, Bon, Excellent}  (ordonné).
Chaque sous-ensemble A ⊆ Θ est codé par un masque binaire sur 4 bits :
    bit 0 = Faible, bit 1 = Moyen, bit 2 = Bon, bit 3 = Excellent
    ∅ = 0b0000, Θ = 0b1111.
"""
from __future__ import annotations

import numpy as np

CLASSES: tuple[str, ...] = ("Faible", "Moyen", "Bon", "Excellent")
N_CLASSES = len(CLASSES)

# Utilité (valeur numérique) associée à chaque classe : centres d'une partition de Ruspini sur [0, 1].
CLASS_UTILITY = np.array([0.0, 1 / 3, 2 / 3, 1.0])

# Table de correspondance "masse <-> nombre flou" (documentée dans le rapport, section 3).
# Chaque classe linguistique est un nombre flou triangulaire (a, b, c) sur [0, 1].
CLASS_TFN: dict[int, tuple[float, float, float]] = {
    0: (0.0, 0.0, 1 / 3),      # Faible
    1: (0.0, 1 / 3, 2 / 3),    # Moyen
    2: (1 / 3, 2 / 3, 1.0),    # Bon
    3: (2 / 3, 1.0, 1.0),      # Excellent
}

CRITERIA: tuple[str, ...] = ("technique", "soft_skills", "experience", "adequation")
CRITERIA_LABELS = {
    "technique": "Compétences techniques",
    "soft_skills": "Soft skills",
    "experience": "Expérience",
    "adequation": "Adéquation au poste",
}
N_CRITERIA = len(CRITERIA)

# Matrice de comparaisons par paires AHP « experte » (échelle de Saaty), ordre = CRITERIA.
AHP_EXPERT_MATRIX = np.array([
    [1.0, 2.0, 2.0, 2.0],
    [1 / 2, 1.0, 2.0, 1.0],
    [1 / 2, 1 / 2, 1.0, 1.0],
    [1 / 2, 1.0, 1.0, 1.0],
])
# Variante volontairement incohérente (pour illustrer la réélicitation).
AHP_INCONSISTENT_MATRIX = np.array([
    [1.0, 5.0, 1 / 3, 7.0],
    [1 / 5, 1.0, 6.0, 1 / 4],
    [3.0, 1 / 6, 1.0, 5.0],
    [1 / 7, 4.0, 1 / 5, 1.0],
])

# Vecteurs BWM (Best = technique, Worst = experience).
BWM_BEST = 0
BWM_WORST = 2
BWM_BEST_TO_OTHERS = np.array([1.0, 2.0, 3.0, 2.0])
BWM_OTHERS_TO_WORST = np.array([3.0, 2.0, 1.0, 2.0])

# Poids « vrais » (inconnus du système) servant à la vérité de référence du générateur.
TRUE_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.20])

GROUPS = ("A", "B")  # attribut protégé synthétique (ex. genre) — jamais vu par les opérateurs
