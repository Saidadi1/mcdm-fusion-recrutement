"""Pondération des critères : AHP (Saaty) et BWM (Rezaei, version linéaire), avec contrôle de cohérence."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

# Indices aléatoires de Saaty
RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
# Indices de cohérence BWM (Rezaei 2015) selon a_BW
BWM_CONSISTENCY_INDEX = {1: 0.0, 2: 0.44, 3: 1.0, 4: 1.63, 5: 2.30, 6: 3.0, 7: 3.73, 8: 4.47, 9: 5.23}


@dataclass
class WeightResult:
    weights: np.ndarray
    consistency: float          # CR (AHP) ou ratio de cohérence ξ/CI (BWM)
    consistent: bool
    method: str
    details: dict


def ahp_weights(matrix: np.ndarray, threshold: float = 0.10) -> WeightResult:
    """Poids AHP par vecteur propre principal ; CR = CI / RI avec CI = (λmax − n)/(n − 1)."""
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    if a.shape != (n, n) or not np.allclose(a * a.T, 1.0, atol=1e-6):
        raise ValueError("la matrice AHP doit être carrée et réciproque (a_ji = 1/a_ij)")
    vals, vecs = np.linalg.eig(a)
    i = int(np.argmax(vals.real))
    lam = float(vals[i].real)
    w = np.abs(vecs[:, i].real)
    w = w / w.sum()
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return WeightResult(w, cr, cr < threshold, "AHP", {"lambda_max": lam, "CI": ci, "RI": ri})


def bwm_weights(best: int, worst: int, best_to_others: np.ndarray, others_to_worst: np.ndarray,
                threshold: float = 0.10) -> WeightResult:
    """BWM linéaire : min ξ  s.c. |w_B − a_Bj w_j| ≤ ξ,  |w_j − a_jW w_W| ≤ ξ,  Σ w = 1, w ≥ 0."""
    ab = np.asarray(best_to_others, dtype=float)
    aw = np.asarray(others_to_worst, dtype=float)
    n = len(ab)
    # variables : w_0..w_{n-1}, ξ
    c = np.zeros(n + 1)
    c[-1] = 1.0
    rows, rhs = [], []
    for j in range(n):
        r = np.zeros(n + 1); r[best] += 1; r[j] -= ab[j]; r[-1] = -1; rows.append(r); rhs.append(0)
        r = np.zeros(n + 1); r[best] -= 1; r[j] += ab[j]; r[-1] = -1; rows.append(r); rhs.append(0)
        r = np.zeros(n + 1); r[j] += 1; r[worst] -= aw[j]; r[-1] = -1; rows.append(r); rhs.append(0)
        r = np.zeros(n + 1); r[j] -= 1; r[worst] += aw[j]; r[-1] = -1; rows.append(r); rhs.append(0)
    a_eq = np.zeros((1, n + 1)); a_eq[0, :n] = 1
    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), A_eq=a_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    w = res.x[:n]
    xi = float(res.x[-1])
    a_bw = int(round(ab[worst]))
    ci = BWM_CONSISTENCY_INDEX.get(a_bw, 5.23)
    ratio = xi / ci if ci > 0 else 0.0
    return WeightResult(w / w.sum(), ratio, ratio < threshold, "BWM", {"xi": xi, "CI": ci})


def perturb_weights(weights: np.ndarray, index: int, delta: float) -> np.ndarray:
    """Multiplie le poids `index` par (1 + delta) puis renormalise (analyse de sensibilité)."""
    w = np.asarray(weights, dtype=float).copy()
    w[index] *= (1.0 + delta)
    return w / w.sum()
