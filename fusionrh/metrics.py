"""Critères d'évaluation des opérateurs : exactitude, conflit, robustesse, sensibilité, équité, coût."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import kendalltau, mannwhitneyu, spearmanr


def kendall(pred: np.ndarray, truth: np.ndarray) -> float:
    t = kendalltau(pred, truth).statistic
    return float(0.0 if np.isnan(t) else t)


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    r = spearmanr(pred, truth).statistic
    return float(0.0 if np.isnan(r) else r)


def top_k_precision(pred: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Part des k vrais meilleurs retenus dans le top-k prédit."""
    p = set(np.argsort(-pred, kind="stable")[:k].tolist())
    t = set(np.argsort(-truth, kind="stable")[:k].tolist())
    return len(p & t) / k


def top_k_overlap(scores_a: np.ndarray, scores_b: np.ndarray, k: int = 3) -> float:
    a = set(np.argsort(-scores_a, kind="stable")[:k].tolist())
    b = set(np.argsort(-scores_b, kind="stable")[:k].tolist())
    return len(a & b) / k


def discordant_pairs(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Proportion de paires dont l'ordre relatif s'inverse (inversion de rang)."""
    a = np.sign(scores_a[:, None] - scores_a[None, :])
    b = np.sign(scores_b[:, None] - scores_b[None, :])
    iu = np.triu_indices(len(scores_a), 1)
    return float(np.mean((a[iu] * b[iu]) < 0)) if len(iu[0]) else 0.0


def selection_mask(scores: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(len(scores), dtype=bool)
    m[np.argsort(-scores, kind="stable")[:k]] = True
    return m


def four_fifths(selected: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Ratio d'impact (taux de sélection min / max entre groupes) ; règle des 4/5 si ≥ 0,8."""
    rates = {g: float(selected[groups == g].mean()) if np.any(groups == g) else np.nan
             for g in np.unique(groups)}
    vals = [v for v in rates.values() if not np.isnan(v)]
    ratio = (min(vals) / max(vals)) if vals and max(vals) > 0 else 1.0
    return {"ratio_impact": float(ratio), "respecte_4_5": float(ratio >= 0.8),
            **{f"taux_{g}": v for g, v in rates.items()}}


def detection_scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, bool); truth = np.asarray(truth, bool)
    tp = np.sum(pred & truth); fp = np.sum(pred & ~truth); fn = np.sum(~pred & truth)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": float(prec), "rappel": float(rec), "f1": float(f1),
            "taux_renvoi": float(pred.mean())}


def auc(score: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, bool)
    if truth.all() or (~truth).all():
        return np.nan
    u = mannwhitneyu(score[truth], score[~truth], alternative="two-sided").statistic
    return float(u / (truth.sum() * (~truth).sum()))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Correction de Holm-Bonferroni."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
