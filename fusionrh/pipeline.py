"""Pipeline de décision : fusion (couche 4) → agrégation/TOPSIS (couche 5) → supervision humaine (couche 6).

Usage minimal :
    >>> system = make_system("Yager", weights, reliabilities)
    >>> ranking = system.evaluate(scenario)
    >>> print(ranking.explain(0))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .aggregation import (DEFAULT_RULES, Aggregator, ChoquetAggregator, FuzzyTopsisAggregator,
                          MamdaniAggregator, TopsisAggregator, TwoAdditiveCapacity, rule_text)
from .config import CLASSES, CRITERIA, CRITERIA_LABELS
from .fusion import CriterionFusion, FusionOperator, get_operator
from .fusion.fuzzy_ops import WeightedFuzzyMean
from .fuzzy import nearest_class

METHOD_NAMES = ["Dempster", "Yager", "Dubois-Prade", "Murphy", "Moyenne floue",
                "OWA (la plupart)", "OWA (au moins un)", "Choquet", "Mamdani"]
EVIDENTIAL_METHODS = METHOD_NAMES[:4]
FUZZY_METHODS = METHOD_NAMES[4:]


# ---------------------------------------------------------------------------
# Supervision humaine
# ---------------------------------------------------------------------------

@dataclass
class Supervisor:
    """Renvoie au jury humain les candidats à fort conflit ou forte ignorance."""

    tau_conflict: float = 0.6
    tau_ignorance: float = 0.45

    def reasons(self, conflict: float, ignorance: float) -> list[str]:
        r = []
        if conflict > self.tau_conflict:
            r.append(f"conflit élevé entre agents ({conflict:.2f} > {self.tau_conflict:.2f})")
        if ignorance > self.tau_ignorance:
            r.append(f"information insuffisante (ignorance {ignorance:.2f} > {self.tau_ignorance:.2f})")
        return r

    @staticmethod
    def fit(conflicts: np.ndarray, ignorances: np.ndarray, hard: np.ndarray) -> "Supervisor":
        """Choisit les seuils maximisant le F1 de détection des cas difficiles (jeu de calibration)."""
        best, best_f1 = Supervisor(), -1.0
        cq = np.unique(np.quantile(conflicts, np.linspace(0.4, 1.0, 31)))
        iq = np.unique(np.quantile(ignorances, np.linspace(0.4, 1.0, 31)))
        for tc in cq:
            for ti in iq:
                pred = (conflicts > tc) | (ignorances > ti)
                tp = np.sum(pred & hard); fp = np.sum(pred & ~hard); fn = np.sum(~pred & hard)
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                if f1 > best_f1 + 1e-12:
                    best_f1, best = f1, Supervisor(float(tc), float(ti))
        return best


# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    cid: int
    score: float
    low: float
    high: float
    conflict: float
    ignorance: float
    criteria: list[CriterionFusion]
    details: dict
    flag_reasons: list[str] = field(default_factory=list)
    rank: int = 0

    @property
    def flagged(self) -> bool:
        return bool(self.flag_reasons)


@dataclass
class RankingResult:
    method: str
    results: list[CandidateResult]
    elapsed: float = 0.0
    rules: Optional[list] = None

    @property
    def scores(self) -> np.ndarray:
        return np.array([r.score for r in self.results])

    @property
    def ranks(self) -> np.ndarray:
        return np.array([r.rank for r in self.results])

    @property
    def order(self) -> np.ndarray:
        """Indices des candidats du meilleur au moins bon."""
        return np.argsort(self.ranks)

    @property
    def flagged(self) -> np.ndarray:
        return np.array([r.flagged for r in self.results])

    @property
    def conflicts(self) -> np.ndarray:
        return np.array([r.conflict for r in self.results])

    @property
    def ignorances(self) -> np.ndarray:
        return np.array([r.ignorance for r in self.results])

    def top(self, k: int) -> set[int]:
        return set(self.order[:k].tolist())

    def explain(self, i: int) -> str:
        return explain_candidate(self, i)


def _rank(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Système de décision
# ---------------------------------------------------------------------------

class DecisionSystem:
    """Assemble un opérateur de fusion interchangeable, un agrégateur et un superviseur."""

    def __init__(self, name: str, operator: FusionOperator, aggregator: Aggregator,
                 weights: np.ndarray, reliabilities: Mapping[str, float] | None = None,
                 supervisor: Supervisor | None = None):
        self.name = name
        self.operator = operator
        self.aggregator = aggregator
        self.weights = np.asarray(weights, float)
        self.reliabilities = dict(reliabilities or {})
        self.supervisor = supervisor or Supervisor()

    def fuse_scenario(self, scenario) -> list[list[CriterionFusion]]:
        names = scenario.agent_names()
        rel = [self.reliabilities.get(n, 1.0) for n in names]
        out = []
        for ops in scenario.opinions:
            row = []
            for j in range(len(CRITERIA)):
                items = [(o, rel[a]) for (a, jj), o in sorted(ops.items()) if jj == j]
                row.append(self.operator.fuse([o for o, _ in items], [r for _, r in items]))
            out.append(row)
        return out

    def evaluate(self, scenario) -> RankingResult:
        t0 = time.perf_counter()
        fused = self.fuse_scenario(scenario)
        crisp = np.array([[f.crisp for f in row] for row in fused])
        low = np.array([[f.low for f in row] for row in fused])
        high = np.array([[f.high for f in row] for row in fused])
        tfns = [[f.tfn for f in row] for row in fused]
        agg = self.aggregator.aggregate(crisp, low, high, tfns, self.weights)
        results = []
        for i, row in enumerate(fused):
            conflict = max(f.conflict for f in row)
            ign = float(np.dot([f.ignorance for f in row], self.weights))
            results.append(CandidateResult(scenario.candidates[i].cid, float(agg.scores[i]),
                                           float(agg.lower[i]), float(agg.upper[i]), conflict, ign,
                                           row, agg.details[i], self.supervisor.reasons(conflict, ign)))
        for r, rk in zip(results, _rank(np.array([r.score for r in results]))):
            r.rank = int(rk)
        rules = getattr(self.aggregator, "rules", None)
        return RankingResult(self.name, results, time.perf_counter() - t0, rules)


def make_system(name: str, weights: np.ndarray, reliabilities: Mapping[str, float] | None = None,
                capacity: TwoAdditiveCapacity | None = None,
                supervisor: Supervisor | None = None) -> DecisionSystem:
    """Fabrique une méthode complète à partir de son nom (voir METHOD_NAMES)."""
    if name in ("Dempster", "Yager", "Dubois-Prade", "Murphy"):
        op, agg = get_operator(name), TopsisAggregator()
    elif name in ("Moyenne floue", "OWA (la plupart)", "OWA (au moins un)"):
        op, agg = get_operator(name), FuzzyTopsisAggregator()
    elif name == "Choquet":
        op, agg = WeightedFuzzyMean(), ChoquetAggregator(capacity)
    elif name == "Mamdani":
        op, agg = WeightedFuzzyMean(), MamdaniAggregator()
    else:
        raise KeyError(f"méthode inconnue : {name}")
    return DecisionSystem(name, op, agg, weights, reliabilities, supervisor)


# ---------------------------------------------------------------------------
# Explication lisible
# ---------------------------------------------------------------------------

def explain_candidate(ranking: RankingResult, i: int) -> str:
    r = ranking.results[i]
    lines = [f"Candidat #{r.cid} — rang {r.rank}/{len(ranking.results)} ({ranking.method})"]
    if "utilite_ponderee" in r.details:
        lines.append(f"  Indice de proximité TOPSIS : {r.score:.3f} ; utilité pondérée "
                     f"{r.details['utilite_ponderee']:.2f} ∈ [croyance {r.low:.2f} ; plausibilité {r.high:.2f}]")
    else:
        lines.append(f"  Score global : {r.score:.3f} ∈ [{r.low:.2f} ; {r.high:.2f}] (bornes basse / haute)")
    for j, f in enumerate(r.criteria):
        lab = CLASSES[nearest_class(f.crisp)]
        line = f"  • {CRITERIA_LABELS[CRITERIA[j]]} : {lab} ({f.crisp:.2f}, [{f.low:.2f} ; {f.high:.2f}])"
        if f.contributions:
            vals = [v for _, v, _ in f.contributions]
            hi_a = max(f.contributions, key=lambda t: t[1]); lo_a = min(f.contributions, key=lambda t: t[1])
            gap = max(vals) - min(vals) if len(vals) >= 2 else 0.0
            if gap > 0.45:
                line += (f" — DÉSACCORD : {hi_a[0]} ({CLASSES[nearest_class(hi_a[1])]}) vs "
                         f"{lo_a[0]} ({CLASSES[nearest_class(lo_a[1])]})")
            elif gap > 0.25:
                line += f" — léger écart ({hi_a[0]} plus favorable que {lo_a[0]})"
            else:
                line += " — agents en accord (" + ", ".join(a for a, _, _ in f.contributions) + ")"
        else:
            line += " — aucun agent ne s'est prononcé"
        if f.mass is not None:
            line += f" ; K={f.conflict:.2f}"
        if f.note:
            line += f" [{f.note}]"
        lines.append(line)
    if "regles" in r.details and ranking.rules:
        lines.append("  Règles floues activées :")
        for idx, s in r.details["regles"][:4]:
            lines.append(f"    R{idx + 1} (degré {s:.2f}) : {rule_text(ranking.rules[idx])}")
    if "interactions" in r.details and r.details["interactions"]:
        lines.append(f"  Interactions entre critères (Choquet) : {r.details['interactions']}")
    if r.flagged:
        lines.append("  ⚠ RENVOYÉ AU JURY HUMAIN : " + " ; ".join(r.flag_reasons))
    else:
        lines.append("  Décision automatique possible (conflit et ignorance sous les seuils), à valider par le jury.")
    return "\n".join(lines)
