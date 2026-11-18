"""Calibration des agents (couche 3) sur un jeu de calibration DISTINCT du jeu de test.

* Fiabilité : taux d'accord (classe argmax BetP = vraie classe), corrigé du hasard, puis pénalisé
  par la surconfiance mesurée (confiance moyenne − exactitude).
* Calibration des masses : score de Brier et diagramme de fiabilité sur BetP, ECE.
* Variante : optimisation directe des coefficients α minimisant l'erreur de fusion (descente
  par coordonnées sur une grille).
* Agent LLM : stabilité entre exécutions et écart confiance exprimée / exactitude.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config import N_CLASSES

RELIABILITY_MODES = ("égales", "déclarées", "calibrées")
EQUAL_RELIABILITY = 0.8


@dataclass
class AgentCalibration:
    name: str
    n: int
    accuracy: float
    brier: float
    ece: float
    mean_confidence: float
    overconfidence: float
    reliability: float
    bins: list[tuple[float, float, int]] = field(default_factory=list)  # (confiance moy., exactitude, effectif)
    group_gap: float = 0.0   # biais moyen (groupe B − groupe A) de l'avis perçu, mesuré sur la calibration

    @property
    def overconfident(self) -> bool:
        return self.overconfidence > 0.10

    @property
    def biased(self) -> bool:
        return abs(self.group_gap) > 0.05


def _diagram(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.25, 1.0, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if sel.any():
            c, a = float(conf[sel].mean()), float(correct[sel].mean())
            bins.append((c, a, int(sel.sum())))
            ece += sel.mean() * abs(c - a)
    return bins, float(ece)


def calibrate_agents(scenario) -> dict[str, AgentCalibration]:
    """Mesure exactitude, Brier, ECE, surconfiance et fiabilité de chaque agent."""
    out = {}
    for a_idx, agent in enumerate(scenario.agents):
        probs, truth, resid = [], [], {"A": [], "B": []}
        for i, c in enumerate(scenario.candidates):
            for (a, j), o in scenario.opinions[i].items():
                if a == a_idx and not o.abstained:
                    probs.append(o.mass.betp())
                    truth.append(c.true_classes[j])
                    if not o.faulty:
                        resid[c.group].append(o.crisp - c.true_scores[j])
        gap = (float(np.mean(resid["B"])) - float(np.mean(resid["A"]))
               if resid["A"] and resid["B"] else 0.0)
        if not probs:
            out[agent.name] = AgentCalibration(agent.name, 0, 0.25, 0.75, 0.0, 0.25, 0.0, 0.05)
            continue
        p = np.array(probs); t = np.array(truth)
        onehot = np.eye(N_CLASSES)[t]
        pred = p.argmax(axis=1)
        correct = (pred == t).astype(float)
        conf = p.max(axis=1)
        acc = float(correct.mean())
        brier = float(((p - onehot) ** 2).sum(axis=1).mean())
        bins, ece = _diagram(conf, correct)
        over = max(0.0, float(conf.mean()) - acc)
        rel_acc = np.clip((acc - 1 / N_CLASSES) / (1 - 1 / N_CLASSES), 0.05, 1.0)
        rel = float(np.clip(rel_acc * (1.0 - over), 0.05, 1.0))
        out[agent.name] = AgentCalibration(agent.name, len(t), acc, brier, ece, float(conf.mean()),
                                           over, rel, bins, gap)
    return out


def fairness_adjusted(rel: Mapping[str, float], calibration: Mapping[str, AgentCalibration],
                      scale: float = 0.15) -> dict[str, float]:
    """Audit d'équité : un agent dont l'écart moyen entre groupes est |g| est affaibli d'un facteur
    max(0.1, 1 − |g|/scale). L'attribut protégé n'est utilisé QUE pour l'audit sur la calibration,
    jamais pour évaluer un candidat (cf. AI Act art. 10(5))."""
    return {n: float(r * max(0.1, 1 - abs(calibration[n].group_gap) / scale)) if n in calibration else r
            for n, r in rel.items()}


def reliabilities_for(mode: str, scenario, calibration: Mapping[str, AgentCalibration] | None = None,
                      optimized: Mapping[str, float] | None = None) -> dict[str, float]:
    """Vecteur de fiabilités selon l'expérience clé : égales / déclarées / calibrées / optimisées."""
    if mode == "égales":
        return {a.name: EQUAL_RELIABILITY for a in scenario.agents}
    if mode == "déclarées":
        return {a.name: a.profile.declared_reliability for a in scenario.agents}
    if mode == "calibrées":
        if calibration is None:
            raise ValueError("calibration requise")
        return {a.name: calibration[a.name].reliability for a in scenario.agents}
    if mode == "optimisées":
        if optimized is None:
            raise ValueError("fiabilités optimisées requises")
        return dict(optimized)
    raise KeyError(mode)


def optimize_alpha(cal_scenario, operator_name: str = "Dempster",
                   grid: np.ndarray | None = None, passes: int = 2,
                   start: Mapping[str, float] | None = None) -> dict[str, float]:
    """Optimise les coefficients d'affaiblissement α qui minimisent l'erreur quadratique entre
    l'évaluation fusionnée par critère et le score vrai (descente par coordonnées sur une grille)."""
    from .fusion import get_operator  # import local pour éviter un cycle

    grid = np.linspace(0.1, 1.0, 10) if grid is None else grid
    op = get_operator(operator_name)
    names = cal_scenario.agent_names()
    alpha = {n: (start or {}).get(n, EQUAL_RELIABILITY) for n in names}
    # pré-indexation des avis par critère
    items = []
    for i, c in enumerate(cal_scenario.candidates):
        for j in range(len(c.true_scores)):
            ops = [(a, o) for (a, jj), o in sorted(cal_scenario.opinions[i].items()) if jj == j]
            items.append((c.true_scores[j], ops))

    def loss(al):
        err = 0.0
        for truth, ops in items:
            f = op.fuse([o for _, o in ops], [al[names[a]] for a, _ in ops])
            err += (f.crisp - truth) ** 2
        return err / len(items)

    best = loss(alpha)
    for _ in range(passes):
        for n in names:
            for g in grid:
                trial = dict(alpha); trial[n] = float(g)
                l = loss(trial)
                if l < best - 1e-12:
                    best, alpha = l, trial
    return alpha


def llm_stability(scenario, agent_name: str, runs: int = 5) -> dict[str, float]:
    """Stabilité d'un agent (LLM) sur plusieurs exécutions + écart confiance/exactitude."""
    a_idx = scenario.agent_names().index(agent_name)
    vals = []
    for r in range(runs):
        sc = scenario.regenerate(run_id=r)
        vals.append([o.crisp for i in range(sc.n) for (a, j), o in sorted(sc.opinions[i].items())
                     if a == a_idx and not o.abstained])
    m = min(len(v) for v in vals)
    arr = np.array([v[:m] for v in vals])
    cal = calibrate_agents(scenario)[agent_name]
    labels = np.array([[round(x * 3) for x in v] for v in arr])
    agree = float(np.mean([np.all(labels[:, k] == labels[0, k]) for k in range(m)]))
    return {"std_moyen_entre_executions": float(arr.std(axis=0).mean()),
            "accord_label_entre_executions": agree,
            "confiance_moyenne": cal.mean_confidence, "exactitude": cal.accuracy,
            "ecart_confiance_exactitude": cal.mean_confidence - cal.accuracy}
