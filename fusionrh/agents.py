"""Agents évaluateurs.

Chaque agent couvre un sous-ensemble de critères (son domaine) et exprime, pour chaque candidat
et chaque critère couvert, un avis incertain sous deux formes : une fonction de masse sur Θ et un
nombre flou triangulaire (via la table de correspondance de `fuzzy.py`).

Les agents sont simulés (fondés sur des règles bruitées) ; un agent « LLM simulé » reproduit la
variabilité d'un grand modèle de langage d'une exécution à l'autre (`run_noise`) et sa tendance
à la surconfiance. Un vrai agent LLM peut être branché en implémentant la même méthode `evaluate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from .belief import N_SUBSETS, THETA, MassFunction
from .config import CRITERIA, N_CLASSES
from .fuzzy import TFN, mass_to_tfn, nearest_class, ruspini_membership


@dataclass(frozen=True)
class AgentProfile:
    """Paramètres d'un agent (tous documentés dans le rapport, tableau 2)."""

    name: str
    criteria: tuple[int, ...]          # indices des critères couverts
    noise: float = 0.10                # écart-type du bruit d'observation
    p_fault: float = 0.05              # probabilité d'un avis erroné (observation aléatoire)
    sharpness: float = 1.0             # γ : >1 = masses plus tranchées (surconfiance)
    precision: float = 0.6             # q : part de masse sur singletons vs paire adjacente
    ignorance: float = 0.15            # u : masse moyenne sur Θ (ignorance assumée)
    p_abstain: float = 0.02            # probabilité de ne pas se prononcer (masse vide)
    bias_group_b: float = 0.0          # biais additif appliqué aux candidats du groupe B
    declared_reliability: float = 0.8  # fiabilité « déclarée à la main » par le recruteur
    run_noise: float = 0.0             # variabilité entre exécutions (agent LLM)
    kind: str = "règles"

    def with_(self, **kw) -> "AgentProfile":
        return replace(self, **kw)


@dataclass
class Opinion:
    agent: str
    criterion: int
    mass: MassFunction
    tfn: TFN
    observed: Optional[float]   # valeur latente perçue par l'agent (None si abstention)
    faulty: bool = False        # avis erroné ET trompeur (écart > 1/3 à la vérité)
    abstained: bool = False

    @property
    def crisp(self) -> float:
        return self.mass.expected_utility()

    @property
    def label(self) -> str:
        return ["Faible", "Moyen", "Bon", "Excellent"][int(np.argmax(self.mass.betp()))]


def build_mass(observed: float, sharpness: float, precision: float, ignorance: float) -> MassFunction:
    """Construit la masse d'un avis à partir de la valeur perçue.

    1. appartenances de Ruspini μ aux deux classes adjacentes ;
    2. accentuation p_k ∝ μ_k^γ (γ = sharpness) ;
    3. m({k}) = (1−u)·q·p_k ; m({k, k+1}) = (1−u)(1−q) ; m(Θ) = u.
    """
    mu = ruspini_membership(observed)
    low = min(int(np.floor(np.clip(observed, 0.0, 1.0) * (N_CLASSES - 1))), N_CLASSES - 2)
    p = mu[[low, low + 1]] ** sharpness
    p = p / p.sum() if p.sum() > 0 else np.array([0.5, 0.5])
    u = float(np.clip(ignorance, 0.0, 1.0))
    q = float(np.clip(precision, 0.0, 1.0))
    m = np.zeros(N_SUBSETS)
    m[1 << low] += (1 - u) * q * p[0]
    m[1 << (low + 1)] += (1 - u) * q * p[1]
    m[(1 << low) | (1 << (low + 1))] += (1 - u) * (1 - q)
    m[THETA] += u
    return MassFunction(m)


class EvaluatorAgent:
    """Agent évaluateur fondé sur des règles (perception bruitée de la vérité latente)."""

    def __init__(self, profile: AgentProfile):
        self.profile = profile

    @property
    def name(self) -> str:
        return self.profile.name

    def covers(self, criterion: int) -> bool:
        return criterion in self.profile.criteria

    def evaluate(self, true_score: float, group: str, criterion: int, rng: np.random.Generator,
                 run_rng: np.random.Generator, force_abstain: bool = False,
                 extra_noise: float = 0.0, extra_rng: Optional[np.random.Generator] = None) -> Opinion:
        p = self.profile
        # Tirages dans un ordre fixe (reproductibilité / tests contrefactuels).
        eps = rng.normal(0.0, p.noise)
        fault_draw = rng.random()
        fault_value = rng.random()
        abstain_draw = rng.random()
        ign_jitter = rng.normal(0.0, 0.05)
        run_eps = run_rng.normal(0.0, p.run_noise) if p.run_noise > 0 else 0.0
        extra = extra_rng.normal(0.0, extra_noise) if (extra_rng is not None and extra_noise > 0) else 0.0

        if force_abstain or abstain_draw < p.p_abstain:
            m = MassFunction.vacuous()
            return Opinion(p.name, criterion, m, mass_to_tfn(m), None, False, True)

        bias = p.bias_group_b if group == "B" else 0.0
        faulty = fault_draw < p.p_fault
        obs = fault_value if faulty else true_score + bias + eps + run_eps
        obs = float(np.clip(obs + extra, 0.0, 1.0))
        misleading = faulty and abs(nearest_class(obs) - nearest_class(true_score)) >= 2
        m = build_mass(obs, p.sharpness, p.precision, p.ignorance + ign_jitter)
        return Opinion(p.name, criterion, m, mass_to_tfn(m), obs, misleading, False)


def default_agent_profiles() -> list[AgentProfile]:
    """Les 5 agents du système (3 à 5 exigés par le sujet)."""
    T, S, E, A = range(len(CRITERIA))
    return [
        AgentProfile("RH (entretien)", (S, A), noise=0.10, p_fault=0.05, sharpness=1.0, precision=0.6,
                     ignorance=0.15, declared_reliability=0.80),
        AgentProfile("Manager technique", (T, E, A), noise=0.08, p_fault=0.05, sharpness=1.5,
                     precision=0.7, ignorance=0.10, declared_reliability=0.90),
        AgentProfile("Test psychométrique", (S, T), noise=0.12, p_fault=0.05, sharpness=1.0,
                     precision=0.5, ignorance=0.20, declared_reliability=0.70),
        AgentProfile("Analyse CV", (E, T), noise=0.15, p_fault=0.08, sharpness=1.2, precision=0.6,
                     ignorance=0.15, declared_reliability=0.60),
        AgentProfile("LLM entretien (simulé)", (T, S, E, A), noise=0.12, p_fault=0.08, sharpness=2.5,
                     precision=0.85, ignorance=0.05, declared_reliability=0.85, run_noise=0.06,
                     kind="LLM (simulé)"),
    ]
