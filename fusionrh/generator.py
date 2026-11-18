"""Générateur de scénarios synthétiques paramétrable (calibration et validation).

Principe
--------
1. Chaque candidat reçoit des scores « vrais » latents s_j ∈ [0,1] sur les 4 critères
   (corrélation paramétrable entre technique et expérience) et un attribut protégé (groupe A/B)
   tiré indépendamment des scores (=> toute différence entre groupes est un biais).
2. L'utilité vraie est une intégrale de Choquet 2-additive des s_j (poids vrais + interactions) ;
   avec des interactions nulles, c'est une somme pondérée. Elle définit le **vrai classement**.
3. Chaque agent perçoit s_j avec bruit, biais éventuel, avis erronés (pannes), abstentions,
   puis émet une masse + un nombre flou (voir agents.py).
4. Un sous-ensemble de candidats reçoit un « classement expert » (utilité + bruit humain).
5. Un candidat est un **cas difficile** (devant être renvoyé au jury) si son profil est incomplet
   ou s'il a reçu au moins 2 avis erronés et trompeurs.

Tous les tirages aléatoires sont indexés par (graine, candidat, agent, critère) : on peut donc
modifier un seul facteur (groupe, bruit, agent retiré…) toutes choses égales par ailleurs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from scipy.stats import norm

from .aggregation import choquet_2additive
from .agents import AgentProfile, EvaluatorAgent, Opinion, default_agent_profiles
from .config import N_CRITERIA, TRUE_WEIGHTS
from .fuzzy import nearest_class


@dataclass
class ScenarioConfig:
    """Paramètres d'un scénario (voir docs/GENERATEUR.md pour la description complète)."""

    name: str = "S1 Nominal"
    n_candidates: int = 40
    agents: list[AgentProfile] = field(default_factory=default_agent_profiles)
    true_weights: np.ndarray = field(default_factory=lambda: TRUE_WEIGHTS.copy())
    true_interactions: dict = field(default_factory=dict)   # {(i, j): I_ij}
    corr_tech_exp: float = 0.5        # corrélation latente technique / expérience
    corr_other: float = 0.1           # corrélation entre les autres critères
    p_incomplete: float = 0.05        # proportion de profils incomplets
    abstain_if_incomplete: float = 0.6
    extra_fault: float = 0.0          # augmentation globale de la probabilité de panne (conflit)
    extra_ignorance: float = 0.0      # augmentation globale de l'ignorance des agents
    group_b_ratio: float = 0.5
    expert_subset: int = 12
    expert_noise: float = 0.05
    hard_fault_threshold: int = 2
    description: str = ""

    def effective_agents(self) -> list[AgentProfile]:
        return [a.with_(p_fault=min(1.0, a.p_fault + self.extra_fault),
                        ignorance=min(0.95, a.ignorance + self.extra_ignorance)) for a in self.agents]


@dataclass
class Candidate:
    cid: int
    group: str
    true_scores: np.ndarray
    utility: float
    incomplete: bool

    @property
    def true_classes(self) -> list[int]:
        return [nearest_class(s) for s in self.true_scores]


@dataclass
class Scenario:
    config: ScenarioConfig
    seed: int
    candidates: list[Candidate]
    agents: list[EvaluatorAgent]
    opinions: list[dict[tuple[int, int], Opinion]]  # opinions[i][(agent_idx, critère)]
    run_id: int = 0
    extra_noise: float = 0.0

    # ---- vérité de référence --------------------------------------------
    @property
    def n(self) -> int:
        return len(self.candidates)

    @property
    def utilities(self) -> np.ndarray:
        return np.array([c.utility for c in self.candidates])

    @property
    def groups(self) -> np.ndarray:
        return np.array([c.group for c in self.candidates])

    @property
    def fault_cases(self) -> np.ndarray:
        """Candidats ayant reçu au moins `hard_fault_threshold` avis erronés et trompeurs (conflit)."""
        th = self.config.hard_fault_threshold
        return np.array([sum(o.faulty for o in self.opinions[i].values()) >= th
                         for i in range(len(self.candidates))])

    @property
    def incomplete_cases(self) -> np.ndarray:
        return np.array([c.incomplete for c in self.candidates])

    @property
    def hard_cases(self) -> np.ndarray:
        """Cas devant être renvoyés au jury : profil incomplet OU avis trompeurs multiples."""
        return self.fault_cases | self.incomplete_cases

    def expert_ranking(self) -> tuple[np.ndarray, np.ndarray]:
        """Indices du sous-ensemble annoté et scores de l'expert humain (bruités)."""
        k = min(self.config.expert_subset, self.n)
        rng = np.random.default_rng([self.seed, 777])
        idx = np.arange(k)
        return idx, self.utilities[idx] + rng.normal(0, self.config.expert_noise, k)

    def agent_names(self) -> list[str]:
        return [a.name for a in self.agents]

    # ---- variantes contrôlées -------------------------------------------
    def regenerate(self, run_id: Optional[int] = None, extra_noise: Optional[float] = None) -> "Scenario":
        """Même candidats, nouveaux avis (autre exécution des agents et/ou perturbation)."""
        rid = self.run_id if run_id is None else run_id
        eps = self.extra_noise if extra_noise is None else extra_noise
        ops = [generate_opinions(c, self.agents, self.seed, self.config, rid, eps) for c in self.candidates]
        return replace(self, opinions=ops, run_id=rid, extra_noise=eps)

    def with_flipped_group(self, i: int) -> "Scenario":
        """Contrefactuel : seul l'attribut protégé du candidat i change."""
        cands = list(self.candidates)
        c = cands[i]
        cands[i] = replace(c, group="A" if c.group == "B" else "B")
        ops = list(self.opinions)
        ops[i] = generate_opinions(cands[i], self.agents, self.seed, self.config, self.run_id, self.extra_noise)
        return replace(self, candidates=cands, opinions=ops)

    def without_agent(self, agent_idx: int) -> "Scenario":
        agents = [a for k, a in enumerate(self.agents) if k != agent_idx]
        remap = {old: new for new, old in enumerate(k for k in range(len(self.agents)) if k != agent_idx)}
        ops = [{(remap[a], j): o for (a, j), o in d.items() if a != agent_idx} for d in self.opinions]
        return replace(self, agents=agents, opinions=ops)

    def with_extra_candidates(self, k: int = 1, seed_offset: int = 99991) -> "Scenario":
        """Ajoute k candidats (test d'inversion de rang)."""
        extra = generate_candidates(self.config, self.seed + seed_offset, k, id_offset=self.n)
        ops = [generate_opinions(c, self.agents, self.seed, self.config, self.run_id, self.extra_noise)
               for c in extra]
        return replace(self, candidates=self.candidates + extra, opinions=self.opinions + ops)

    def subset(self, idx) -> "Scenario":
        idx = list(idx)
        return replace(self, candidates=[self.candidates[i] for i in idx],
                       opinions=[self.opinions[i] for i in idx])


def _corr_matrix(cfg: ScenarioConfig) -> np.ndarray:
    c = np.full((N_CRITERIA, N_CRITERIA), cfg.corr_other)
    np.fill_diagonal(c, 1.0)
    c[0, 2] = c[2, 0] = cfg.corr_tech_exp
    return c


def generate_candidates(cfg: ScenarioConfig, seed: int, n: int, id_offset: int = 0) -> list[Candidate]:
    rng = np.random.default_rng([seed, 1])
    z = rng.multivariate_normal(np.zeros(N_CRITERIA), _corr_matrix(cfg), size=n)
    scores = norm.cdf(z)
    groups = np.where(rng.random(n) < cfg.group_b_ratio, "B", "A")
    incomplete = rng.random(n) < cfg.p_incomplete
    util = choquet_2additive(scores, cfg.true_weights, cfg.true_interactions)
    return [Candidate(id_offset + i, str(groups[i]), scores[i], float(util[i]), bool(incomplete[i]))
            for i in range(n)]


def generate_opinions(c: Candidate, agents: list[EvaluatorAgent], seed: int, cfg: ScenarioConfig,
                      run_id: int = 0, extra_noise: float = 0.0) -> dict[tuple[int, int], Opinion]:
    out = {}
    for a_idx, agent in enumerate(agents):
        # identifiant stable de l'agent (hash du nom) : retirer un agent ne change pas les autres tirages
        a_key = sum(ord(ch) * (k + 1) for k, ch in enumerate(agent.name)) % 100003
        for j in agent.profile.criteria:
            rng = np.random.default_rng([seed, c.cid, a_key, j])
            run_rng = np.random.default_rng([seed, c.cid, a_key, j, 1000 + run_id])
            extra_rng = np.random.default_rng([seed, c.cid, a_key, j, 5000 + run_id])
            draw = rng.random()
            force = c.incomplete and draw < cfg.abstain_if_incomplete
            out[(a_idx, j)] = agent.evaluate(c.true_scores[j], c.group, j, rng, run_rng,
                                             force_abstain=force, extra_noise=extra_noise,
                                             extra_rng=extra_rng)
    return out


def generate_scenario(cfg: ScenarioConfig, seed: int = 0, run_id: int = 0) -> Scenario:
    agents = [EvaluatorAgent(p) for p in cfg.effective_agents()]
    cands = generate_candidates(cfg, seed, cfg.n_candidates)
    ops = [generate_opinions(c, agents, seed, cfg, run_id) for c in cands]
    return Scenario(cfg, seed, cands, agents, ops, run_id)


# ---------------------------------------------------------------------------
# Scénarios contrôlés de l'étude
# ---------------------------------------------------------------------------

def _scenario_nominal() -> ScenarioConfig:
    return ScenarioConfig("S1 Nominal", description="Agents fiables, peu de conflit, utilité additive.")


def _scenario_conflict() -> ScenarioConfig:
    return ScenarioConfig("S2 Conflit élevé", extra_fault=0.25,
                          description="Probabilité de panne des agents augmentée de +25 points.")


def _scenario_overconfident() -> ScenarioConfig:
    agents = default_agent_profiles()
    agents[3] = agents[3].with_(noise=0.30, sharpness=5.0, precision=0.98, ignorance=0.01,
                                declared_reliability=0.90)
    return ScenarioConfig("S3 Agent surconfiant", agents=agents,
                          description="L'agent « Analyse CV » est très bruité mais émet des masses quasi catégoriques ; "
                                      "le recruteur lui déclare une fiabilité de 0,9.")


def _scenario_ignorance() -> ScenarioConfig:
    return ScenarioConfig("S4 Ignorance forte", p_incomplete=0.30, extra_ignorance=0.25,
                          description="30 % de profils incomplets et ignorance des agents augmentée de +0,25.")


def _scenario_bias() -> ScenarioConfig:
    agents = default_agent_profiles()
    agents[3] = agents[3].with_(bias_group_b=-0.15)
    agents[1] = agents[1].with_(bias_group_b=-0.08)
    return ScenarioConfig("S5 Agents biaisés", agents=agents,
                          description="Analyse CV (−0,15) et Manager (−0,08) sous-évaluent le groupe B.")


def _scenario_interactions() -> ScenarioConfig:
    return ScenarioConfig("S6 Interactions", true_interactions={(0, 2): -0.20, (1, 3): 0.20},
                          description="Utilité vraie non additive : redondance technique/expérience (I=−0,2), "
                                      "complémentarité soft skills/adéquation (I=+0,2).")


SCENARIOS = {
    "S1": _scenario_nominal,
    "S2": _scenario_conflict,
    "S3": _scenario_overconfident,
    "S4": _scenario_ignorance,
    "S5": _scenario_bias,
    "S6": _scenario_interactions,
}


def make_config(key: str, **overrides) -> ScenarioConfig:
    cfg = SCENARIOS[key]()
    return replace(cfg, **overrides) if overrides else cfg
