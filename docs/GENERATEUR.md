# Générateur de scénarios synthétiques — documentation

Module : `fusionrh/generator.py`. Il produit des jeux de **calibration** et de **test** contrôlés,
avec une vérité de référence connue, pour comparer les opérateurs de fusion.

## 1. Modèle génératif

| Étape | Description | Paramètre(s) |
|---|---|---|
| Scores latents | Pour chaque candidat, 4 scores vrais `s_j ∈ [0,1]` (technique, soft skills, expérience, adéquation) obtenus par `Φ(z)`, `z ~ N(0, Σ)` | `corr_tech_exp` (0,5), `corr_other` (0,1) |
| Attribut protégé | Groupe A/B tiré **indépendamment** des scores ⇒ toute différence de sélection est un biais | `group_b_ratio` (0,5) |
| Utilité vraie | Intégrale de Choquet 2-additive : `U = Σ φ_j s_j − ½ Σ I_jk |s_j − s_k|` (somme pondérée si I = 0) | `true_weights`, `true_interactions` |
| Vrai classement | Ordre décroissant de `U` | — |
| Profils incomplets | Une part des candidats a un dossier lacunaire : chaque agent s'abstient avec prob. 0,6 | `p_incomplete`, `abstain_if_incomplete` |
| Avis d'un agent | `o = s_j + biais(groupe) + N(0, σ) + bruit d'exécution` ; avec prob. `p_fault`, `o ~ U(0,1)` (avis erroné) | voir §2 |
| Masse | Appartenances de Ruspini aux 2 classes voisines, accentuées (γ), partagées singletons/paire (q), `m(Θ)=u` | `sharpness`, `precision`, `ignorance` |
| Nombre flou | Table de correspondance `TFN(m) = Σ m(A)·TFN(A)` | — |
| Classement expert | Sur les `expert_subset` premiers candidats : `U + N(0, expert_noise)` | `expert_subset` (12), `expert_noise` (0,05) |
| Cas difficile | Profil incomplet **ou** ≥ `hard_fault_threshold` avis erronés **et trompeurs** (écart ≥ 2 classes) | `hard_fault_threshold` (2) |

## 2. Paramètres d'un agent (`AgentProfile`)

| Paramètre | Rôle | RH | Manager | Psy | CV | LLM |
|---|---|---|---|---|---|---|
| `criteria` | critères couverts | S, A | T, E, A | S, T | E, T | T, S, E, A |
| `noise` (σ) | bruit de perception | 0,10 | 0,08 | 0,12 | 0,15 | 0,12 |
| `p_fault` | prob. d'avis erroné | 0,05 | 0,05 | 0,05 | 0,08 | 0,08 |
| `sharpness` (γ) | > 1 : masses tranchées | 1,0 | 1,5 | 1,0 | 1,2 | 2,5 |
| `precision` (q) | part sur singletons | 0,6 | 0,7 | 0,5 | 0,6 | 0,85 |
| `ignorance` (u) | masse sur Θ | 0,15 | 0,10 | 0,20 | 0,15 | 0,05 |
| `declared_reliability` | fiabilité « à la main » | 0,80 | 0,90 | 0,70 | 0,60 | 0,85 |
| `run_noise` | variabilité entre exécutions | 0 | 0 | 0 | 0 | 0,06 |
| `bias_group_b` | biais contre le groupe B | 0 | 0 | 0 | 0 | 0 |

## 3. Scénarios contrôlés

| Clé | Nom | Modification par rapport à S1 |
|---|---|---|
| S1 | Nominal | — |
| S2 | Conflit élevé | `extra_fault = +0,25` sur tous les agents |
| S3 | Agent surconfiant | Analyse CV : σ = 0,30, γ = 5, q = 0,98, u = 0,01, fiabilité déclarée 0,9 |
| S4 | Ignorance forte | `p_incomplete = 0,30`, `extra_ignorance = +0,25` |
| S5 | Agents biaisés | Analyse CV −0,15 et Manager −0,08 pour le groupe B |
| S6 | Interactions | `I(technique, expérience) = −0,2` (redondance), `I(soft, adéquation) = +0,2` (complémentarité) |

## 4. Reproductibilité et variantes « toutes choses égales par ailleurs »

Chaque tirage est indexé par `(graine, candidat, agent, critère)` et effectué dans un ordre fixe :

* `scenario.with_flipped_group(i)` — **contrefactuel** : seul l'attribut protégé du candidat *i* change ;
* `scenario.without_agent(k)` — retrait d'un agent (les autres avis sont inchangés) ;
* `scenario.with_extra_candidates(k)` — ajout de candidats (test d'inversion de rang) ;
* `scenario.regenerate(run_id, extra_noise)` — nouvelle exécution des agents / perturbation des avis.

## 5. Exemple

```python
from fusionrh import make_config, generate_scenario
cfg = make_config("S3", n_candidates=50, extra_fault=0.1)   # surcharge de paramètres
cal  = generate_scenario(cfg, seed=10_000)                  # jeu de calibration
test = generate_scenario(cfg, seed=1)                       # jeu de test (graine distincte)
test.utilities, test.hard_cases, test.expert_ranking()
```
