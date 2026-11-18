# Système multi-agents d'aide à la sélection de candidats fondé sur le MCDM

Étude comparative des opérateurs de fusion des avis (Dempster-Shafer, Yager, Dubois-Prade, Murphy,
moyenne floue, OWA, Choquet, Mamdani), avec calibration des agents et évaluation sur scénarios synthétiques.

> **Aide à la décision uniquement.** Le système classe, explique et **renvoie au jury humain** les cas à fort
> conflit ou forte ignorance. Le projet n'utilise que des données synthétiques.

## Livrables

| Livrable | Emplacement |
|---|---|
| Code modulaire, opérateurs interchangeables | [fusionrh/](fusionrh/) (interface [fusion/base.py](fusionrh/fusion/base.py)) |
| Tests unitaires (49) | [tests/](tests/) |
| Générateur de scénarios paramétrable et documenté | [fusionrh/generator.py](fusionrh/generator.py) + [docs/GENERATEUR.md](docs/GENERATEUR.md) |
| Rapport (≤ 30 pages) | [livrables/Rapport_Fusion_Avis_MCDM.docx](livrables/Rapport_Fusion_Avis_MCDM.docx) |
| Présentation | [livrables/Presentation_Fusion_Avis_MCDM.pptx](livrables/Presentation_Fusion_Avis_MCDM.pptx) |
| Démonstration | [demo/app.py](demo/app.py) (Streamlit) et [demo/demo_cli.py](demo/demo_cli.py) |
| Résultats bruts et figures | [results/](results/), [figures/](figures/) |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
pytest                                       # tests unitaires
streamlit run demo/app.py                    # démonstration interactive
python demo/demo_cli.py --scenario S2        # démonstration console
python -m experiments.run_experiments        # campagne complète (~30 min, --quick pour un essai)
python -m experiments.fairness_pooled        # équité : taux de sélection cumulés sur les réplications
python -m experiments.figures                # figures
python livrables/build_report.py             # régénère le rapport
python livrables/build_slides.py             # régénère la présentation
powershell -File livrables/export_pdf.ps1    # table des matières + export PDF (Microsoft Office requis)
```

### Dans le code

```python
from fusionrh import make_config, generate_scenario, make_system, ahp_weights
from fusionrh.calibration import calibrate_agents, reliabilities_for
from fusionrh.config import AHP_EXPERT_MATRIX

cfg  = make_config("S3")                                  # agent surconfiant
cal  = generate_scenario(cfg, seed=10_000)                # jeu de calibration
test = generate_scenario(cfg, seed=1)                     # jeu de test
rel  = reliabilities_for("calibrées", test, calibrate_agents(cal))
w    = ahp_weights(AHP_EXPERT_MATRIX).weights

for op in ["Dempster", "Yager", "OWA (la plupart)", "Choquet"]:   # opérateurs interchangeables
    ranking = make_system(op, w, rel).evaluate(test)
    print(op, ranking.order[:3])
print(ranking.explain(0))
```

Ajouter un opérateur : hériter de `FusionOperator`, implémenter `fuse(opinions, reliabilities)`,
l'enregistrer dans `fusionrh/fusion/__init__.py::OPERATORS`.

## Architecture

| Couche | Module |
|---|---|
| 1. Agents évaluateurs (5, dont un LLM simulé) | `agents.py` |
| 2. Incertitude : masse sur Θ ⇄ nombre flou | `belief.py`, `fuzzy.py` |
| 3. Calibration (Brier, ECE, surconfiance, audit d'équité) et poids (AHP, BWM) | `calibration.py`, `weighting.py` |
| 4. Fusion interchangeable | `fusion/` |
| 5. Décision et explication (TOPSIS, fuzzy TOPSIS, Choquet, Mamdani) | `aggregation.py`, `pipeline.py` |
| 6. Supervision humaine (renvoi au jury) | `pipeline.py::Supervisor` |
