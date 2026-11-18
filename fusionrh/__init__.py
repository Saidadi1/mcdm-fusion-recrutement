"""fusionrh — Système multi-agents d'aide à la sélection de candidats fondé sur le MCDM.

Étude comparative d'opérateurs de fusion (Dempster, Yager, Dubois-Prade, Murphy, moyenne floue,
OWA, Choquet, Mamdani) avec calibration des agents et évaluation sur scénarios synthétiques.
"""
from .belief import MassFunction
from .fuzzy import TFN
from .generator import ScenarioConfig, generate_scenario, make_config, SCENARIOS
from .pipeline import METHOD_NAMES, DecisionSystem, Supervisor, make_system
from .weighting import ahp_weights, bwm_weights

__version__ = "1.0.0"
__all__ = ["MassFunction", "TFN", "ScenarioConfig", "generate_scenario", "make_config", "SCENARIOS",
           "METHOD_NAMES", "DecisionSystem", "Supervisor", "make_system", "ahp_weights", "bwm_weights"]
