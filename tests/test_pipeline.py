import numpy as np
import pytest

from fusionrh.agents import build_mass
from fusionrh.aggregation import (ChoquetAggregator, MamdaniAggregator, TwoAdditiveCapacity, choquet_2additive,
                                  fit_interactions, topsis_closeness)
from fusionrh.belief import MassFunction
from fusionrh.calibration import calibrate_agents, optimize_alpha, reliabilities_for
from fusionrh.config import TRUE_WEIGHTS
from fusionrh.fusion import OPERATORS, FusionOperator, get_operator
from fusionrh.generator import generate_scenario, make_config
from fusionrh.metrics import discordant_pairs, four_fifths, holm, kendall, top_k_precision
from fusionrh.pipeline import METHOD_NAMES, Supervisor, make_system

W = TRUE_WEIGHTS


@pytest.fixture(scope="module")
def scen():
    return generate_scenario(make_config("S1", n_candidates=20), seed=3)


def test_build_mass_valid():
    for x in np.linspace(0, 1, 11):
        mf = build_mass(x, 2.0, 0.7, 0.1)
        assert mf.is_valid() and mf.ignorance == pytest.approx(0.1)


def test_generator_reproducible(scen):
    again = generate_scenario(make_config("S1", n_candidates=20), seed=3)
    np.testing.assert_allclose(scen.utilities, again.utilities)
    o1 = scen.opinions[0][(1, 0)].mass.m; o2 = again.opinions[0][(1, 0)].mass.m
    np.testing.assert_allclose(o1, o2)


def test_counterfactual_only_changes_bias():
    s = generate_scenario(make_config("S1", n_candidates=10), seed=5)
    i = 0
    flipped = s.with_flipped_group(i)
    # sans agent biaisé, les avis sont strictement identiques
    for k, o in s.opinions[i].items():
        np.testing.assert_allclose(o.mass.m, flipped.opinions[i][k].mass.m)


def test_without_agent(scen):
    s2 = scen.without_agent(0)
    assert len(s2.agents) == 4
    assert all(a < 4 for d in s2.opinions for (a, _) in d)


@pytest.mark.parametrize("name", list(OPERATORS))
def test_operators_share_interface(name, scen):
    op = get_operator(name)
    assert isinstance(op, FusionOperator)
    ops = [o for (a, j), o in sorted(scen.opinions[0].items()) if j == 0]
    f = op.fuse(ops, [0.8] * len(ops))
    assert 0 <= f.crisp <= 1 and f.low <= f.high + 1e-9 and 0 <= f.conflict <= 1


def test_operator_on_empty_list():
    for name in OPERATORS:
        f = get_operator(name).fuse([], [])
        assert f.crisp == pytest.approx(0.5)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_all_methods_rank(name, scen):
    r = make_system(name, W, {a.name: 0.8 for a in scen.agents}).evaluate(scen)
    assert sorted(r.ranks.tolist()) == list(range(1, scen.n + 1))
    assert kendall(r.scores, scen.utilities) > 0.3      # bien meilleur que le hasard
    assert "Candidat" in r.explain(0)


def test_choquet_reduces_to_weighted_sum():
    x = np.random.default_rng(0).random((5, 4))
    np.testing.assert_allclose(choquet_2additive(x, W, {}), x @ W)


def test_choquet_monotone_projection_and_fit():
    cap = TwoAdditiveCapacity(W.copy(), {(0, 2): -0.9}).project_monotone()
    assert cap.is_monotone()
    x = np.random.default_rng(1).random((300, 4))
    y = choquet_2additive(x, W, {(0, 2): -0.2, (1, 3): 0.2})
    fitted = fit_interactions(x, y, W)
    assert fitted.interactions[(0, 2)] == pytest.approx(-0.2, abs=0.02)
    assert fitted.interactions[(1, 3)] == pytest.approx(0.2, abs=0.02)


def test_topsis_dominance():
    x = np.array([[0.9, 0.9], [0.5, 0.5], [0.1, 0.1]])
    cc = topsis_closeness(x, np.array([0.5, 0.5]))
    assert cc[0] > cc[1] > cc[2]


def test_mamdani_rules_fire():
    agg = MamdaniAggregator()
    hi, acts = agg.infer(np.array([0.95, 0.95, 0.9, 0.95]))
    lo, _ = agg.infer(np.array([0.05, 0.1, 0.1, 0.1]))
    assert hi > 0.7 and lo < 0.3 and acts


def test_calibration_detects_overconfident_agent():
    s = generate_scenario(make_config("S3", n_candidates=80), seed=11)
    cal = calibrate_agents(s)
    assert cal["Analyse CV"].overconfident
    assert cal["Analyse CV"].reliability < cal["Manager technique"].reliability
    rel = reliabilities_for("calibrées", s, cal)
    assert set(rel) == set(s.agent_names())


def test_optimize_alpha_improves_loss():
    s = generate_scenario(make_config("S3", n_candidates=15), seed=2)
    alpha = optimize_alpha(s, passes=1, grid=np.array([0.2, 0.6, 1.0]))
    assert alpha["Analyse CV"] <= 0.6


def test_supervisor_fit():
    rng = np.random.default_rng(0)
    hard = rng.random(200) < 0.2
    conflict = np.where(hard, 0.8, 0.2) + rng.normal(0, 0.05, 200)
    sup = Supervisor.fit(conflict, np.zeros(200), hard)
    assert 0.2 < sup.tau_conflict < 0.8


def test_metrics():
    assert top_k_precision(np.array([3, 2, 1.0]), np.array([3, 2, 1.0]), 2) == 1.0
    assert discordant_pairs(np.array([1, 2, 3.0]), np.array([3, 2, 1.0])) == 1.0
    ff = four_fifths(np.array([1, 1, 0, 1, 0, 0], bool), np.array(list("AAABBB")))
    assert ff["ratio_impact"] == pytest.approx(0.5)
    assert holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
