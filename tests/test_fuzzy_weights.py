import numpy as np
import pytest

from fusionrh.belief import MassFunction, mask_of
from fusionrh.config import AHP_EXPERT_MATRIX, AHP_INCONSISTENT_MATRIX, BWM_BEST, BWM_BEST_TO_OTHERS, \
    BWM_OTHERS_TO_WORST, BWM_WORST
from fusionrh.fuzzy import (TFN, importance_owa_weights, mass_to_tfn, owa_weights, q_at_least_one, q_most,
                            ruspini_membership, tfn_to_mass, tri_membership, weighted_tfn_mean)
from fusionrh.weighting import ahp_weights, bwm_weights, perturb_weights


def test_tfn_basic():
    t = TFN(0.2, 0.5, 0.8)
    assert t.centroid == pytest.approx(0.5)
    assert t.alpha_cut(1.0) == pytest.approx((0.5, 0.5))
    with pytest.raises(ValueError):
        TFN(0.5, 0.2, 0.8)


def test_membership_shoulders():
    assert float(tri_membership(0.0, 0.0, 0.0, 1 / 3)) == 1.0
    assert float(tri_membership(1.0, 2 / 3, 1.0, 1.0)) == 1.0
    assert float(tri_membership(0.5, 0.0, 0.5, 1.0)) == 1.0


def test_ruspini_partition():
    for x in np.linspace(0, 1, 37):
        mu = ruspini_membership(x)
        assert mu.sum() == pytest.approx(1.0) and np.count_nonzero(mu) <= 2


def test_correspondence_table():
    assert mass_to_tfn(MassFunction.categorical("Bon")).as_tuple() == pytest.approx((1 / 3, 2 / 3, 1.0))
    assert mass_to_tfn(MassFunction.vacuous()).as_tuple() == pytest.approx((0.0, 0.5, 1.0))
    # aller-retour sur singletons et ignorance totale
    for lab in ("Moyen", "Bon"):
        back = tfn_to_mass(mass_to_tfn(MassFunction.categorical(lab)))
        assert back.m[mask_of(lab)] == pytest.approx(1.0)
    assert tfn_to_mass(TFN(0, 0.5, 1)).ignorance == pytest.approx(1.0)


def test_weighted_mean():
    t = weighted_tfn_mean([TFN(0, 0, 0), TFN(1, 1, 1)], [1, 3])
    assert t.as_tuple() == pytest.approx((0.75, 0.75, 0.75))


def test_owa_quantifiers():
    assert owa_weights(4, q_at_least_one).tolist() == [1, 0, 0, 0]
    w = owa_weights(5, q_most)
    assert w.sum() == pytest.approx(1.0) and w[0] == 0.0
    wi = importance_owa_weights([1, 1, 1, 1], q_most)
    np.testing.assert_allclose(wi, owa_weights(4, q_most))


def test_ahp_consistency():
    r = ahp_weights(AHP_EXPERT_MATRIX)
    assert r.consistent and r.weights.sum() == pytest.approx(1.0)
    assert r.weights.argmax() == 0
    bad = ahp_weights(AHP_INCONSISTENT_MATRIX)
    assert not bad.consistent and bad.consistency > 0.1


def test_ahp_perfectly_consistent():
    w = np.array([0.4, 0.3, 0.2, 0.1])
    r = ahp_weights(w[:, None] / w[None, :])
    np.testing.assert_allclose(r.weights, w, atol=1e-9)
    assert r.consistency == pytest.approx(0.0, abs=1e-9)


def test_bwm():
    r = bwm_weights(BWM_BEST, BWM_WORST, BWM_BEST_TO_OTHERS, BWM_OTHERS_TO_WORST)
    assert r.weights.sum() == pytest.approx(1.0)
    assert r.weights.argmax() == BWM_BEST and r.weights.argmin() == BWM_WORST
    assert r.consistent


def test_perturb_weights():
    w = perturb_weights(np.array([0.4, 0.2, 0.2, 0.2]), 0, 0.2)
    assert w.sum() == pytest.approx(1.0) and w[0] > 0.4
