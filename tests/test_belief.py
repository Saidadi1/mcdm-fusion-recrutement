import numpy as np
import pytest

from fusionrh.belief import (THETA, MassFunction, TotalConflictError, conflict_degree, conjunctive,
                             dempster, dubois_prade, dubois_prade_pair, mask_of, murphy, yager)


def m(d):
    return MassFunction.from_dict(d)


def test_mask_and_vacuous():
    assert mask_of("Faible") == 1 and mask_of("Excellent") == 8
    assert mask_of(["Bon", "Excellent"]) == 12 and mask_of("Theta") == THETA
    v = MassFunction.vacuous()
    assert v.is_valid() and v.ignorance == 1.0


def test_bel_pl_betp():
    x = m({"Bon": 0.5, ("Bon", "Excellent"): 0.3, "Theta": 0.2})
    assert x.bel("Bon") == pytest.approx(0.5)
    assert x.pl("Bon") == pytest.approx(1.0)
    assert x.bel(["Bon", "Excellent"]) == pytest.approx(0.8)
    assert x.pl("Faible") == pytest.approx(0.2)
    betp = x.betp()
    assert betp.sum() == pytest.approx(1.0)
    assert betp[2] == pytest.approx(0.5 + 0.15 + 0.05)
    # Bel ≤ BetP ≤ Pl
    for k, lab in enumerate(["Faible", "Moyen", "Bon", "Excellent"]):
        assert x.bel(lab) - 1e-9 <= betp[k] <= x.pl(lab) + 1e-9


def test_discounting():
    x = m({"Bon": 1.0}).discount(0.7)
    assert x.m[mask_of("Bon")] == pytest.approx(0.7)
    assert x.ignorance == pytest.approx(0.3)
    assert m({"Bon": 1.0}).discount(0.0).ignorance == pytest.approx(1.0)


def test_vacuous_is_neutral_for_dempster():
    x = m({"Bon": 0.6, "Moyen": 0.3, "Theta": 0.1})
    r = dempster([x.m, MassFunction.vacuous().m])
    np.testing.assert_allclose(r, x.m, atol=1e-12)


def test_dempster_commutative_associative():
    a = m({"Bon": 0.6, "Theta": 0.4}); b = m({"Moyen": 0.3, ("Bon", "Excellent"): 0.5, "Theta": 0.2})
    c = m({"Excellent": 0.2, "Bon": 0.5, "Theta": 0.3})
    np.testing.assert_allclose(dempster([a, b]), dempster([b, a]), atol=1e-12)
    ab_c = dempster([dempster([a, b]), c.m]); abc = dempster([a, b, c])
    np.testing.assert_allclose(ab_c, abc, atol=1e-12)


def test_zadeh_paradox():
    """Exemple de Zadeh : Dempster attribue toute la masse à l'hypothèse jugée quasi impossible."""
    m1 = m({"Excellent": 0.99, "Moyen": 0.01})
    m2 = m({"Faible": 0.99, "Moyen": 0.01})
    assert conflict_degree([m1.m, m2.m]) == pytest.approx(0.9999)
    d = MassFunction(dempster([m1.m, m2.m]))
    assert d.m[mask_of("Moyen")] == pytest.approx(1.0)
    y = MassFunction(yager([m1.m, m2.m]))
    assert y.ignorance == pytest.approx(0.9999)          # Yager : le conflit devient ignorance
    dp = MassFunction(dubois_prade([m1.m, m2.m]))
    assert dp.m[mask_of(["Faible", "Excellent"])] == pytest.approx(0.99 * 0.99)
    assert dp.is_valid()


def test_total_conflict_raises():
    with pytest.raises(TotalConflictError):
        dempster([m({"Bon": 1}).m, m({"Faible": 1}).m])


def test_yager_and_dp_preserve_mass():
    a = m({"Bon": 0.7, "Theta": 0.3}); b = m({"Faible": 0.6, "Moyen": 0.2, "Theta": 0.2})
    for r in (yager([a, b]), dubois_prade_pair(a, b), conjunctive(a, b)):
        assert r.sum() == pytest.approx(1.0)
    assert yager([a, b])[0] == 0.0


def test_murphy_reduces_to_average_combination():
    a = m({"Bon": 0.8, "Theta": 0.2}); b = m({"Moyen": 0.8, "Theta": 0.2})
    r = MassFunction(murphy([a, b]))
    assert r.m[mask_of("Bon")] == pytest.approx(r.m[mask_of("Moyen")])
    assert r.is_valid()


def test_expectations_bracket_betp():
    x = m({("Moyen", "Bon"): 0.5, "Excellent": 0.3, "Theta": 0.2})
    assert x.lower_expectation() <= x.expected_utility() <= x.upper_expectation()
