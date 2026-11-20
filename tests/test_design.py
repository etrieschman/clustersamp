from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

import clustersamp as cs

ALL = np.arange(3)
NONE = np.zeros((3, 2))


def test_toy_by_hand(toy, design):
    visits = cs.visit(toy, design, ALL, NONE)
    assert visits.visit.tolist() == [0, 1, 2]
    assert visits.tree.tolist() == [0, 0, 1]
    assert visits.C.tolist() == [2.0, 2.0, 1.0]
    t, w = cs.totals(toy, visits)
    assert t.tolist() == [1.0, 1.0, 4.0]
    assert w.tolist() == [0.5, 0.5, 1.0]
    assert cs.mu_hat(t) == toy.mu == 2.0  # every pin once: the expectation of one visit
    assert cs.mu_hat(w) == pytest.approx(toy.alpha)


def test_plot_mean_counts_a_tree_once_per_plot(toy, design):
    """Tree A is in two plots, so the plot means (2, 2, 4) average to 8/3, not mu = 2."""
    visits = cs.visit(toy, design, ALL, NONE)
    assert cs.plot_mean(toy, visits).tolist() == [2.0, 2.0, 4.0]


def test_toy_shifted(toy, design):
    """The same navigation error at every pin. Standing at (0.7, -0.4) tree A is
    0.5 m away and its apparent location (0.3, 0.4) is within R of pin 0 only;
    from pin 1 it is 2.73 m away, outside the plot. So C = 1 and t = (2, 0, 4)."""
    eps = np.tile([0.7, -0.4], (3, 1))
    t, w = cs.totals(toy, cs.visit(toy, design, ALL, eps))
    assert t.tolist() == [2.0, 0.0, 4.0]
    assert cs.mu_hat(t) == toy.mu


def test_estimators_by_hand():
    t, w = np.array([1.0, 1.0, 4.0]), np.array([0.5, 0.5, 1.0])
    assert cs.mu_hat(t) == 2.0
    assert cs.v_hat(t) == pytest.approx(1.0)  # sample variance 3, over k = 3
    assert cs.mu_ratio(t, w, alpha=2 / 3) == pytest.approx(2.0)
    # residuals t - 3 w = (-.5, -.5, 1), sum of squares 1.5; (4/9) 1.5 / (6 (2/3)^2) = 0.25
    assert cs.v_ratio(t, w, alpha=2 / 3) == pytest.approx(0.25)


@pytest.mark.parametrize("R", [3.0, 6.0, 12.0])
def test_exact_given_a_common_error(R):
    """Theorem 1, conditional reading. With one navigation error shared by every
    pin, the mean of t over the pins equals (1/N) sum of y_j over the trees whose
    apparent location has a pin within R. Exact, for any error and any radius."""
    rng = np.random.default_rng(1)
    population = cs.synthetic(cs.Gaussian(2.5), rng)
    pins = cKDTree(population.pins)
    for eps in rng.normal(scale=4.0, size=(5, 2)):
        visits = cs.visit(population, cs.Design(R=R), np.arange(population.N), np.tile(eps, (population.N, 1)))
        t, _ = cs.totals(population, visits)
        covered = pins.query_ball_point(population.x - eps, R, return_length=True) > 0
        assert cs.mu_hat(t) == pytest.approx(population.y[covered].sum() / population.N, rel=1e-12)


def test_unbiased_when_the_radius_covers_the_errors():
    """Uniform errors of radius 5 at the pin and at the visit, R = 10.5 > 5 + 5."""
    rng = np.random.default_rng(2)
    error = cs.Uniform(5.0)
    population = cs.synthetic(error, rng)
    design = cs.Design(R=10.5)
    assert cs.coverage(population, design, error, 200, rng).min() == 1.0
    e = cs.enumerate_visits(population, design, error, 200, rng)
    for name in ("t", "t1"):
        assert abs(e[name].mean() - population.mu) < 4 * e[name].std() / np.sqrt(len(e[name]))
    assert abs(e["w"].mean() - 1.0) < 4 * e["w"].std() / np.sqrt(len(e["w"]))


def test_membership_must_follow_the_recorded_offset():
    """With 1.5 m of offset error the estimator stays unbiased when membership
    uses the recorded offset, and is biased up when it uses the true distance."""
    rng = np.random.default_rng(3)
    error = cs.Uniform(3.0)
    population = cs.synthetic(error, rng)
    z = {}
    for recorded in (True, False):
        t = cs.enumerate_visits(population, cs.Design(R=14.0, offset_sd=1.5, recorded=recorded), error, 200, rng)["t"]
        z[recorded] = (t.mean() - population.mu) / (t.std() / np.sqrt(len(t)))
    assert abs(z[True]) < 4
    assert z[False] > 4


def test_one_tree_is_drawn_in_proportion_to_one_over_C(toy):
    """One plot holding both trees: a pin at 10 m with R = 9.5 reaches A (C = 3:
    pins 0, 3 and 10) and B (C = 2: pins 20 and 10). w = 1/3 + 1/2 = 5/6, so A is
    drawn with probability 2/5 and t' is (5/6) 2 or (5/6) 4, with mean t = 8/3."""
    population = cs.Population(pins=np.vstack([toy.pins, [10.0, 0.0]]), x=toy.x, y=toy.y)
    visits = cs.visit(population, cs.Design(R=9.5), np.full(20_000, 3), np.zeros((20_000, 2)))
    assert visits.C[:2].tolist() == [3.0, 2.0]
    t1 = cs.one_tree(population, visits, np.random.default_rng(4))
    assert set(np.round(t1, 12)) == {round(5 / 6 * 2, 12), round(5 / 6 * 4, 12)}
    assert np.mean(np.isclose(t1, 5 / 6 * 2)) == pytest.approx(0.4, abs=0.015)
    assert t1.mean() == pytest.approx(8 / 3, abs=0.03)
