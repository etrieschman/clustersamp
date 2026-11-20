from __future__ import annotations

import numpy as np
import pytest

import clustersamp as cs


def test_toy_targets(toy):
    assert (toy.N, toy.mu, toy.alpha) == (3, 2.0, pytest.approx(2 / 3))
    assert toy.S2 == pytest.approx((4 + 16) / 3 - 4)  # values (2, 4, 0) over the three planted trees


def test_errors():
    rng = np.random.default_rng(0)
    assert np.linalg.norm(cs.draw(cs.Uniform(5.0), rng, 10_000), axis=1).max() <= 5.0
    assert cs.draw(cs.Gaussian(2.5), rng, 50_000).std(0) == pytest.approx([2.5, 2.5], rel=0.02)


def test_kentland():
    population = cs.kentland(cs.Uniform(5.0), np.random.default_rng(0))
    assert population.N == len(population.x) == 107
    assert 150 < np.ptp(population.pins, axis=0).max() < 500  # the site is a few hundred meters across
    assert np.linalg.norm(population.x - population.pins, axis=1).max() <= 5.0


def test_mortality_keeps_the_pins():
    rng = np.random.default_rng(0)
    population = cs.synthetic(cs.Uniform(5.0), rng)
    thinned = cs.mortality(population, rng)
    assert thinned.N == population.N
    assert 0.8 < thinned.alpha < 1.0
