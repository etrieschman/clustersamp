"""The visit and the estimators: plots around the standing point, apparent
locations, pin counts, and the plot totals t_r and w_r."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .population import Gaussian, Population, Uniform, draw


@dataclass(frozen=True)
class Design:
    """How a plot is measured. ``R`` is the plot radius [m]. ``offset_sd`` is the
    standard deviation of additive error in each coordinate of a recorded offset
    [m]; ``bearing_sd`` that of the recorded bearing [degrees]. ``recorded`` says
    plot membership follows the recorded offset, the rule the estimator needs;
    False decides membership by true distance while the pin count still uses the
    recorded offset, which biases the estimator when offsets carry error."""

    R: float
    offset_sd: float = 0.0
    bearing_sd: float = 0.0
    recorded: bool = True


@dataclass(frozen=True)
class Visits:
    """What ``k`` visits recorded, one entry per (visit, tree in its plot):
    the visit's index, the tree's index into the population, and the tree's pin
    count C_j, the number of pins within R of pin + recorded offset."""

    k: int
    visit: np.ndarray
    tree: np.ndarray
    C: np.ndarray


def visit(
    population: Population,
    design: Design,
    pin: np.ndarray,
    eps: np.ndarray,
    rng: np.random.Generator | None = None,
) -> Visits:
    """Visit pins ``pin`` (k,) with navigation errors ``eps`` (k, 2): the crew
    stands at pin + eps and records each nearby tree's offset. ``rng`` draws the
    offset errors, if the design has any."""
    pin = np.asarray(pin)
    stand = population.pins[pin] + eps
    reach = design.R * (1 + 5 * np.deg2rad(design.bearing_sd)) + 5 * design.offset_sd
    near = cKDTree(stand).sparse_distance_matrix(cKDTree(population.x), reach, output_type="ndarray")
    v, j = near["i"], near["j"]
    offset = population.x[j] - stand[v]
    recorded = offset
    if design.bearing_sd > 0:
        turn = np.deg2rad(rng.normal(scale=design.bearing_sd, size=len(v)))
        recorded = np.column_stack(
            [
                np.cos(turn) * offset[:, 0] - np.sin(turn) * offset[:, 1],
                np.sin(turn) * offset[:, 0] + np.cos(turn) * offset[:, 1],
            ]
        )
    if design.offset_sd > 0:
        recorded = recorded + rng.normal(scale=design.offset_sd, size=offset.shape)
    inside = np.linalg.norm(recorded if design.recorded else offset, axis=1) < design.R
    v, j, recorded = v[inside], j[inside], recorded[inside]
    apparent = population.pins[pin[v]] + recorded
    C = cKDTree(population.pins).query_ball_point(apparent, design.R, return_length=True, workers=-1)
    order = np.lexsort((j, v))
    # under the recorded rule the visited pin is always counted, so C >= 1
    return Visits(k=len(pin), visit=v[order], tree=j[order], C=np.maximum(C[order], 1).astype(float))


def sample(
    population: Population,
    design: Design,
    error: Gaussian | Uniform,
    k: int,
    rng: np.random.Generator,
) -> Visits:
    """Draw ``k`` pins with replacement and equal probability and visit each
    with a navigation error from ``error``."""
    return visit(population, design, rng.integers(population.N, size=k), draw(error, rng, k), rng)


def totals(population: Population, visits: Visits) -> tuple[np.ndarray, np.ndarray]:
    """The plot totals (t, w), each (k,): t_r = sum of y_j / C_j and
    w_r = sum of 1 / C_j over the trees in visit r's plot."""
    t = np.bincount(visits.visit, weights=population.y[visits.tree] / visits.C, minlength=visits.k)
    w = np.bincount(visits.visit, weights=1 / visits.C, minlength=visits.k)
    return t, w


def one_tree(population: Population, visits: Visits, rng: np.random.Generator) -> np.ndarray:
    """t'_r = w_r y_J, with one tree J drawn from each plot with probability
    (1 / C_j) / w_r; zero for an empty plot. Biomass is measured on J only."""
    _, w = totals(population, visits)
    t = np.zeros(visits.k)
    if len(visits.visit):
        key = -np.log(visits.C) + rng.gumbel(size=len(visits.C))  # the largest key wins with probability ~ 1/C
        order = np.lexsort((key, visits.visit))
        last = np.r_[visits.visit[order][1:] != visits.visit[order][:-1], True]
        chosen = order[last]
        t[visits.visit[chosen]] = w[visits.visit[chosen]] * population.y[visits.tree[chosen]]
    return t


def plot_mean(population: Population, visits: Visits) -> np.ndarray:
    """The mean of y over each plot, zero for an empty plot. Averaging these is
    the naive estimator: it weights a tree by 1 / (trees in the plot), not by
    1 / (plots the tree is in)."""
    n = np.bincount(visits.visit, minlength=visits.k)
    return np.bincount(visits.visit, weights=population.y[visits.tree], minlength=visits.k) / np.maximum(n, 1)


def mu_hat(t: np.ndarray) -> np.ndarray:
    """The pin-count estimator of the mean per planted tree: the mean of t over
    the last axis. With w in place of t it estimates the survival rate; with
    t at event 2 less t at event 1 at the same pins, the change."""
    return t.mean(-1)


def v_hat(t: np.ndarray) -> np.ndarray:
    """Unbiased estimator of the variance of ``mu_hat``."""
    return t.var(-1, ddof=1) / t.shape[-1]


def mu_ratio(t: np.ndarray, w: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """The ratio form, alpha x sum(t) / sum(w), for a known survival rate alpha."""
    return alpha * t.sum(-1) / w.sum(-1)


def v_ratio(t: np.ndarray, w: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Linearized estimator of the variance of ``mu_ratio``."""
    k = t.shape[-1]
    residual = t - (t.sum(-1, keepdims=True) / w.sum(-1, keepdims=True)) * w
    return alpha**2 * (residual**2).sum(-1) / (k * (k - 1) * w.mean(-1) ** 2)
