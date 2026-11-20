"""What the estimators do on a population: expectation and variance by
enumeration, the coverage probabilities, the measurement burden against a
tagged sample, and the variance estimators under repeated sampling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.spatial import cKDTree

from .design import Design, mu_hat, mu_ratio, one_tree, plot_mean, totals, v_hat, v_ratio, visit
from .population import Gaussian, Population, Uniform, draw

ESTIMATORS = ("tagged sample", "plot mean", "pin count", "pin count, ratio", "pin count, one tree", "pin count, ratio, one tree")


def enumerate_visits(
    population: Population,
    design: Design,
    error: Gaussian | Uniform,
    draws: int,
    rng: np.random.Generator,
    block: int = 200_000,
) -> dict[str, np.ndarray]:
    """Visit every pin ``draws`` times, each with its own navigation error.
    Returns flat arrays of length N x draws: the plot totals ``t`` and ``w``,
    the one-tree total ``t1``, the plot size ``n`` and the plot mean ``ybar``.
    A sample of k pins averages k of these, so the mean of an array is the
    expectation of one visit and its variance over k is the variance of the
    sample mean."""
    pin = np.repeat(np.arange(population.N), draws)
    out = {name: np.zeros(len(pin)) for name in ("t", "w", "t1", "n", "ybar")}
    for start in range(0, len(pin), block):
        rows = slice(start, start + block)
        visits = visit(population, design, pin[rows], draw(error, rng, len(pin[rows])), rng)
        out["t"][rows], out["w"][rows] = totals(population, visits)
        out["t1"][rows] = one_tree(population, visits, rng)
        out["n"][rows] = np.bincount(visits.visit, minlength=visits.k)
        out["ybar"][rows] = plot_mean(population, visits)
    return out


def coverage(
    population: Population,
    design: Design,
    error: Gaussian | Uniform,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """kappa_j for each live tree: the probability that some pin lies within R
    of the tree's apparent location, x_j less the navigation error plus the
    offset error."""
    pins = cKDTree(population.pins)
    kappa = np.zeros(len(population.x))
    for j, x in enumerate(population.x):
        apparent = x - draw(error, rng, draws) + rng.normal(scale=design.offset_sd, size=(draws, 2))
        kappa[j] = (pins.query_ball_point(apparent, design.R, return_length=True) > 0).mean()
    return kappa


def summary(
    population: Population,
    design: Design,
    error: Gaussian | Uniform,
    draws: int,
    k: int,
    reps: int,
    rng: np.random.Generator,
    level: float = 0.9,
) -> list[dict]:
    """One row per estimator in ``ESTIMATORS``. The tagged sample is the
    benchmark: k trees drawn with replacement from a tagged population, a tree
    that died counting as zero.

    From the enumeration: ``bias`` [% of mu] with its Monte Carlo standard error
    ``bias_se``; ``variance``, the variance of one visit over the variance of one
    tagged tree (the ratio forms use the linearized variance); ``located`` and
    ``measured``, the trees located and measured for biomass per visit; and
    ``burden`` = variance x measured, the biomass measurements this design needs
    per measurement of a tagged simple random sample at the same precision.

    From ``reps`` samples of ``k`` visits drawn from the enumerated ones: ``cv``,
    the standard deviation of the estimates [% of mu]; ``v_ratio``, the mean
    variance estimate over the variance of the estimates; ``ci``, how often the
    ``level`` t-interval covers mu [%]. The ratio forms take their bias from
    these samples, since it depends on k, and use the true survival rate."""
    mu, alpha, S2 = population.mu, population.alpha, population.S2
    e = enumerate_visits(population, design, error, draws, rng)
    kappa = coverage(population, design, error, draws, rng)
    idx = rng.integers(len(e["t"]), size=(reps, k))
    t, w, t1, ybar = e["t"][idx], e["w"][idx], e["t1"][idx], e["ybar"][idx]
    located = float(e["n"].mean())
    unit = np.r_[population.y, np.zeros(population.N - len(population.y))]  # one value per planted tree
    tagged = unit[rng.integers(population.N, size=(reps, k))]
    forms = {  # one visit's value, the estimates, their variance estimates, trees located and measured per visit
        "tagged sample": (unit, mu_hat(tagged), v_hat(tagged), 0.0, 1.0),
        "plot mean": (e["ybar"], mu_hat(ybar), v_hat(ybar), located, located),
        "pin count": (e["t"], mu_hat(t), v_hat(t), located, located),
        "pin count, ratio": (e["t"] - mu / alpha * e["w"], mu_ratio(t, w, alpha), v_ratio(t, w, alpha), located, located),
        "pin count, one tree": (e["t1"], mu_hat(t1), v_hat(t1), located, float((e["n"] > 0).mean())),
        "pin count, ratio, one tree": (e["t1"] - mu / alpha * e["w"], mu_ratio(t1, w, alpha), v_ratio(t1, w, alpha), located, float((e["n"] > 0).mean())),
    }
    shared = {
        "N": population.N,
        "mu": mu,
        "alpha": alpha,
        "trees_per_plot": located,
        "empty": 100 * float((e["n"] == 0).mean()),
        "coverage_bias": 100 * (float((kappa * population.y).sum() / population.y.sum()) - 1),
        "kappa_min": float(kappa.min()),
    }
    quantile = stats.t.ppf(1 - (1 - level) / 2, k - 1)
    rows = []
    for name, (one_visit, estimates, variances, n_located, measured) in forms.items():
        is_ratio = "ratio" in name
        expected = estimates.mean() if is_ratio else one_visit.mean()
        se = estimates.std(ddof=1) / np.sqrt(reps) if is_ratio else one_visit.std() / np.sqrt(len(one_visit))
        if name == "tagged sample":
            expected, se = mu, 0.0  # unbiased by construction
        variance = float(one_visit.var() / S2)
        rows.append(
            {
                "estimator": name,
                "bias": 100 * (float(expected) / mu - 1),
                "bias_se": 100 * float(se) / mu,
                "variance": variance,
                "located": n_located,
                "measured": measured,
                "burden": variance * measured,
                "cv": 100 * float(estimates.std(ddof=1)) / mu,
                "v_ratio": float(variances.mean() / estimates.var(ddof=1)),
                "ci": 100 * float((np.abs(estimates - mu) < quantile * np.sqrt(variances)).mean()),
                **shared,
            }
        )
    return rows


def survival(
    population: Population,
    design: Design,
    error: Gaussian | Uniform,
    draws: int,
    k: int,
    reps: int,
    rng: np.random.Generator,
) -> dict:
    """The pin-count row of ``summary`` with y = 1 for every live tree: the
    estimator of the survival rate."""
    ones = replace(population, y=np.ones(len(population.x)))
    return next(row for row in summary(ones, design, error, draws, k, reps, rng) if row["estimator"] == "pin count")


def tex(
    table: pl.DataFrame,
    path: str | Path,
    digits: int = 2,
    groups: dict[str, list[str]] | None = None,
    signed: tuple[str, ...] = (),
) -> None:
    """Write ``table`` as a booktabs ``tabular`` for ``\\input`` inside a table
    environment the document controls. Column names are the headers, written as
    given; floats get ``digits`` decimals, with a sign in the ``signed``
    columns; nulls print as --; a repeated leading label is left blank.
    ``groups`` maps a header that spans several adjacent columns to their names."""

    def cell(v, name: str) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "--"
        if isinstance(v, float):
            v = round(v, digits) or 0.0  # no -0
            return f"${v:+,.{digits}f}$" if name in signed else f"{v:,.{digits}f}"
        return f"{v:,d}" if isinstance(v, int) else str(v)

    align = "".join("r" if dt.is_numeric() else "l" for dt in table.dtypes)
    header = [" & ".join(table.columns) + r" \\"]
    if groups:
        cols, cells, rules = list(table.columns), [""] * table.width, []
        for name, members in groups.items():
            idx = [cols.index(c) for c in members]
            a, b = min(idx), max(idx)
            if idx != list(range(a, b + 1)):
                raise ValueError(f"group {name!r} must span adjacent columns")
            cells[a] = rf"\multicolumn{{{b - a + 1}}}{{c}}{{{name}}}"
            cells[a + 1 : b + 1] = [None] * (b - a)
            rules.append(rf"\cmidrule(lr){{{a + 1}-{b + 1}}}")
        header = [" & ".join(c for c in cells if c is not None) + r" \\", " ".join(rules), *header]
    body, previous = [], None
    for row in table.iter_rows():
        cells = [cell(v, name) for v, name in zip(row, table.columns)]
        shown = list(cells)
        for i, dt in enumerate(table.dtypes):  # blank the leading labels a row shares with the one above
            if dt.is_numeric() or previous is None or cells[i] != previous[i]:
                break
            shown[i] = ""
        body.append(" & ".join(shown) + r" \\")
        previous = cells
    lines = [rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule", *header, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
