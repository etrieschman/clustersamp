"""The frame and the trees: pins c_i, live trees' true locations x_j and
attributes y_j, and the position-error models that place them."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

EARTH_RADIUS = 6_371_008.8  # m
KENTLAND = Path(__file__).resolve().parents[2] / "data" / "wt_kentland_data.csv"


@dataclass(frozen=True)
class Gaussian:
    """Position error with independent normal coordinates, each with standard deviation ``sd`` [m]."""

    sd: float


@dataclass(frozen=True)
class Uniform:
    """Position error uniform on a disc of radius ``radius`` [m]."""

    radius: float


def draw(error: Gaussian | Uniform, rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` errors, shape (n, 2) [m]."""
    if isinstance(error, Gaussian):
        return rng.normal(scale=error.sd, size=(n, 2))
    r = error.radius * np.sqrt(rng.uniform(size=n))
    angle = rng.uniform(0, 2 * np.pi, size=n)
    return np.column_stack([r * np.cos(angle), r * np.sin(angle)])


@dataclass(frozen=True)
class Population:
    """The frame and the live trees. ``pins`` (N, 2) are the coordinates recorded
    at planting, one per planted tree [m]. ``x`` (N_a, 2) and ``y`` (N_a,) are the
    live trees' true locations [m] and attribute. Dead trees have no row in ``x``;
    their pins stay in ``pins``."""

    pins: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @property
    def N(self) -> int:
        return len(self.pins)

    @property
    def mu(self) -> float:
        """Mean per planted tree, tau / N."""
        return float(self.y.sum() / self.N)

    @property
    def alpha(self) -> float:
        """Survival rate, N_a / N."""
        return len(self.x) / self.N

    @property
    def S2(self) -> float:
        """Variance of the attribute over planted trees, dead trees at zero:
        the variance of one tree drawn from a tagged population."""
        return float((self.y**2).sum() / self.N - self.mu**2)


def kentland(error: Gaussian | Uniform, rng: np.random.Generator, path: Path = KENTLAND) -> Population:
    """The 107 trees measured by phone at Kentland Farm. The recorded coordinates
    are the pins; true locations are the pins displaced by one draw of ``error``.
    Latitude and longitude are projected to meters on a local tangent plane."""
    trees = pl.read_csv(path, columns=["lat", "lon", "biomass_TCO2e"])
    lat, lon = np.deg2rad(trees["lat"].to_numpy()), np.deg2rad(trees["lon"].to_numpy())
    pins = EARTH_RADIUS * np.column_stack([(lon - lon.mean()) * np.cos(lat.mean()), lat - lat.mean()])
    return Population(pins=pins, x=pins + draw(error, rng, len(pins)), y=trees["biomass_TCO2e"].to_numpy())


def synthetic(
    error: Gaussian | Uniform,
    rng: np.random.Generator,
    corners: tuple[tuple[float, float], ...] = ((0, 0), (150, 20), (40, 160), (220, 180), (300, 60)),
    side: tuple[float, float] = (40.0, 90.0),
    area_per_tree: tuple[float, float] = (25.0, 60.0),
    spacing: float = 2.0,
    mean: float = 0.5,
    shape: float = 6.0,
    competition: float = 0.35,
    neighborhood: float = 8.0,
) -> Population:
    """Stands of different planting density in which a tree's attribute falls
    with its number of neighbors, so size is correlated with local density and
    there is a lot of stand edge. One stand per corner, sides drawn from ``side``
    [m], density from ``area_per_tree`` [m^2]; no two trees closer than
    ``spacing``. Attributes are gamma with ``mean`` and ``shape``, times
    exp(-``competition`` x standardized neighbor count within ``neighborhood``).
    Pins are true locations displaced by one draw of ``error``."""
    stands = []
    for corner in corners:
        width, height = rng.uniform(*side, size=2)
        n = int(width * height / rng.uniform(*area_per_tree))
        stands.append(np.asarray(corner) + rng.uniform(size=(n, 2)) * [width, height])
    x = np.vstack(stands)
    keep = np.ones(len(x), dtype=bool)
    for i, j in sorted(cKDTree(x).query_pairs(spacing)):
        if keep[i] and keep[j]:
            keep[j] = False
    x = x[keep]
    neighbors = cKDTree(x).query_ball_point(x, neighborhood, return_length=True) - 1
    y = rng.gamma(shape, mean / shape, size=len(x)) * np.exp(-competition * (neighbors - neighbors.mean()) / neighbors.std())
    return Population(pins=x + draw(error, rng, len(x)), x=x, y=y)


def mortality(
    population: Population,
    rng: np.random.Generator,
    patches: int = 12,
    radius: float = 15.0,
    peak: float = 0.7,
) -> Population:
    """Kill trees in patches. A tree dies with probability ``peak`` at a patch
    center, falling off with distance over ``radius`` [m]. Dead trees leave no
    trace; their pins stay in the frame."""
    centers = rng.uniform(population.x.min(0), population.x.max(0), size=(patches, 2))
    nearest = np.linalg.norm(population.x[:, None] - centers[None], axis=2).min(1)
    alive = rng.uniform(size=len(population.x)) > peak * np.exp(-((nearest / radius) ** 2))
    return replace(population, x=population.x[alive], y=population.y[alive])
