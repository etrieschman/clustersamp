"""A hand-sized population for the tests: three pins and two live trees on a
line, numbers you can check on paper. No data file."""

from __future__ import annotations

import numpy as np
import pytest

from clustersamp import Design, Population


@pytest.fixture
def toy() -> Population:
    """Pins at 0, 3 and 20 m. Tree A (y = 2) at 1 m, tree B (y = 4) at 19 m; the
    third planted tree is dead. tau = 6, mu = 2, alpha = 2/3.

    With R = 2.5 and no error, tree A is in the plots of pins 0 and 1 (C = 2)
    and tree B in the plot of pin 2 (C = 1), so t = (1, 1, 4) and w = (.5, .5, 1)."""
    return Population(
        pins=np.array([[0.0, 0.0], [3.0, 0.0], [20.0, 0.0]]),
        x=np.array([[1.0, 0.0], [19.0, 0.0]]),
        y=np.array([2.0, 4.0]),
    )


@pytest.fixture
def design() -> Design:
    return Design(R=2.5)
