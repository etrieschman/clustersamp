# clustersamp

Sampling trees from a frame of noisy GPS pins. A project plants N trees and
records one phone pin per tree. GPS error is as large as tree spacing, so a
pin does not identify its tree. The pin-count estimator samples pins, measures
a fixed-radius plot at each, and weights every tree by one over the number of
pins within the plot radius of pin + the tree's recorded offset.

The method is specified in the note (`draft_pin_count.tex` in the Overleaf
project, `~/dev/incubator/clustersamp`), and this package transcribes it.
`CLAUDE.md` holds the notation table, the design decisions and the to-do list.

Three modules, each importing only the ones before it: `population` (the
frame, the trees, the error models), `design` (the visit and the estimators),
`metrics` (bias, variance, measurement burden, tables).

```python
import numpy as np
import clustersamp as cs

rng = np.random.default_rng(0)
population = cs.synthetic(cs.Uniform(5.0), rng)                  # pins, live trees, biomass
design = cs.Design(R=10.0)                                       # the plot radius
visits = cs.sample(population, design, cs.Uniform(5.0), 45, rng) # 45 pins, each visited with a navigation error
t, w = cs.totals(population, visits)                             # plot totals of y / C and of 1 / C
cs.mu_hat(t), cs.v_hat(t)                                        # mean per planted tree and its variance
cs.mu_hat(w)                                                     # survival rate
cs.mu_hat(cs.one_tree(population, visits, rng))                  # biomass measured on one tree per plot
```

## Run

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run python notebooks/results.py
```

`notebooks/results.py` runs the grid in its settings cell and caches each row
in `notebooks/tables/rows.parquet`; a row already there is read, not
recomputed. Tables go to `notebooks/tables` and figures to `notebooks/figures`.
With `CLUSTERSAMP_OUT` set (in the environment or in `.env` at the repo root),
the `.tex` tables and the figures are also copied there.

`notebooks/walkthrough.py` works one visit by hand.
