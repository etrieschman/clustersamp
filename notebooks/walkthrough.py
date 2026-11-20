# %% [markdown]
# # Walkthrough: one visit by hand, then a sample
#
# Run cell by cell, or as a script: `uv run python notebooks/walkthrough.py`.
#
# A project planted three trees and recorded a pin for each, at 0, 3 and 20 m
# along a line. Tree A (2 t) stands at 1 m and tree B (4 t) at 19 m. The third
# tree died. The total is 6 t, the mean per planted tree is 2 t, and survival
# is 2/3.

# %%
import numpy as np

import clustersamp as cs

toy = cs.Population(
    pins=np.array([[0.0, 0.0], [3.0, 0.0], [20.0, 0.0]]),
    x=np.array([[1.0, 0.0], [19.0, 0.0]]),
    y=np.array([2.0, 4.0]),
)
design = cs.Design(R=2.5)
print(f"mu = {toy.mu}, alpha = {toy.alpha:.3f}")

# %% [markdown]
# ## A visit with no navigation error
#
# Sent to pin 0, the crew stands at 0 m. Tree A is 1 m away, inside the 2.5 m
# plot. Its offset is +1 m, so its apparent location is pin 0 + 1 m = 1 m.
# Two pins lie within 2.5 m of 1 m: pin 0 and pin 1. The crew would have
# measured tree A from either, so its pin count is 2 and it enters at half
# weight: t = 2 / 2 = 1.
#
# From pin 1 the crew finds tree A again, also with count 2. From pin 2 they
# find tree B, which no other pin reaches: count 1, t = 4.

# %%
visits = cs.visit(toy, design, pin=np.arange(3), eps=np.zeros((3, 2)))
t, w = cs.totals(toy, visits)
print("visit:", visits.visit, " tree:", visits.tree, " pin count:", visits.C)
print("t =", t, " mean", cs.mu_hat(t), "= mu")
print("w =", w, " mean", round(cs.mu_hat(w), 3), "= alpha")
print("plot means:", cs.plot_mean(toy, visits), " mean", round(cs.plot_mean(toy, visits).mean(), 3), "(tree A counted twice)")

# %% [markdown]
# ## The same visits with a navigation error
#
# The phone is off by (0.7, -0.4) m, so the crew stands there when it reads
# pin 0. Tree A is now 0.5 m away with offset (0.3, 0.4). Its apparent
# location, pin 0 + offset, is within 2.5 m of pin 0 only. That matches what
# would happen: sent to pin 1 with the same error, the crew would stand 2.73 m
# from tree A and miss it. The count is 1 and t = 2. The mean over the three
# pins is still 2.

# %%
eps = np.tile([0.7, -0.4], (3, 1))
t, w = cs.totals(toy, cs.visit(toy, design, np.arange(3), eps))
print("t =", t, " mean", cs.mu_hat(t))

# %% [markdown]
# ## A sample
#
# The synthetic stand: tree size falls with local density. Pins and visits
# both carry uniform error of radius 5 m, and R = 10 m covers their sum.

# %%
rng = np.random.default_rng(0)
error = cs.Uniform(5.0)
population = cs.synthetic(error, rng)
design = cs.Design(R=10.0)
visits = cs.sample(population, design, error, k=45, rng=rng)
t, w = cs.totals(population, visits)
t1 = cs.one_tree(population, visits, rng)
print(f"N = {population.N}, mu = {population.mu:.4f}; {len(visits.tree)} trees located at 45 pins")
for name, estimate, variance in (
    ("pin count", cs.mu_hat(t), cs.v_hat(t)),
    ("ratio form", cs.mu_ratio(t, w), cs.v_ratio(t, w)),
    ("one tree per plot", cs.mu_hat(t1), cs.v_hat(t1)),
    ("ratio form, one tree", cs.mu_ratio(t1, w), cs.v_ratio(t1, w)),
):
    print(f"{name:22s} {estimate:.4f}  (s.e. {np.sqrt(variance):.4f})")

# %% [markdown]
# ## What the estimators do on average
#
# `summary` visits every pin many times. `bias` is in percent of mu.
# `burden` is the number of biomass measurements the design needs for each
# measurement of a tagged simple random sample at the same precision.

# %%
import polars as pl

rows = pl.DataFrame(cs.summary(population, design, error, draws=500, k=45, reps=5000, rng=rng))
print(rows.select("estimator", "bias", "bias_se", "variance", "located", "measured", "burden", "ci"))
