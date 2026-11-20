# clustersamp — working notes

Sampling trees from a frame of noisy GPS pins. **The spec is the note**
(`draft_pin_count.tex` in the Overleaf project, `~/dev/incubator/clustersamp`).
This repo transcribes it: one record per object, one function per equation.

## To do

State on 2026-11-20. The estimator, its proofs and the simulations exist as a first draft. Nothing has been read by anyone outside this project. Items are in the order they block a submission.

Check, before the note goes to anyone:

1. The proofs. Theorem 1 and Propositions 1 to 5 were derived in one sitting and checked only by simulation and by `test_exact_given_a_common_error`. Get a survey statistician to read Section 4 of the note.
2. The closest prior work. Read Bouriaud et al. 2024, Chauvet et al. 2023 and Lavallée and Caron 2001 in full before claiming novelty; so far only their abstracts and metadata have been checked. The bibliography was checked against Crossref or the documents on 2026-11-20; `bib.bib` marks the open points (Gregoire and Valentine's year, two abstracts not opened).
3. The VM0047 v1.1 wording. Section and equation numbers are confirmed (§4.3 items 6 and 7, §8.3.1.1 Eq. 24, §8.5.2 Eq. 29 to 31, §9.2). The note paraphrases; copy exact wording from the PDF where it quotes.

Decide:

4. Venue and form: a note in Canadian Journal of Forest Research (the closest prior work is there) or Survey Methodology. Authors and order.
5. The protocol the note recommends: offsets from one standing point (one error per plot, needs a rangefinder and compass), or a phone reading at each tree (no extra tools, one error per tree). The theorem covers both. The simulations cover only the first.
6. The error values. Gaussian sd 2.5 m and uniform radius 5 m are placeholders. Replace them with measured phone error for the devices and canopy in question; it sets the plot radius and so the burden.
7. Whether the plot-mean baseline stays in the note. It is there to answer "why not average what you find at each pin".

Build:

8. Change over time: simulate two events at the same pins and report the variance of the paired estimator against fresh pins (note, Proposition 5 has no numbers yet).
9. Strata: the estimator with weights 1 / sum_h (k_h / N_h) C_h(u), as a proposition and a `stratified` function. The stratum-by-stratum version double counts trees near a boundary.
10. Sample size: k for a target 90% half-width, from the burden table, as a function.
11. Assumption 1 failing: a position error that grows across the site (canopy gradient). How large is the bias.
12. Kentland over many draws of true locations, not one. The plot-mean bias on Kentland changes sign with the draw.
13. Selection without replacement and a finite-population correction.

Field:

14. A trial on tagged trees: the estimate against the tagged truth, minutes per plot against minutes per tagged tree, and the phone's navigation error.

Write:

15. The `\todo{}` lines in `draft_pin_count.tex`.

## Working rules

- Commits go to `main` and are pushed, here and in the Overleaf repo (Erich, 2026-11-20).
- Use the note's terms. A capitalized class is a record for one of the note's objects, with no behavior. If a name needs explaining, it is wrong.
- Letters where the note has a letter, words where it has a word (table below).
- An option exists only if a question varies it.
- The walkthrough notebook is the documentation.

## Notation: note → code

| note | code | where |
|---|---|---|
| frame: pins c_1..c_N; live trees U with x_j, y_j | `Population(pins, x, y)`; `.N`, `.mu` (τ/N), `.alpha` (N_a/N), `.S2` (variance of one tagged tree, dead = 0) | `population.py` |
| position error distribution G | `Gaussian(sd)`, `Uniform(radius)`; `draw(error, rng, n)` | `population.py` |
| the two populations; mortality | `kentland(error, rng)`, `synthetic(error, rng)`; `mortality(population, rng)` | `population.py` |
| plot radius R; error in the recorded offset; membership by the recorded offset | `Design(R, offset_sd, bearing_sd, recorded)` | `design.py` |
| a visit: standing point c_i + ε, recorded offsets, apparent locations u_ij, plot s_i, pin counts C_j | `visit(population, design, pin, eps, rng) -> Visits(k, visit, tree, C)` | `design.py` |
| k draws I_1..I_k with equal probability | `sample(population, design, error, k, rng)` | `design.py` |
| plot totals t_r, w_r | `totals(population, visits) -> (t, w)` | `design.py` |
| one tree per plot, t'_r = w_r y_J | `one_tree(population, visits, rng)` | `design.py` |
| plot mean ȳ_i (the naive estimator) | `plot_mean(population, visits)` | `design.py` |
| μ̂, v̂; α̂ = μ̂ of w; Δ̂ = μ̂ of t⁽²⁾ − t⁽¹⁾ | `mu_hat(t)`, `v_hat(t)` | `design.py` |
| ratio form μ̂_R and its variance | `mu_ratio(t, w, alpha)`, `v_ratio(t, w, alpha)` | `design.py` |
| coverage probability κ_j | `coverage(population, design, error, draws, rng)` | `metrics.py` |
| expectation and variance of one visit, by enumeration | `enumerate_visits(population, design, error, draws, rng)` | `metrics.py` |
| bias, variance against a tagged tree, measurement burden, interval coverage | `summary(population, design, error, draws, k, reps, rng)`; `survival(...)`; `ESTIMATORS` | `metrics.py` |
| tables | `tex(table, path, digits, groups, signed)` | `metrics.py` |

Conventions: capitalized = record; lowercase = function; errors are in meters; every random draw takes an explicit `rng`.

## Design decisions (settled; do not relitigate)

| topic | decision |
|---|---|
| the weight | A tree in a plot is weighted by 1 / C_j, the number of pins within R of pin + recorded offset. No GPS error model enters the estimator. The model inclusion probabilities of the 2024 code are gone. |
| membership | A tree is in the plot iff its recorded offset is shorter than R, the same offset the pin count uses. `Design(recorded=False)` exists only to show the bias of deciding membership by true distance. |
| the 2024 design | Removed: pins drawn with probability proportional to expected plot size, scaled plot means, inclusion probabilities by quadrature. Its expectation is that of the plot mean, which stays as a baseline in `plot_mean`. The last commit with the 2024 code is `c6c47a8`. |
| evidence | Bias and variance come from enumeration (every pin, many error draws), not from repeated samples. Repeated samples are drawn from the enumerated visits and are used for the ratio forms' bias, the variance estimators and interval coverage. No bootstrap. |
| measurement burden | variance of one visit / variance of one tagged tree, times biomass measurements per visit. It is the number of biomass measurements per measurement of a tagged simple random sample at equal precision. The ratio forms use the linearized variance. |
| ratio form | Uses the true survival rate in simulation. In the field it needs a known survival rate; without one it estimates the mean per live tree. |
| errors | Pin error and navigation error are drawn from the same distribution in the base runs; the scenario `navigation error from another distribution` breaks that. Gaussian sd 2.5 m, uniform radius 5 m: placeholders. |
| projection | Kentland latitude and longitude go to meters on a local tangent plane. The site is 300 m across; no pyproj. |
| running many configurations | One script, `notebooks/results.py`: a grid in the settings cell, rows cached in parquet by key, tables and figures written and copied to `CLUSTERSAMP_OUT`. No workflow manager. The whole grid takes under a minute. |
| parameter objects | Frozen dataclasses, arrays inside them. Plain functions for computation. |

## Layout

```
src/clustersamp/
  population.py  Gaussian, Uniform, draw; Population; kentland, synthetic, mortality
  design.py      Design; Visits; visit, sample; totals, one_tree, plot_mean; mu_hat, v_hat, mu_ratio, v_ratio
  metrics.py     ESTIMATORS; enumerate_visits; coverage; summary; survival; tex
notebooks/walkthrough.py   jupytext cells; one visit by hand, then a sample; the documentation
notebooks/results.py       the tables and figures of the note; rows cached in notebooks/tables/rows.parquet
tests/                     conftest (three pins, two trees); test_design, test_population
data/wt_kentland_data.csv  107 trees measured by phone
```
