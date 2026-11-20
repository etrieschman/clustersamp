# %% [markdown]
# # Results: the pin-count estimator on two populations
#
# Run cell by cell, or as a script: `uv run python notebooks/results.py`.
# The settings cell says what is run. To drop a population, an error model, a
# radius or a scenario, take it out of its list there.
#
# | table | question | rows |
# |---|---|---|
# | bias | Is the estimator unbiased, and what does a small radius cost? | population x error x radius |
# | burden | What does not tagging cost in measurements? | population x error x radius |
# | scenarios | Does it hold when the plot is measured with error, and under mortality? | scenario x error |
# | sampling | Do the variance estimators and intervals work? | population x error x estimator |
#
# Rows are cached in `notebooks/tables/rows.parquet`: a row already there is
# read, not recomputed. Set `REFRESH = True` after the code changes. Tables go
# to `notebooks/tables` as .csv and booktabs .tex, figures to
# `notebooks/figures`; with `CLUSTERSAMP_OUT` set (in the environment or in
# `.env` at the repo root) the .tex tables and the figures are also copied there.

# %%
import os
import shutil
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import clustersamp as cs

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(1000)

# %%
# Settings.
POPULATIONS = {"Kentland": cs.kentland, "synthetic": cs.synthetic}
ERRORS = {"Gaussian": cs.Gaussian(2.5), "uniform": cs.Uniform(5.0)}  # the pin error and the navigation error [m]
RADII = (6.0, 10.0, 15.0, 20.0)  # 6 m does not cover the errors
BASE_R = 10.0  # the radius of the scenario and sampling tables
SCENARIOS = {  # changes to the base design; run on the synthetic population at BASE_R
    "offset error 0.5 m": {"offset_sd": 0.5},
    "offset error 0.5 m, membership by true distance": {"offset_sd": 0.5, "recorded": False},
    "offset error 1.5 m": {"offset_sd": 1.5},
    "offset error 1.5 m, membership by true distance": {"offset_sd": 1.5, "recorded": False},
    "bearing error 2 degrees": {"bearing_sd": 2.0},
    "bearing error 5 degrees": {"bearing_sd": 5.0},
}
OTHER = {"Gaussian": cs.Uniform(3.0), "uniform": cs.Gaussian(1.5)}  # a navigation error unlike the pin error
DRAWS = 1000  # navigation errors per pin in the enumeration
K = 45  # pins per sample in the repeated-sampling columns
REPS = 20_000  # samples
SEED = 20261120
REFRESH = False  # True recomputes every row

here = Path(__file__).parent if "__file__" in globals() else Path(".")
tables, figures = here / "tables", here / "figures"
tables.mkdir(exist_ok=True)
figures.mkdir(exist_ok=True)
dotenv = here.parent / ".env"
if dotenv.exists():  # KEY=VALUE lines; the environment wins
    for line in dotenv.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
OUT = Path(os.environ["CLUSTERSAMP_OUT"]).expanduser() if os.environ.get("CLUSTERSAMP_OUT") else None


def publish(path: Path) -> None:
    """Copy a table or figure to CLUSTERSAMP_OUT, if set."""
    if OUT is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, OUT / path.name)


def save(table: pl.DataFrame, name: str, **kwargs) -> None:
    """Print a table and write it as .csv and .tex."""
    print(f"\n{name}")
    print(table)
    table.write_csv(tables / f"{name}.csv")
    cs.tex(table, tables / f"{name}.tex", **kwargs)
    publish(tables / f"{name}.tex")


# %%
# The populations: one draw of each per error model, the same across radii and scenarios.
populations = {
    (p, e): make(ERRORS[e], np.random.default_rng([SEED, i, j]))
    for (i, (p, make)), (j, e) in product(enumerate(POPULATIONS.items()), enumerate(ERRORS))
}
for (p, e), population in populations.items():
    print(f"{p}, {e} error: N = {population.N}, mu = {population.mu:.3f}")

# %%
# The runs: every population, error and radius at the base design, then the
# scenarios on the synthetic population at BASE_R.
runs = [(p, e, R, "base") for p, e, R in product(POPULATIONS, ERRORS, RADII)]
extra = [*SCENARIOS, "navigation error from another distribution", "patchy mortality", "patchy mortality: survival"]
runs += [("synthetic", e, BASE_R, s) for e, s in product(ERRORS, extra) if "synthetic" in POPULATIONS]


def run(p: str, e: str, R: float, scenario: str, rng: np.random.Generator) -> list[dict]:
    population, error, design = populations[p, e], ERRORS[e], cs.Design(R=R)
    if scenario in SCENARIOS:
        design = replace(design, **SCENARIOS[scenario])
    if scenario.startswith("navigation"):
        error = OTHER[e]
    if scenario.startswith("patchy"):
        population = cs.mortality(population, np.random.default_rng([SEED, 99]))
    if scenario.endswith("survival"):
        return [cs.survival(population, design, error, DRAWS, K, REPS, rng)]
    return cs.summary(population, design, error, DRAWS, K, REPS, rng)


KEY = ["population", "error", "R", "scenario", "draws", "k", "reps", "seed"]
path = tables / "rows.parquet"
cached = pl.read_parquet(path) if path.exists() and not REFRESH else None
done = set(cached.select(KEY).iter_rows()) if cached is not None else set()
new = []
t0 = time.time()
for i, (p, e, R, scenario) in enumerate(runs):
    key = (p, e, R, scenario, DRAWS, K, REPS, SEED)
    if key in done:
        continue
    rows = run(p, e, R, scenario, np.random.default_rng([SEED, 1, i]))
    new += [{**dict(zip(KEY, key)), **row} for row in rows]
    print(f"{time.time() - t0:5.0f} s\t{p}\t{e}\tR = {R:g}\t{scenario}", flush=True)
if new:
    cached = pl.concat([x for x in (cached, pl.DataFrame(new)) if x is not None], how="diagonal_relaxed")
    cached.write_parquet(path)
rows = cached.join(pl.DataFrame([dict(zip(KEY, (*r, DRAWS, K, REPS, SEED))) for r in runs], schema=cached.select(KEY).schema), on=KEY)
print(f"rows: {rows.height} ({len(new)} computed, {rows.height - len(new)} read)")
base = rows.filter(pl.col("scenario") == "base")
print(f"largest Monte Carlo standard error of a bias [%]: {rows['bias_se'].max():.2f}")

# %%
# Table: bias. Relative bias of each estimator by enumeration, and the bias
# Theorem 1 predicts from the coverage probabilities.
ID = ["population", "error", "R"]
bias = (
    base.pivot(on="estimator", index=ID, values="bias")
    .join(base.group_by(ID).agg(pl.col("trees_per_plot").first(), pl.col("coverage_bias").first()), on=ID)
    .sort(ID)
    .select(
        pl.col("population"),
        pl.col("error").alias("GPS error"),
        pl.col("R").cast(pl.Int64).alias("$R$ [m]"),
        pl.col("trees_per_plot").alias("trees per plot"),
        pl.col("plot mean"),
        pl.col("pin count"),
        pl.col("pin count, ratio").alias("ratio form"),
        pl.col("coverage_bias").alias("predicted"),
    )
)
save(bias, "table_bias", signed=("plot mean", "pin count", "ratio form", "predicted"), groups={"relative bias [\\%]": ["plot mean", "pin count", "ratio form", "predicted"]})

# %%
# Table: measurement burden. What a tagged simple random sample would need one
# of, this design needs `visits` pin visits and `measurements` biomass
# measurements for the same precision, and it locates `trees located per
# visit` trees at each visit. Ratio forms, radii that cover the errors.
covered = base.filter(pl.col("R") >= BASE_R)
every = covered.filter(pl.col("estimator") == "pin count, ratio").select(*ID, "located", pl.col("variance").alias("visits"), pl.col("burden").alias("measurements"))
one = covered.filter(pl.col("estimator") == "pin count, ratio, one tree").select(*ID, pl.col("burden").alias("measurements "))
burden = (
    every.join(one, on=ID)
    .sort(ID)
    .select(pl.col("population"), pl.col("error").alias("GPS error"), pl.col("R").cast(pl.Int64).alias("$R$ [m]"), pl.col("located").alias("trees located per visit"), "visits", "measurements", "measurements ")
)
save(burden, "table_burden", groups={"every tree measured": ["visits", "measurements"], "one tree measured": ["measurements "]})

# %%
# Table: scenarios. Relative bias on the synthetic population at BASE_R when the
# plot is measured with error, the navigation error differs from the pin error,
# and trees have died.
order = ["base", *extra]
scenarios = (
    rows.filter((pl.col("population") == "synthetic") & (pl.col("R") == BASE_R) & pl.col("estimator").is_in(["plot mean", "pin count", "pin count, ratio"]))
    .filter(~(pl.col("scenario").str.ends_with("survival") & (pl.col("estimator") != "pin count")))
    .filter(~(pl.col("scenario").str.starts_with("patchy") & (pl.col("estimator") == "plot mean")))  # the plot mean estimates the mean per live tree
    .with_columns((pl.col("error") + ": " + pl.col("estimator").str.replace("pin count, ratio", "ratio form")).alias("column"))
    .pivot(on="column", index="scenario", values="bias")
    .with_columns(pl.col("scenario").replace_strict({s: i for i, s in enumerate(order)}, return_dtype=pl.Int64).alias("order"))
    .sort("order")
    .drop("order")
)
columns = [f"{e}: {name}" for e in ERRORS for name in ("plot mean", "pin count", "ratio form")]
scenarios = scenarios.select("scenario", *columns).rename({c: c.split(": ")[1] + " " * i for i, e in enumerate(ERRORS) for c in columns if c.startswith(e)})
save(scenarios, "table_scenarios", signed=tuple(scenarios.columns[1:]), groups={f"{e} error": scenarios.columns[1 + 3 * i : 4 + 3 * i] for i, e in enumerate(ERRORS)})

# %%
# Table: sampling. REPS samples of K pins at BASE_R: the spread of the
# estimates, the variance estimator against it, and interval coverage.
sampling = (
    base.filter(pl.col("R") == BASE_R)
    .sort(ID)
    .select(
        pl.col("population"),
        pl.col("error").alias("GPS error"),
        pl.col("estimator").str.replace("pin count, ratio", "ratio form"),
        pl.col("cv").alias("CV [\\%]"),
        pl.col("v_ratio").alias("$\\hat v$ / variance"),
        pl.col("ci").alias("90\\% interval covers [\\%]"),
    )
)
save(sampling, "table_sampling")

# %%
# Figure: the populations, uniform error.
fig, ax = plt.subplots(ncols=len(POPULATIONS), figsize=(5 * len(POPULATIONS), 4.5), squeeze=False)
for a, p in zip(ax[0], POPULATIONS):
    population = populations[p, "uniform"]
    a.scatter(*population.x.T, s=60 * population.y, alpha=0.5, label="tree (area: biomass)")
    a.scatter(*population.pins.T, s=5, color="C3", label="pin")
    a.set_aspect("equal")
    a.set_title(p, fontsize=10)
    a.set_xlabel("m")
ax[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(figures / "fig_populations.png", dpi=200)
publish(figures / "fig_populations.png")

# %%
# Figure: bias by radius.
fig, ax = plt.subplots(nrows=len(POPULATIONS), ncols=len(ERRORS), sharex=True, figsize=(4 * len(ERRORS), 2.8 * len(POPULATIONS)), squeeze=False)
for (i, p), (j, e) in product(enumerate(POPULATIONS), enumerate(ERRORS)):
    for c, (name, label) in enumerate({"plot mean": "plot mean", "pin count": "pin count", "pin count, ratio": "ratio form"}.items()):
        d = base.filter((pl.col("population") == p) & (pl.col("error") == e) & (pl.col("estimator") == name)).sort("R")
        ax[i, j].errorbar(d["R"] + 0.15 * (c - 1), d["bias"], yerr=2 * d["bias_se"], marker="o", markersize=4, linewidth=1, capsize=2, label=label)
    ax[i, j].axhline(0, color="k", linewidth=0.5)
    ax[i, j].set_title(f"{p}, {e} error", fontsize=10)
    ax[i, j].set_xticks(RADII)
    ax[i, 0].set_ylabel("relative bias [%]")
    ax[-1, j].set_xlabel("plot radius R [m]")
ax[0, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(figures / "fig_bias_by_radius.png", dpi=200)
publish(figures / "fig_bias_by_radius.png")
