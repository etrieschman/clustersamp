# clustersamp

Simulations for sampling trees from a frame of noisy GPS pins (one pin per planted tree, no tags).
The note lives in the Overleaf project (`~/dev/incubator/clustersamp`, `draft_pin_count.tex`).

## Run

```bash
uv run --python 3.11 --with-requirements requirements.txt snakemake -c8
```

Snakemake 7 needs Python 3.11 or older. Everything is seeded from `config.yaml`.

## Pipeline

| Rule | Script | Output |
|---|---|---|
| `make_data` | `code/make_data.py` | pins, true tree locations and biomass for each population and GPS error type |
| `get_inclusion_probability` | `code/get_inclusion_probability.py` | model inclusion probabilities, used only by the 2024 design |
| `get_expected_values` | `code/get_expected_values.py` | near-exact bias and one-visit variance of each estimator, by enumerating every pin with many error draws |
| `simulate_sample_design` | `code/simulate_sample_design.py` | repeated samples at each sample size, with analytical and bootstrap variance estimates |
| `inspect_results` | `code/inspect_results.py` | LaTeX tables and the bias-by-radius figure |

## Designs (`code/sample.py`)

| Config name | Class | What it is |
|---|---|---|
| `SRS` | `SRS` | benchmark: simple random sample of tagged trees |
| `PPSWR-SRSWR` | `PPSWR_SRS` | the 2024 design: pins with probability proportional to expected plot size, scaled plot means |
| `PIN-HH` | `PinCount(ratio=False)` | pin-count estimator: each tree weighted by 1 / (pins within R of pin + recorded offset) |
| `PIN-RATIO` | `PinCount(ratio=True)` | ratio form, needs the survival rate |
| `PIN-HH-1TREE`, `PIN-RATIO-1TREE` | `PinCount(subsample=1)` | locate every tree in the plot, measure biomass on one |

## Notation

| Note | Code |
|---|---|
| pins $c_i$ | `cluster_locs`, `punit_locs` |
| true tree locations $x_j$ | `tree_locs`, `sunit_locs` |
| attribute $y_j$ | `tree_bm`, `sunits` |
| plot radius $R$ | `radius_measure`, `measurement_rad` |
| pin count $C_j$ | `pin_counts` |
| plot totals $t_r$, $w_r$ | `cluster_total`, `cluster_weight` |

## Populations

- `kentland`: 107 trees measured by phone (`data/raw/wt_kentland_data.csv`). Recorded coordinates are the pins; true locations are the pins plus one draw of GPS error.
- `synthetic`: five small stands where biomass falls with the number of neighbors, so size is correlated with local density.
