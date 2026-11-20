# %%
# Expected value and one-draw variance of each estimator, by enumeration.
# Instead of repeating whole samples, visit every pin with many draws of the
# navigation error. A sample of k pins with replacement averages k such draws,
# so bias = E[draw] - target and Var(k draws) = Var(draw) / k.
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from make_data import record_noisy_gps_locs, get_patchy_mortality
from sample import measure_plots, choose_one_per_plot
from utils import mock_snakemake

def enumerate_draws(tree_bm, tree_locs, cluster_locs, radius_measure,
                    gps_error_type, gps_error, n_error_draws, **measure_kwargs):
    '''Returns (n pins, n error draws) arrays of plot statistics.'''
    N = len(cluster_locs)
    sunit_tree, punit_tree = cKDTree(tree_locs), cKDTree(cluster_locs)
    keys = ['size', 'mean', 'total', 'weight', 'total_1tree']
    draws = {key: np.zeros((N, n_error_draws)) for key in keys}
    for i in tqdm(range(N)):
        punit_locs_sampled = np.repeat(cluster_locs[[i]], n_error_draws, axis=0)
        plot_locs = record_noisy_gps_locs(punit_locs_sampled, gps_error_type, gps_error)
        plot_idx, sunit_idx, pin_counts = measure_plots(
            plot_locs, punit_locs_sampled, tree_locs, sunit_tree, punit_tree,
            radius_measure, **measure_kwargs)
        if len(sunit_idx) == 0:
            continue
        size = np.bincount(plot_idx, minlength=n_error_draws)
        weight = np.bincount(plot_idx, weights=1/pin_counts, minlength=n_error_draws)
        draws['size'][i] = size
        draws['mean'][i] = np.bincount(plot_idx, weights=tree_bm[sunit_idx], minlength=n_error_draws) / np.maximum(size, 1)
        draws['total'][i] = np.bincount(plot_idx, weights=tree_bm[sunit_idx]/pin_counts, minlength=n_error_draws)
        draws['weight'][i] = weight
        chosen = choose_one_per_plot(plot_idx, pin_counts)
        draws['total_1tree'][i][plot_idx[chosen]] = weight[plot_idx[chosen]] * tree_bm[sunit_idx[chosen]]
    return draws


def get_coverage(tree_locs, cluster_locs, radius_measure, gps_error_type, gps_error, n_error_draws):
    '''P(at least one pin within the measurement radius of the tree's apparent location), by tree.'''
    punit_tree = cKDTree(cluster_locs)
    coverage = np.zeros(len(tree_locs))
    for j in range(len(tree_locs)):
        tree_loc = np.repeat(tree_locs[[j]], n_error_draws, axis=0)
        # apparent location = pin + (tree - (pin + error)) = tree - error
        apparent_locs = 2*tree_loc - record_noisy_gps_locs(tree_loc, gps_error_type, gps_error)
        coverage[j] = (punit_tree.query_ball_point(apparent_locs, radius_measure, return_length=True) > 0).mean()
    return coverage


def summarize_draws(draws, target, unit_var, punit_weights=None, punit_scales=None,
                    survival=1., k=30, n_resamples=20000):
    '''One row per design: expected value, relative bias, and variance relative to SRS of trees.'''
    N, n_error_draws = draws['total'].shape
    rows = []
    def add_row(design, expected, var_draw, mc_se):
        rows.append({
            'design': design, 'expected': expected,
            'rel_bias_pct': 100 * (expected / target - 1),
            'mc_se_pct': 100 * mc_se / target,
            'deff': var_draw / unit_var})

    # 2024 design: pps selection, scaled plot mean
    if punit_weights is not None:
        z = punit_scales[:, np.newaxis] * draws['mean']
        expected = (punit_weights[:, np.newaxis] * z).sum() / n_error_draws
        var_draw = (punit_weights[:, np.newaxis] * (z - expected)**2).sum() / n_error_draws
        add_row('PPSWR-SRSWR', expected, var_draw, np.sqrt(var_draw / (N * n_error_draws)))

    # pin-count design: equal-probability selection
    for design, key in [('PIN-HH', 'total'), ('PIN-HH-1TREE', 'total_1tree')]:
        var_draw = draws[key].var()
        add_row(design, draws[key].mean(), var_draw, np.sqrt(var_draw / draws[key].size))

    # the ratio forms depend on the sample size, so resample the enumerated draws
    idx = np.random.randint(N * n_error_draws, size=(n_resamples, k))
    weights = draws['weight'].ravel()[idx].sum(1)
    for design, key in [('PIN-RATIO', 'total'), ('PIN-RATIO-1TREE', 'total_1tree')]:
        ratios = survival * draws[key].ravel()[idx].sum(1) / weights
        add_row(design, ratios.mean(), ratios.var() * k, ratios.std() / np.sqrt(n_resamples))

    return pd.DataFrame(rows)


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('get_expected_values',
                                   population='synthetic',
                                   gps_error_type='uniform',
                                   radius_measure=10)

    # parameters
    radius_measure = int(snakemake.wildcards.radius_measure)
    gps_error_type = snakemake.wildcards.gps_error_type
    gps_error = float(snakemake.params.gps_error[gps_error_type])
    n_error_draws = int(snakemake.params.n_error_draws)
    np.random.seed(int(snakemake.params.seed))

    # data
    tree_locs = np.loadtxt(snakemake.input.tree_locs, delimiter=',')
    cluster_locs = np.loadtxt(snakemake.input.cluster_locs, delimiter=',')
    tree_bm = np.loadtxt(snakemake.input.tree_bm, delimiter=',')
    inc_probs = np.loadtxt(snakemake.input.inc_probs, delimiter=',')
    N = len(cluster_locs)

    # selection weights and scales of the 2024 design
    ENis = inc_probs.sum(0)
    EM = ENis.sum()
    ppswr = {'punit_weights': ENis/EM, 'punit_scales': EM / (N * ENis)}

    # scenarios: how the plot is measured, and who navigates with what error
    other_gps = {'uniform': ('gaussian', 1.5), 'gaussian': ('uniform', 3.)}[gps_error_type]
    scenarios = {
        'base': {},
        'offset error 0.5m, recorded rule': {'offset_error': 0.5},
        'offset error 0.5m, true-distance rule': {'offset_error': 0.5, 'inclusion_rule': 'true'},
        'offset error 1.5m, recorded rule': {'offset_error': 1.5},
        'offset error 1.5m, true-distance rule': {'offset_error': 1.5, 'inclusion_rule': 'true'},
        'bearing error 2deg, recorded rule': {'bearing_error': 2.},
        'bearing error 5deg, recorded rule': {'bearing_error': 5.},
    }

    summ = []
    for scenario, measure_kwargs in scenarios.items():
        print(scenario)
        draws = enumerate_draws(tree_bm, tree_locs, cluster_locs, radius_measure,
                                gps_error_type, gps_error, n_error_draws, **measure_kwargs)
        summiter = summarize_draws(draws, tree_bm.mean(), tree_bm.var(), **ppswr)
        summiter['scenario'] = scenario
        summiter['target'] = 'mean'
        if scenario == 'base':
            coverage = get_coverage(tree_locs, cluster_locs, radius_measure,
                                    gps_error_type, gps_error, n_error_draws)
            summiter['trees_per_plot'] = draws['size'].mean()
            summiter['empty_plot_pct'] = 100 * (draws['size'] == 0).mean()
            summiter['coverage_bias_pct'] = 100 * ((coverage * tree_bm).sum() / tree_bm.sum() - 1)
            summiter['coverage_min'] = coverage.min()
        summ.append(summiter)

    # the surveyor's navigation error has a different distribution than the one
    # that produced the pins (the 2024 design's weights assume they match)
    scenario = f'navigation error {other_gps[0]} ({other_gps[1]}m)'
    print(scenario)
    draws = enumerate_draws(tree_bm, tree_locs, cluster_locs, radius_measure,
                            other_gps[0], other_gps[1], n_error_draws)
    summiter = summarize_draws(draws, tree_bm.mean(), tree_bm.var(), **ppswr)
    summiter['scenario'] = scenario
    summiter['target'] = 'mean'
    summ.append(summiter)

    # patchy mortality: dead trees leave no trace and their pins stay in the frame.
    # Targets are per planted tree: survival rate, and mean biomass with dead = 0.
    print('mortality')
    alive = get_patchy_mortality(tree_locs)
    for target, values in [('survival', np.ones(alive.sum())), ('mean per planted tree', tree_bm[alive])]:
        draws = enumerate_draws(values, tree_locs[alive], cluster_locs, radius_measure,
                                gps_error_type, gps_error, n_error_draws)
        unit_values = np.r_[values, np.zeros(N - alive.sum())]
        summiter = summarize_draws(draws, unit_values.mean(), unit_values.var(), survival=alive.mean())
        summiter['scenario'] = 'patchy mortality'
        summiter['target'] = target
        summ.append(summiter)

    summ = pd.concat(summ, ignore_index=True)
    summ.to_csv(snakemake.output.expected, index=False)
    print(summ.round(3).to_string())

# %%
