# %%
# SETUP
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt

from utils import mock_snakemake

PATH_RESULTS = 'results/'
PATH_DATA = 'data/'

DESIGN_LABELS = {
    'SRS': 'SRS of tagged trees',
    'PPSWR-SRSWR': '2024 design',
    'PIN-HH': 'Pin count',
    'PIN-RATIO': 'Pin count, ratio',
    'PIN-HH-1TREE': 'Pin count, one tree',
    'PIN-RATIO-1TREE': 'Pin count, ratio, one tree',
}

# helper functions
def read_expected(populations, gps_error_types, radius_measures):
    summ = []
    for population in populations:
        for gps_error_type in gps_error_types:
            for radius_measure in radius_measures:
                summiter = pd.read_csv(PATH_RESULTS + f'{population}/{gps_error_type}/expected_rad{radius_measure}.csv')
                summiter[['population', 'gps_error_type', 'radius_measure']] = population, gps_error_type, radius_measure
                summ.append(summiter)
    return pd.concat(summ, ignore_index=True)

def fmt_bias(x):
    return '-' if pd.isna(x) else f'{x:+.2f}'

def get_ci_coverage(means, vars, true_mean, df, alpha=0.1):
    tstats = (means - true_mean) / np.sqrt(vars)
    return (np.abs(tstats) < sps.t.ppf(1 - alpha/2, df)).mean()


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('inspect_results')

    # set conditions
    populations = snakemake.params.population
    gps_error_types = snakemake.params.gps_error_type
    radius_measures = sorted(snakemake.params.radius_measure)
    radius_measures_expected = sorted(snakemake.params.radius_measure_expected)
    sample_designs = snakemake.params.sample_design
    nmin = int(snakemake.params.n_samples_min)
    ns = [25, 45, 65]

    expected = read_expected(populations, gps_error_types, radius_measures_expected)

    # %%
    # TABLE: EXPECTED VALUES (near-exact bias and variance relative to SRS)
    base = expected.loc[(expected.scenario == 'base') & expected.radius_measure.isin(radius_measures)]
    index = ['population', 'gps_error_type', 'radius_measure']
    bias = base.pivot(index=index, columns='design', values='rel_bias_pct')
    deff = base.pivot(index=index, columns='design', values='deff')
    plots = base.groupby(index)[['trees_per_plot']].first()
    summ = pd.concat([
        plots.trees_per_plot.map('{:.1f}'.format).rename(('', 'Trees per plot')),
        *[bias[d].map(fmt_bias).rename(('Relative bias (\\%)', DESIGN_LABELS[d]))
          for d in ['PPSWR-SRSWR', 'PIN-HH', 'PIN-RATIO']],
        *[deff[d].map('{:.2f}'.format).rename(('Variance relative to SRS', DESIGN_LABELS[d]))
          for d in ['PPSWR-SRSWR', 'PIN-HH', 'PIN-RATIO', 'PIN-RATIO-1TREE']],
        ], axis=1)
    summ.columns = pd.MultiIndex.from_tuples(summ.columns)
    summ.index.names = ['Population', 'GPS error', 'Radius (m)']
    summ.to_latex(buf=snakemake.output.tab_expected, multicolumn_format='c',
                  column_format='lll|r|rrr|rrrr')
    print(summ.to_string())
    print('max mc se of relative bias (%):', base.groupby('design').mc_se_pct.max().round(3).to_dict())

    # %%
    # TABLE: SCENARIOS (how the plot is measured), synthetic population at the smallest radius
    radius_measure = radius_measures[0]
    scen = expected.loc[(expected.population == 'synthetic') & (expected.radius_measure == radius_measure)].copy()
    small = expected.loc[(expected.population == 'synthetic') & (expected.scenario == 'base')
                         & (expected.radius_measure == radius_measures_expected[0])].copy()
    small['scenario'] = f'measurement radius {radius_measures_expected[0]}m'
    scen = pd.concat([scen, small], ignore_index=True)
    scen['scenario'] = scen.scenario.str.replace(r'navigation error.*', 'navigation error from another distribution', regex=True)
    scen.loc[scen.target != 'mean', 'scenario'] = scen.scenario + ': ' + scen.target
    # the ratio form of the survival rate is the known survival rate itself
    scen = scen.loc[~((scen.target == 'survival') & scen.design.str.contains('RATIO'))]
    scenario_order = list(dict.fromkeys(scen.scenario))
    summ = scen.pivot(index='scenario', columns=['gps_error_type', 'design'], values='rel_bias_pct')
    summ = summ.loc[scenario_order, [(g, d) for g in gps_error_types for d in ['PPSWR-SRSWR', 'PIN-HH', 'PIN-RATIO']]]
    summ = summ.map(fmt_bias).rename(columns=DESIGN_LABELS, level=1)
    summ.index.name = 'Scenario'
    summ.columns.names = ['GPS error', '']
    summ.to_latex(buf=snakemake.output.tab_scenarios, multicolumn_format='c',
                  column_format='l|rrr|rrr')
    print(summ.to_string())

    # %%
    # FIGURE: BIAS BY MEASUREMENT RADIUS
    base = expected.loc[expected.scenario == 'base']
    fig, ax = plt.subplots(nrows=len(populations), ncols=len(gps_error_types), sharex=True,
                           figsize=(4*len(gps_error_types), 3*len(populations)), squeeze=False)
    for i, population in enumerate(populations):
        for j, gps_error_type in enumerate(gps_error_types):
            for c, design in enumerate(['PPSWR-SRSWR', 'PIN-HH', 'PIN-RATIO']):
                d = base.loc[(base.population == population) & (base.gps_error_type == gps_error_type)
                             & (base.design == design)].sort_values('radius_measure')
                ax[i,j].errorbar(d.radius_measure + 0.15*(c-1), d.rel_bias_pct, yerr=2*d.mc_se_pct,
                                 marker='o', markersize=4, linewidth=1, capsize=2,
                                 color=f'C{c}', label=DESIGN_LABELS[design])
            ax[i,j].axhline(0, color='k', linewidth=0.5)
            ax[i,j].set_title(f'{population}, {gps_error_type} GPS error', fontsize=10)
            ax[i,j].set_xticks(radius_measures_expected)
        ax[i,0].set_ylabel('relative bias (%)')
    for j in range(len(gps_error_types)):
        ax[-1,j].set_xlabel('measurement radius (m)')
    ax[0,0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(snakemake.output.fig_bias, dpi=300, bbox_inches='tight')

    # %%
    # TABLES: REPEATED SAMPLES (do the variance estimators and intervals work?)
    radius_measure = radius_measures[0]
    for n, outfile in zip(ns, snakemake.output.tab_simulation):
        nindex = n - nmin
        summ = []
        for population in populations:
            for gps_error_type in gps_error_types:
                tree_bm = np.loadtxt(PATH_DATA + f'processed/{population}/{gps_error_type}/tree_bm.txt', delimiter=',')
                true_mean = tree_bm.mean()
                for sample_design in sample_designs:
                    results = np.load(
                        PATH_RESULTS + f'{population}/{gps_error_type}/results_rad{radius_measure}_{sample_design}.npz')
                    means, vars, vars_boot = results['means'][nindex], results['vars'][nindex], results['vars_boot'][nindex]
                    summ.append({
                        'Population': population,
                        'GPS error': gps_error_type,
                        'Design': DESIGN_LABELS[sample_design],
                        'Relative bias (\\%)': f'{100*(means.mean()/true_mean - 1):+.2f}',
                        '(s.e.)': f'({100*means.std(ddof=1)/np.sqrt(len(means))/true_mean:.2f})',
                        'CV (\\%)': f'{100*means.std(ddof=1)/true_mean:.1f}',
                        'Var. est. / var.': f'{vars.mean()/means.var(ddof=1):.2f}',
                        'Bootstrap / var.': f'{vars_boot.mean()/means.var(ddof=1):.2f}',
                        '90\\% CI coverage': f'{100*get_ci_coverage(means, vars, true_mean, n-1):.1f}',
                        'Trees located': f"{results['n_samples'][nindex].mean():.0f}",
                        'Trees measured': f"{results['n_measured'][nindex].mean():.0f}",
                    })
        summ = pd.DataFrame(summ).set_index(['Population', 'GPS error', 'Design'])
        summ.to_latex(buf=outfile, column_format='lll|rr|r|rr|r|rr')
        print(f'n = {n} pins, radius = {radius_measure}m')
        print(summ.to_string())

# %%
