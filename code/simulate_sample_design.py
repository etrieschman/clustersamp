# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
from scipy.stats import norm

from sample import Sample, SRS, PPSWR_SRS, PinCount
from utils import mock_snakemake

PATH_REPO = '..'
PATH_RESULTS = PATH_REPO + '/results/'

def get_bootsrapped_results(sampler:Sample, n_min=2, ns=100, n_repeats=500):
    # arrays to store results
    means = np.zeros((ns, n_repeats))
    vars = np.zeros_like(means)
    vars_boot = np.zeros_like(means)
    n_samples = np.zeros_like(means) # trees located
    n_measured = np.zeros_like(means) # trees measured for biomass

    # run sampler
    for i in tqdm(range(ns)):
        for j in range(n_repeats):
            sample_idxs = sampler.sample(n_min+i)
            if 'unit_idxs' in sample_idxs:
                n_samples[i,j] = len(sample_idxs['unit_idxs'])
            if 'sunit_idxs' in sample_idxs:
                for k in sample_idxs['sunit_idxs']:
                    n_samples[i,j] += len(k)
            n_measured[i,j] = sample_idxs['n_measured']
            means[i, j] = sampler.sample_mean
            vars[i, j] = sampler.sample_mean_variance
            vars_boot[i, j] = sampler.sample_mean_variance_boot
    results = {
        'means': means, 'vars': vars, 'vars_boot': vars_boot, 
        'n_samples': n_samples, 'n_measured': n_measured,
        'params':{'n_min':n_min, 'ns':ns, 'n_repeats':n_repeats}
    }

    return results

def plot_bootstrapped_results(true_mean, results, outfile=None, alpha=0.05, n_tree_ylim=None):
    # plot
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 8))
    flierprops = dict(marker='.', markerfacecolor='grey', markersize=5, markeredgecolor='none', alpha=0.75)
    boxprops = {'linewidth':0.5}
    ax[2].boxplot(results['n_samples'].T, patch_artist=False, 
                  boxprops=boxprops, flierprops=flierprops)
    meanprops = {'linewidth':1, 
                'color':'C3',
                'alpha':0.75
                }
    ax[1].boxplot(results['means'].T, patch_artist=False,
                  boxprops=boxprops, flierprops=flierprops, 
                  meanprops=meanprops, showmeans=True, meanline=True)
    # add true mean
    ax[1].axhline(true_mean, alpha=0.75, color='C0', linestyle='-', linewidth=1, label='true mean')
    ax[0].plot(results['means'].var(1), alpha=0.75, color='k', linestyle='--', linewidth=1, 
               label='variance of means across repeats')
    zscore = norm.ppf(1 - alpha/2)
    ax[0].plot(results['vars'].mean(1), alpha=0.75, color='C0', label='analytical derivation')
    ax[0].fill_between(x=np.arange(len(results['vars'].mean(1))),
                       y1=results['vars'].min(1),
                       y2=results['vars'].max(1),
                       alpha=0.15, color='C0')
    ax[0].plot(results['vars_boot'].mean(1), alpha=0.75, color='C1', label='bootstrapped estimation')
    ax[0].fill_between(x=np.arange(len(results['vars_boot'].mean(1))),
                       y1=results['vars_boot'].min(1), 
                       y2=results['vars_boot'].max(1),
                       alpha=0.15, color='C1', label=f'range (min-max)')
    ax[2].set_xticks(
        np.arange(results['params']['ns'], step=10), 
        np.arange(results['params']['n_min'], 
                  results['params']['ns']+results['params']['n_min'], step=10))
    ax[2].set_xlabel('n primary samples')
    ax[1].set_ylabel('distribution of sample means')
    ax[2].set_ylabel('n trees located')
    if n_tree_ylim is not None:
        ax[2].set_ylim(n_tree_ylim)
    ax[0].set_ylabel('estimator variance')
    # add lines just for legend
    h_sample_med = mlines.Line2D([], [], color='C1', marker=None, linestyle='--', linewidth=1, label='sample mean median')
    h_sample_mean = mlines.Line2D([], [], color='C3', marker=None, linestyle='--', linewidth=1, label='sample mean mean')
    h_true_mean = mlines.Line2D([], [], color='C0', marker=None, linestyle='-', linewidth=1, label='true mean')
    ax[1].legend(handles=[h_true_mean, h_sample_med, h_sample_mean], frameon=False)
    ax[0].legend(frameon=False)
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('simulate_sample_design',
                                   population='synthetic',
                                   gps_error_type='uniform',
                                   radius_measure=10,
                                   sample_design='PIN-RATIO')
    
    # parameters
    radius_measure = int(snakemake.wildcards.radius_measure)
    gps_error_type = snakemake.wildcards.gps_error_type
    gps_error = float(snakemake.params.gps_error[gps_error_type])
    sample_design = snakemake.wildcards.sample_design
    bootstraps_for_variance = int(snakemake.params.bootstraps_for_variance)
    n_samples_min = int(snakemake.params.n_samples_min)
    n_samples_inc = int(snakemake.params.n_samples_inc)
    n_repeats = int(snakemake.params.n_repeats)
    np.random.seed(int(snakemake.params.seed))

    # data
    tree_locs = np.loadtxt(snakemake.input.tree_locs, delimiter=',')
    cluster_locs = np.loadtxt(snakemake.input.cluster_locs, delimiter=',')
    tree_bm = np.loadtxt(snakemake.input.tree_bm, delimiter=',')
    inc_probs = np.loadtxt(snakemake.input.inc_probs, delimiter=',')

    # calculate expected value of cluster size and scaled values (2024 design only)
    # row is secondary unit
    # col is primary unit
    ENis = inc_probs.sum(0)
    EM = ENis.sum()
    N = len(tree_locs)
    Zcoef = (EM / (N * ENis))

    # parameter_dictionary
    ppswr_params = {
        'sunits':tree_bm, 'sunit_locs':tree_locs, 
        'punit_weights':ENis/EM, 'punit_scales':Zcoef,
        'punit_locs':cluster_locs, 
        'gps_error_type':gps_error_type,
        'gps_error':gps_error, 
        'measurement_rad':radius_measure, 
        'n_bootstraps':bootstraps_for_variance}
    pin_params = {
        'sunits':tree_bm, 'sunit_locs':tree_locs, 
        'punit_locs':cluster_locs, 
        'gps_error_type':gps_error_type,
        'gps_error':gps_error, 
        'measurement_rad':radius_measure, 
        'n_bootstraps':bootstraps_for_variance}
    sampler_dict = {
        'SRS': {
            'sampler':SRS,
            'params':{'units':tree_bm, 'n_bootstraps':bootstraps_for_variance}
            },
        'PPSWR-SRSWOR': {
            'sampler':PPSWR_SRS,
            'params':{**ppswr_params, 'replace':False}
            },
        'PPSWR-SRSWR': {
            'sampler':PPSWR_SRS,
            'params':{**ppswr_params, 'replace':True}
            },
        'PIN-HH': {
            'sampler':PinCount,
            'params':{**pin_params, 'ratio':False, 'subsample':None}
            },
        'PIN-RATIO': {
            'sampler':PinCount,
            'params':{**pin_params, 'ratio':True, 'subsample':None}
            },
        'PIN-HH-1TREE': {
            'sampler':PinCount,
            'params':{**pin_params, 'ratio':False, 'subsample':1}
            },
        'PIN-RATIO-1TREE': {
            'sampler':PinCount,
            'params':{**pin_params, 'ratio':True, 'subsample':1}
            },
    }

    # define sampler object using sampler_dict
    sampler = sampler_dict[sample_design]['sampler'](**sampler_dict[sample_design]['params'])

    # get results
    results = get_bootsrapped_results(sampler, n_min=n_samples_min, ns=n_samples_inc, n_repeats=n_repeats)
    np.savez(snakemake.output.results, **results)

    # plot results
    plot_bootstrapped_results(true_mean=tree_bm.mean(), results=results, 
                                outfile=snakemake.output.fig, n_tree_ylim=None)


# %%
