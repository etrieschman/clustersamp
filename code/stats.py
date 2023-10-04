import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sample import Sample

PATH_REPO = '..'
PATH_RESULTS = PATH_REPO + '/results/'

def get_bootsrapped_results(sampler:Sample, n_min=2, ns=100, n_repeats=500):
    # arrays to store results
    means = np.zeros((ns, n_repeats))
    vars = np.zeros_like(means)
    vars_boot = np.zeros_like(means)
    n_samples = np.zeros_like(means)

    # run sampler
    for i in tqdm(range(ns)):
        for j in range(n_repeats):
            sample_idxs = sampler.sample(n_min+i)
            if 'unit_idxs' in sample_idxs:
                n_samples[i,j] = len(sample_idxs['unit_idxs'])
            if 'sunit_idxs' in sample_idxs:
                for k in sample_idxs['sunit_idxs']:
                    n_samples[i,j] += len(k)
            means[i, j] = sampler.sample_mean
            vars[i, j] = sampler.sample_mean_variance
            vars_boot[i, j] = sampler.sample_mean_variance_boot
    results = {
        'means': means, 'vars': vars, 'vars_boot': vars_boot, 'n_samples': n_samples,
        'params':{'n_min':n_min, 'ns':ns, 'n_repeats':n_repeats}
    }

    return results

def plot_bootstrapped_results(true_mean, results, outfile=None, n_tree_ylim=None):
    # plot
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6, 8))
    ax[2].boxplot(results['n_samples'].T, patch_artist=False)
    ax[1].boxplot(results['means'].T, patch_artist=False)
    ax[1].axhline(true_mean, label='true mean')
    ax[0].plot(results['means'].var(1), label='bootstrap mean variance')
    ax[0].plot(results['vars'].mean(1), label='sample mean variance (analytical)')
    ax[0].plot(results['vars_boot'].mean(1), label='sample mean variance (bootstrap)')
    ax[2].set_xticks(
        np.arange(results['params']['ns'], step=10), 
        np.arange(results['params']['n_min'], 
                  results['params']['ns']+results['params']['n_min'], step=10))
    ax[2].set_xlabel('n primary samples')
    ax[1].set_ylabel('distribution of sample means')
    ax[2].set_ylabel('n tree measurements')
    if n_tree_ylim is not None:
        ax[2].set_ylim(n_tree_ylim)
    ax[0].set_ylabel('variance in sample mean')
    ax[1].legend()
    ax[0].legend()
    if outfile is not None:
        plt.savefig(PATH_RESULTS + outfile, bbox_inches='tight')
    plt.show()