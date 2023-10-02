# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from make_data import get_random_tree_locs, get_tree_bm, record_noisy_gps_locs
from sample import SRS, PPSWR_SRS
from inclusion_prob import GPS_ERROR_VARIANCE, MEASUREMENT_RADIUS
from inclusion_prob import get_distance_matrix, get_rough_inclusion_probs

# %%
# GENERATE DATA AND VISUALIZE
N = 200
tree_locs = get_random_tree_locs(n_trees=N)
tree_bm = get_tree_bm(tree_locs)
cluster_locs = record_noisy_gps_locs(tree_locs, GPS_ERROR_VARIANCE)

plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.75, s=tree_bm*8, label='true tree location (and size)')
plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='noisy GPS tree location')
plt.legend()
plt.title('Tree and cluster locations')
plt.show()

# %%
# CALCULATE INCLUSION PROBABILITIES
inc_probs = get_rough_inclusion_probs(cluster_locs, eps=1e-1)
# CALCULATE EXPECTED VALUE OF CLUSTER SIZE (UNNORMALIZED)
EN = inc_probs.sum(0)
EM = EN.sum()
EF = inc_probs.sum(1)
# CALCULATE SCALED VALUES
Z = EM / (N*EF) * tree_bm


# %%
# BOOTSTRAP RESULTS FROM SIMPLE RANDOM SAMPLING APPROACH
srs = SRS(tree_bm, 50)
n_min = 2
ns = 100
n_repeats = 500

def get_bootsrapped_results(sampler, n_min=2, ns=100, n_repeats=500):
    # arrays to store results
    means = np.zeros((ns, n_repeats))
    vars = np.zeros_like(means)
    vars_boot = np.zeros_like(means)

    # run sampler
    for i in tqdm(range(ns)):
        for j in range(n_repeats):
            sampler.sample(n_min+i)
            means[i, j] = sampler.sample_mean
            vars[i, j] = sampler.sample_mean_variance
            vars_boot[i, j] = sampler.sample_mean_variance_boot

    # plot
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax[1].boxplot(means.T, patch_artist=False)
    ax[1].axhline(sampler.true_mean, label='true mean')
    ax[0].plot(means.var(1), label='bootstrap mean variance')
    ax[0].plot(vars.mean(1), label='sample mean variance (analytical)')
    ax[0].plot(vars_boot.mean(1), label='sample mean variance (bootstrap)')
    ax[1].set_xticks(np.arange(ns, step=10), np.arange(n_min, ns+n_min, step=10))
    ax[1].set_xlabel('number of samples')
    ax[1].set_ylabel('distribution of sample means')
    ax[0].set_ylabel('variance in sample mean')
    ax[1].legend()
    ax[0].legend()
    plt.show()


# %%
# BOOTSTRAP RESULTS FROM PPSWR-SRSWR SAMPLING APPROACH
ppswr_srswr = PPSWR_SRS(
    sunits_scaled=Z, sunit_locs=tree_locs, 
    punit_weights=EN/EM, punit_locs=cluster_locs, 
    gps_error_var=GPS_ERROR_VARIANCE, 
    measurement_rad=MEASUREMENT_RADIUS, 
    replace=True, n_bootstraps=50)

ppswr_srswr.sample(5)
ppswr_srswr.get_stats()

# %%
n_min = 2
ns = 100
n_repeats = 25
means = np.zeros((ns, n_repeats))
vars = np.zeros_like(means)
vars_boot = np.zeros_like(means)

for i in tqdm(range(ns)):
    for j in range(n_repeats):
        ppswr_srswr.sample(n_min+i)
        means[i, j] = ppswr_srswr.sample_mean
        vars[i, j] = ppswr_srswr.sample_mean_variance
        vars_boot[i, j] = ppswr_srswr.sample_mean_variance_boot

# PLOT
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax[1].boxplot(means.T, patch_artist=False)
ax[1].axhline(ppswr_srswr.true_mean, label='true mean')
ax[0].plot(means.var(1), label='bootstrap mean variance')
ax[0].plot(vars.mean(1), label='sample mean variance (analytical)')
ax[0].plot(vars_boot.mean(1), label='sample mean variance (bootstrap)')
ax[1].set_xticks(np.arange(ns, step=10), np.arange(n_min, ns+n_min, step=10))
ax[1].set_xlabel('number of samples')
ax[1].set_ylabel('distribution of sample means')
ax[0].set_ylabel('variance in sample mean')
ax[1].legend()
ax[0].legend()
plt.show()
# %%
