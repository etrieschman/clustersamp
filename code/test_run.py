# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


from make_data import get_real_tree_data
from sample import SRS, PPSWR_SRS
from inclusion_prob import GPS_ERROR_VARIANCE, MEASUREMENT_RADIUS
from inclusion_prob import get_inclusion_probs
from stats import get_bootsrapped_results, plot_bootstrapped_results


radius_gps, radius_measure = 5, 7
# %%
# READIN DATA
tree_locs, cluster_locs, tree_bm = get_real_tree_data(radius_gps)

plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.75, s=tree_bm*50, label='true tree location (and size)')
plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='noisy GPS tree location')
plt.legend()
plt.title('Tree and cluster locations')
plt.show()

# %%
# CALCULATE INCLUSION PROBABILITIES AND SCALED VALUES
# inclusion probabilities
inc_probs = get_inclusion_probs(cluster_locs, radius_gps, radius_measure)

# %%
inc_probs_ii = inc_probs.copy()
inc_probs_ii = inc_probs_ii + inc_probs_ii.T - np.diag(inc_probs_ii.diagonal())

# calculate expected value of cluster size
EN = inc_probs.sum(0)
EM = EN.sum()
EF = inc_probs.sum(1)
# calculate scaled values
N = len(tree_locs)
Z = EM / (N*EF) * tree_bm

# %%
# BOOTSTRAP RESULTS FROM SIMPLE RANDOM SAMPLING APPROACH
srs = SRS(tree_bm, 50)
srs_results = get_bootsrapped_results(srs, n_min=10, ns=90, n_repeats=100)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=srs_results, 
                          outfile='srs.png', n_tree_ylim=(0, 200))

# %%
# BOOTSTRAP RESULTS FROM PPSWR-SRSWR SAMPLING APPROACH
ppswr_srswr = PPSWR_SRS(
    sunits_scaled=Z, sunit_locs=tree_locs, 
    punit_weights=EN/EM, punit_locs=cluster_locs, 
    gps_error_type='unif',
    gps_error=5, 
    measurement_rad=10, 
    replace=True, n_bootstraps=50)

ppswr_srswr_results = get_bootsrapped_results(ppswr_srswr, n_min=10, ns=90, n_repeats=100)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=ppswr_srswr_results, 
                          outfile='ppswr_srswr.png', n_tree_ylim=(0, 200))

# %%
# BOOTSTRAP RESULTS FROM PPSWR-SRSWOR SAMPLING APPROACH
ppswr_srswor = PPSWR_SRS(
    sunits_scaled=Z, sunit_locs=tree_locs, 
    punit_weights=EN/EM, punit_locs=cluster_locs, 
    gps_error_type='unif',
    gps_error=5,
    measurement_rad=5, 
    replace=False, n_bootstraps=50)

ppswr_srswor_results = get_bootsrapped_results(ppswr_srswor, n_min=10, ns=90, n_repeats=100)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=ppswr_srswor_results, 
                          outfile='ppswr_srswor.png', n_tree_ylim=(0, 200))

# %%
