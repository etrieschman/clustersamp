# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyproj import Transformer

from make_data import get_random_tree_locs, get_random_tree_bm, record_noisy_gps_locs_gaus
from sample import SRS, PPSWR_SRS
from get_inclusion_probability import get_distance_matrix, get_rough_inclusion_probs
from simulate_sample_design import get_bootsrapped_results, plot_bootstrapped_results

GPS_ERROR_VARIANCE = 0.01
MEASUREMENT_RADIUS = 10
# GENERATE DATA AND VISUALIZE
N = 200
tree_locs = get_random_tree_locs(n_trees=N)
tree_bm = get_random_tree_bm(tree_locs)
cluster_locs = record_noisy_gps_locs_gaus(tree_locs, GPS_ERROR_VARIANCE)

plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.75, s=tree_bm*8, label='true tree location (and size)')
plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='noisy GPS tree location')
plt.legend()
plt.title('Tree and cluster locations')
plt.show()

# CALCULATE INCLUSION PROBABILITIES AND SCALED VALUES
# get inclusion probabilities
inc_probs = get_rough_inclusion_probs(cluster_locs, eps=1e-1)
# calculate expected value of cluster size (UNNORMALIZED)
EN = inc_probs.sum(0)
EM = EN.sum()
EF = inc_probs.sum(1)
EFk = (1/EN) * (inc_probs @ EF)
N = len(tree_locs)
Zcoef = (EM / N) * (1/EFk)
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
    sunits=tree_bm, sunit_locs=tree_locs, 
    punit_weights=EN/EM, punit_scales=Zcoef, 
    punit_locs=cluster_locs, 
    gps_error_type='gaussian',
    gps_error=GPS_ERROR_VARIANCE, 
    measurement_rad=MEASUREMENT_RADIUS, 
    replace=True, n_bootstraps=50)

ppswr_srswr_results = get_bootsrapped_results(ppswr_srswr, n_min=10, ns=45, n_repeats=20)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=ppswr_srswr_results, 
                          outfile='ppswr_srswr.png', n_tree_ylim=(0, 200))

# %%
# BOOTSTRAP RESULTS FROM PPSWR-SRSWOR SAMPLING APPROACH
ppswr_srswor = PPSWR_SRS(
    sunits_scaled=Z, sunit_locs=tree_locs, 
    punit_weights=EN/EM, punit_locs=cluster_locs, 
    gps_error_var=GPS_ERROR_VARIANCE, 
    measurement_rad=MEASUREMENT_RADIUS, 
    replace=False, n_bootstraps=50)

ppswr_srswor_results = get_bootsrapped_results(ppswr_srswor, n_min=10, ns=90, n_repeats=100)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=ppswr_srswor_results, 
                          outfile='ppswr_srswor.png', n_tree_ylim=(0, 200))

# %%
