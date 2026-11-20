# %%
import numpy as np
import matplotlib.pyplot as plt

from make_data import get_random_tree_locs, get_random_tree_bm, record_noisy_gps_locs_gaus
from sample import SRS, PinCount
from simulate_sample_design import get_bootsrapped_results, plot_bootstrapped_results

GPS_ERROR_STD = 2.5
MEASUREMENT_RADIUS = 10
# GENERATE DATA AND VISUALIZE
tree_locs = get_random_tree_locs()
tree_bm = get_random_tree_bm(tree_locs)
cluster_locs = record_noisy_gps_locs_gaus(tree_locs, GPS_ERROR_STD)

plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.75, s=tree_bm*8, label='true tree location (and size)')
plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='noisy GPS pin')
plt.legend()
plt.title('Tree and pin locations')
plt.show()

# %%
# BOOTSTRAP RESULTS FROM SIMPLE RANDOM SAMPLING APPROACH
srs = SRS(tree_bm, 50)
srs_results = get_bootsrapped_results(srs, n_min=10, ns=45, n_repeats=100)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=srs_results)

# %%
# BOOTSTRAP RESULTS FROM THE PIN-COUNT DESIGN (RATIO FORM)
pin = PinCount(
    sunits=tree_bm, sunit_locs=tree_locs, punit_locs=cluster_locs, 
    gps_error_type='gaussian', gps_error=GPS_ERROR_STD, 
    measurement_rad=MEASUREMENT_RADIUS, 
    ratio=True, subsample=None, n_bootstraps=50)

pin_results = get_bootsrapped_results(pin, n_min=10, ns=45, n_repeats=100)
plot_bootstrapped_results(true_mean=tree_bm.mean(), results=pin_results)

# %%
