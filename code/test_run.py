# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyproj import Transformer

from make_data import get_random_tree_locs, get_tree_bm, record_noisy_gps_locs_unif
from sample import SRS, PPSWR_SRS
from inclusion_prob import GPS_ERROR_VARIANCE, MEASUREMENT_RADIUS
from inclusion_prob import get_distance_matrix, get_rough_inclusion_probs
from stats import get_bootsrapped_results, plot_bootstrapped_results

# %%
# READ IN ACTUAL DATA
treedf = pd.read_csv('../data/wt_kentland_data.csv', header=0)
cluster_loc_lat = treedf.lat.values
cluster_loc_lon = treedf.lon.values
tree_bm = treedf.biomass_TCO2e.values

def latlon_to_utm(lat, lon):
    # Define the projection for WGS84 (used for GPS coordinates)
    wgs84_crs = 'EPSG:4326'
    
    # Define the UTM zone based on the longitude
    utm_zone = (((lon[0] + 180) / 6) + 1).astype(int)
    # Northern Hemisphere has positive latitudes, hence 'north=True'
    # For Southern Hemisphere use 'north=False'
    north = lat[0] >= 0
    utm_crs = f'EPSG:{32600 + utm_zone if north else 32700 + utm_zone}'
    
    # Create a Transformer object for converting WGS84 to UTM
    transformer = Transformer.from_crs(wgs84_crs, utm_crs)
    
    # Transform the latitude and longitude to UTM coordinates
    utm_x, utm_y = transformer.transform(lat, lon)
    
    return utm_x, utm_y


cluster_loc_x, cluster_loc_y = latlon_to_utm(cluster_loc_lat, cluster_loc_lon)
cluster_locs = np.array([cluster_loc_x, cluster_loc_y]).T
tree_locs = record_noisy_gps_locs_unif(cluster_locs, radius=5)


plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.75, s=tree_bm*50, label='true tree location (and size)')
plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='noisy GPS tree location')
plt.legend()
plt.title('Tree and cluster locations')
plt.show()

# %%
# CALCULATE INCLUSION PROBABILITIES AND SCALED VALUES
# get inclusion probabilities
N = len(tree_locs)
inc_probs = get_rough_inclusion_probs(cluster_locs, eps=1e-1)
# calculate expected value of cluster size (UNNORMALIZED)
EN = inc_probs.sum(0)
EM = EN.sum()
EF = inc_probs.sum(1)
# calculate scaled values
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
