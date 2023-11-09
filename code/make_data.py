import numpy as np
import pandas as pd
from pyproj import Transformer

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


# READ IN ACTUAL DATA
def get_real_tree_data(radius_gps):
    # readin dataset
    treedf = pd.read_csv('../data/wt_kentland_data.csv', header=0)
    cluster_loc_lat = treedf.lat.values
    cluster_loc_lon = treedf.lon.values
    tree_bm = treedf.biomass_TCO2e.values

    # convert to equal-area
    cluster_loc_x, cluster_loc_y = latlon_to_utm(cluster_loc_lat, cluster_loc_lon)
    cluster_locs = np.array([cluster_loc_x, cluster_loc_y]).T
    tree_locs = record_noisy_gps_locs_unif(cluster_locs, radius=radius_gps)

    return tree_locs, cluster_locs, tree_bm


# GENERATE TREE LOCATIONS FROM A MIXTURE OF GAUSSIANS
# use mixture to simulate clustering/bunching of trees
def get_random_tree_locs(n_trees, n_groups=10, std_scale=0.5):
    # Define the parameters for each Gaussian component
    means = np.random.normal(size=n_groups*2).reshape((n_groups, 2))  # Mean of each component
    std_devs = np.ones(n_groups) * std_scale  # Standard deviation of each component
    mixing_coeffs = np.ones(n_groups) / n_groups  # Mixing coefficients (weights)
    # Initialize an array to store the generated data points
    tree_locs = np.zeros((n_trees, 2))

    # Generate data points from the mixture
    for i in range(n_trees):
        # Randomly select a Gaussian component based on mixing coefficients
        group_idx = np.random.choice(n_groups, p=mixing_coeffs)
        # Generate a random data point from the selected component
        sample = np.random.normal(loc=means[group_idx], scale=std_devs[group_idx])
        # Append the sample to the data_points array
        tree_locs[i] = sample
    
    return tree_locs

# ASSIGN BIOMASS TO TREES
def get_random_tree_bm(tree_locs, gamma_shape=3, gamma_scale=2, loc_corr=0.5, loc_corr_noise_std=0.025):
    # get gamma distributed biomass
    tree_bm = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=len(tree_locs))
    # include correlation with lat/lon, and some noise
    tree_bm += (tree_locs[:,0]*loc_corr + 
                tree_locs[:,1]*loc_corr + 
                np.random.normal(scale=loc_corr_noise_std, size=len(tree_locs)))
    return tree_bm

# RECORD NOISY LOCATIONS (THESE BECOME OUR CLUSTER AND DEPLOYED MEASUREMENT LOCATIONS)
def record_noisy_gps_locs_gaus(tree_locs, gps_noise):
    cluster_locs = np.zeros(tree_locs.shape)
    for i, tree in enumerate(tree_locs):
        cluster_locs[i] = np.random.multivariate_normal(mean=tree, cov=np.identity(2)*gps_noise)
    return cluster_locs

def record_noisy_gps_locs_unif(tree_locs, radius):
    cluster_locs = np.zeros(tree_locs.shape)
    r = radius * np.sqrt(np.random.uniform(0, 1, size=len(tree_locs)))
    theta = 2 * np.pi * np.random.uniform(0, 1, size=len(tree_locs))

    cluster_locs[:,0] = tree_locs[:,0] + r * np.cos(theta)
    cluster_locs[:,1] = tree_locs[:,1] + r * np.sin(theta)

    return cluster_locs