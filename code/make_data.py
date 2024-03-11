# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from utils import mock_snakemake

PATH_DATA = 'data/'

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
def get_real_tree_data(infile, gps_error_type, radius_gps):
    # readin dataset
    treedf = pd.read_csv(infile, header=0)
    cluster_loc_lat = treedf.lat.values
    cluster_loc_lon = treedf.lon.values
    tree_bm = treedf.biomass_TCO2e.values

    # convert to equal-area
    cluster_loc_x, cluster_loc_y = latlon_to_utm(cluster_loc_lat, cluster_loc_lon)
    cluster_locs = np.array([cluster_loc_x, cluster_loc_y]).T
    if gps_error_type == 'gaussian':
        tree_locs = record_noisy_gps_locs_gaus(cluster_locs, radius_gps)
    else:
        tree_locs = record_noisy_gps_locs_unif(cluster_locs, radius_gps)

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

def plot_trees(tree_locs, cluster_locs, outfile=None):
    plt.figure(figsize=(6,6))
    plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.5, s=tree_bm*100, label='[secondary unit] true tree location (and size)')
    plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='[primary unit] noisy GPS tree location')
    plt.legend()
    plt.xlabel('easting')
    plt.ylabel('northing')
    plt.title('Primary and secondary unit locations')
    if outfile is not None:
        plt.savefig(outfile, dpi=300)


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('make_data',
                                   gps_error_type='gaussian',
                                   radius_measure=15,
                                   sample_design='PPSWR-SRSWR')
    
    take_subsample = snakemake.params.take_subsample
    radius_gps = int(snakemake.params.radius_gps)
    gps_error_type = snakemake.wildcards.gps_error_type
    infile = snakemake.input.raw_data

    # get data
    tree_locs, cluster_locs, tree_bm = get_real_tree_data(infile, gps_error_type, radius_gps)

    if take_subsample != 'false':
        tree_locs, cluster_locs, tree_bm = tree_locs[:int(take_subsample)], cluster_locs[:int(take_subsample)], tree_bm[:int(take_subsample)]

    # save to file
    np.savetxt(snakemake.output.tree_locs, tree_locs, delimiter=',')
    np.savetxt(snakemake.output.cluster_locs, cluster_locs, delimiter=',')
    np.savetxt(snakemake.output.tree_bm, tree_bm, delimiter=',')

    # plot trees
    plot_trees(tree_locs, cluster_locs, snakemake.output.fig)
# %%
