# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.spatial import cKDTree
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
# the recorded phone locations are the pins (primary units); the "true" tree
# locations are the pins perturbed by one draw of GPS error
def get_real_tree_data(infile, gps_error_type, gps_error):
    # readin dataset
    treedf = pd.read_csv(infile, header=0)
    cluster_loc_lat = treedf.lat.values
    cluster_loc_lon = treedf.lon.values
    tree_bm = treedf.biomass_TCO2e.values

    # convert to equal-area
    cluster_loc_x, cluster_loc_y = latlon_to_utm(cluster_loc_lat, cluster_loc_lon)
    cluster_locs = np.array([cluster_loc_x, cluster_loc_y]).T
    tree_locs = record_noisy_gps_locs(cluster_locs, gps_error_type, gps_error)

    return tree_locs, cluster_locs, tree_bm


# GENERATE SYNTHETIC TREE LOCATIONS
# several small stands with different planting densities, so there is a lot of
# stand edge and local density varies at the scale of a measurement plot
def get_random_tree_locs(n_stands=5, stand_size=(40, 90), area_per_tree=(25, 60), min_spacing=2):
    stand_corners = np.array([(0, 0), (150, 20), (40, 160), (220, 180), (300, 60)])[:n_stands]
    tree_locs = []
    for corner in stand_corners:
        width, height = np.random.uniform(*stand_size, size=2)
        n_trees = int(width * height / np.random.uniform(*area_per_tree))
        tree_locs += [corner + np.random.uniform(size=(n_trees, 2)) * [width, height]]
    tree_locs = np.vstack(tree_locs)

    # drop one tree of every pair closer than the minimum spacing
    keep = np.ones(len(tree_locs), dtype=bool)
    for i, j in sorted(cKDTree(tree_locs).query_pairs(min_spacing)):
        if keep[i] and keep[j]:
            keep[j] = False
    return tree_locs[keep]

# ASSIGN BIOMASS TO TREES
# biomass falls with the number of neighbors (competition), so tree size is
# correlated with local density; this is the case the 2024 design gets wrong
def get_random_tree_bm(tree_locs, gamma_shape=6, mean_bm=0.5, competition=0.35, competition_rad=8):
    tree_bm = np.random.gamma(shape=gamma_shape, scale=mean_bm/gamma_shape, size=len(tree_locs))
    n_neighbors = cKDTree(tree_locs).query_ball_point(tree_locs, competition_rad, return_length=True) - 1
    n_neighbors_std = (n_neighbors - n_neighbors.mean()) / n_neighbors.std()
    return tree_bm * np.exp(-competition * n_neighbors_std)

def get_synthetic_tree_data(gps_error_type, gps_error):
    tree_locs = get_random_tree_locs()
    tree_bm = get_random_tree_bm(tree_locs)
    # every tree gets one pin at planting
    cluster_locs = record_noisy_gps_locs(tree_locs, gps_error_type, gps_error)
    return tree_locs, cluster_locs, tree_bm

# PATCHY MORTALITY
# dead trees leave no trace in the field; their pins stay in the frame
def get_patchy_mortality(tree_locs, n_patches=12, patch_rad=15, max_mortality=0.7):
    lo, hi = tree_locs.min(0), tree_locs.max(0)
    patch_locs = np.random.uniform(lo, hi, size=(n_patches, 2))
    dist = np.linalg.norm(tree_locs[:, np.newaxis] - patch_locs[np.newaxis], axis=2).min(1)
    alive = np.random.uniform(size=len(tree_locs)) > max_mortality * np.exp(-(dist / patch_rad)**2)
    return alive


# RECORD NOISY LOCATIONS (THESE BECOME OUR CLUSTER AND DEPLOYED MEASUREMENT LOCATIONS)
# gaussian: gps_noise is the standard deviation of each coordinate (m)
def record_noisy_gps_locs_gaus(tree_locs, gps_noise):
    return tree_locs + np.random.normal(scale=gps_noise, size=tree_locs.shape)

# uniform: radius is the radius of the error disc (m)
def record_noisy_gps_locs_unif(tree_locs, radius):
    cluster_locs = np.zeros(tree_locs.shape)
    r = radius * np.sqrt(np.random.uniform(0, 1, size=len(tree_locs)))
    theta = 2 * np.pi * np.random.uniform(0, 1, size=len(tree_locs))

    cluster_locs[:,0] = tree_locs[:,0] + r * np.cos(theta)
    cluster_locs[:,1] = tree_locs[:,1] + r * np.sin(theta)

    return cluster_locs

def record_noisy_gps_locs(tree_locs, gps_error_type, gps_error):
    if gps_error_type == 'gaussian':
        return record_noisy_gps_locs_gaus(tree_locs, gps_error)
    elif gps_error_type == 'uniform':
        return record_noisy_gps_locs_unif(tree_locs, gps_error)
    raise ValueError(f'unknown gps_error_type: {gps_error_type}')

def plot_trees(tree_locs, cluster_locs, tree_bm, outfile=None):
    plt.figure(figsize=(6,6))
    plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.5, s=tree_bm*100, label='true tree location (and size)')
    plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='[primary unit] noisy GPS pin')
    plt.legend()
    plt.axis('equal')
    plt.xlabel('easting')
    plt.ylabel('northing')
    plt.title('Pin and tree locations')
    if outfile is not None:
        plt.savefig(outfile, dpi=300)


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('make_data',
                                   population='synthetic',
                                   gps_error_type='gaussian')

    take_subsample = snakemake.params.take_subsample
    population = snakemake.wildcards.population
    gps_error_type = snakemake.wildcards.gps_error_type
    gps_error = float(snakemake.params.gps_error[gps_error_type])
    infile = snakemake.input.raw_data
    np.random.seed(int(snakemake.params.seed))

    # get data
    if population == 'kentland':
        tree_locs, cluster_locs, tree_bm = get_real_tree_data(infile, gps_error_type, gps_error)
    elif population == 'synthetic':
        tree_locs, cluster_locs, tree_bm = get_synthetic_tree_data(gps_error_type, gps_error)

    if take_subsample != 'false':
        tree_locs, cluster_locs, tree_bm = tree_locs[:int(take_subsample)], cluster_locs[:int(take_subsample)], tree_bm[:int(take_subsample)]

    # save to file
    np.savetxt(snakemake.output.tree_locs, tree_locs, delimiter=',')
    np.savetxt(snakemake.output.cluster_locs, cluster_locs, delimiter=',')
    np.savetxt(snakemake.output.tree_bm, tree_bm, delimiter=',')

    # plot trees
    plot_trees(tree_locs, cluster_locs, tree_bm, snakemake.output.fig)
# %%
