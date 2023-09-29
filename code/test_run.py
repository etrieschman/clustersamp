# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# GENERATE TREE LOCATIONS FROM A MIXTURE OF GAUSSIANS
# use mixture to simulate clustering/bunching of trees
def get_random_tree_locs(n_trees, n_groups, std_scale):
    # Define the parameters of the mixture
    n_groups = 10  # Number of Gaussian components
    n_trees = 100  # Number of data points to generate
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

tree_locs = get_random_tree_locs(100, 10, 0.5)

# %% 
# ASSIGN BIOMASS TO TREES
def get_tree_bm(tree_locs, gamma_shape, gamma_scale, loc_corr, loc_corr_noise_std):
    # get gamma distributed biomass
    tree_bm = np.random.gamma(shape=gamma_shape, scale=gamma_scale, size=len(tree_locs))
    # include correlation with lat/lon, and some noise
    tree_bm += (tree_locs[:,0]*loc_corr + 
                tree_locs[:,1]*loc_corr + 
                np.random.normal(scale=loc_corr_noise_std, size=len(tree_locs)))
    return tree_bm

tree_bm = get_tree_bm(tree_locs, 3, 2, 0.5, 0.025)

# %%
# RECORD NOISY TREE LOCATIONS (THESE BECOME OUR CLUSTER LOCATIONS)
def record_noisy_gps_locs(tree_locs, gps_noise):
    cluster_locs = np.zeros(tree_locs.shape)
    for i, tree in enumerate(tree_locs):
        cluster_locs[i] = np.random.multivariate_normal(mean=tree, cov=np.identity(2)*gps_noise)
    return cluster_locs

cluster_locs = record_noisy_gps_locs(tree_locs, 0.004)

# %%
# VISUALIZE
plt.scatter(x=tree_locs[:,0], y=tree_locs[:,1], alpha=0.75, s=tree_bm*8, label='true tree location (and size)')
plt.scatter(x=cluster_locs[:,0], y=cluster_locs[:,1], alpha=0.5, s=8, label='noisy GPS tree location')
plt.legend()
plt.title('Tree and cluster locations')
plt.show()

# %%
# 0. SAMPLE CLASS
#    - SAMPLE (is_location_noisy)
#    - GET MEAN ESTIMATOR
#    - GET MEAN ESTIMATOR SAMPLE VARIANCE (ANALYTICAL)
#    - GET MEAN ESTIMATOR SAMPLE VARIANCE (BOOTSTRAP)
#    A. SAMPLE TYPES
#       - FOR NOISY AND NON-NOISY LOCATIONS
#          - ONE-STAGE SRSWR
#          - ONE-STAGE PPSWR
#          - TWO-STAGE SRSWR-SRSWR
#          - TWO-STAGE PPSWR-SRSWR
#    B. SUB METHODS FOR PPSWR
#       - CALCULATE INCLUSION PROBABILITIES
#       - CALCULATE EXPECTED CLUSTER SIZE
# 1. COMPARISONS TO TRUE MEAN
#    - PLACEHOLDER