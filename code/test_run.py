# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# MAKE DATA
n_groupings = 10
n_trees_near_in_grouping_mean = 10

grouping_centroids = np.random.normal(size=n_groupings*2).reshape((n_groupings, 2))
# %%
