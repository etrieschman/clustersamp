# %%
# Model inclusion probabilities, P(j in C_i), between pin i and pin j.
# These are only needed for the 2024 design (PPSWR-SRSWR), which selects pins
# with probability proportional to expected plot size. The pin-count design
# does not use them: its weights are counted from the pin map.
import numpy as np
from scipy.stats import ncx2
from tqdm import tqdm

from make_data import record_noisy_gps_locs_unif
from utils import mock_snakemake

# INCLUSION PROBABILITIES ASSUMING GAUSSIAN ERRORS
# tree j sits at pin j plus one error, the surveyor stands at pin i plus another,
# so their separation is N(pin j - pin i, 2 sigma^2 I) and its squared length
# over 2 sigma^2 is noncentral chi-square with 2 degrees of freedom.
# (This replaces the 4-d quadrature, whose polar integrand was missing its
# Jacobian and returned P(i in C_i) = 0.40 where it should be about 1.)
def inclusion_prob_gaus(dist, radius_gps, radius_measure):
    return ncx2.cdf(radius_measure**2 / (2 * radius_gps**2), df=2, nc=dist**2 / (2 * radius_gps**2))

# INCLUSION PROBABILITIES ASSUMING UNIFORM ERRORS
# no closed form; simulate the difference of two uniform-disc errors once and
# tabulate P(|dist + e_j - e_m| < radius_measure) on a grid of pin distances
def inclusion_prob_unif(dist, radius_gps, radius_measure, n_draws=200000, n_grid=400):
    zeros = np.zeros((n_draws, 2))
    diff = record_noisy_gps_locs_unif(zeros, radius_gps) - record_noisy_gps_locs_unif(zeros, radius_gps)
    grid = np.linspace(0, radius_measure + 2*radius_gps, n_grid)
    probs = np.array([(np.hypot(g + diff[:,0], diff[:,1]) < radius_measure).mean() for g in tqdm(grid)])
    return np.interp(dist, grid, probs, right=0.)


def get_inclusion_probs(inclusion_prob_method, locs, radius_gps, radius_measure):
    # row is secondary unit, col is primary unit; symmetric by construction
    dist = get_distance_matrix(locs, locs)
    return inclusion_prob_method(dist, radius_gps, radius_measure)


def get_distance_matrix(locs1, locs2):
    return np.linalg.norm(locs1[:, np.newaxis, :] - locs2[np.newaxis, :, :], axis=2)


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('get_inclusion_probability',
                                   population='kentland',
                                   gps_error_type='gaussian',
                                   radius_measure=10)

    radius_measure = int(snakemake.wildcards.radius_measure)
    gps_error_type = snakemake.wildcards.gps_error_type
    gps_error = float(snakemake.params.gps_error[gps_error_type])
    np.random.seed(int(snakemake.params.seed))
    cluster_locs = np.loadtxt(snakemake.input.cluster_locs, delimiter=',')

    if gps_error_type == 'gaussian':
        inclusion_probs = get_inclusion_probs(
            inclusion_prob_gaus, cluster_locs, gps_error, radius_measure)
    elif gps_error_type == 'uniform':
        inclusion_probs = get_inclusion_probs(
            inclusion_prob_unif, cluster_locs, gps_error, radius_measure)

    np.savetxt(snakemake.output.inc_probs, inclusion_probs, delimiter=',')

# %%
