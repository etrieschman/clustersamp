# %%
import numpy as np
from scipy.integrate import nquad
from tqdm import tqdm

from utils import mock_snakemake

# INCLUSION PROBABILITIES ASSUMING GAUSSIAN ERRORS
def normal_distribution_pdf(x, mean, cov):
    k = len(x)
    coef = 1 / (2*np.pi*cov)**(k/2)
    exp = np.exp(-(1/ (2*cov)) * np.dot(x - mean, x - mean))
    return coef * exp

def integrand_gaus(x_j, y_j, r_m, theta_m, loc_j, loc_i, radius_gps):
    prob = normal_distribution_pdf(
        x=np.array([x_j, y_j, 
                    r_m*np.cos(theta_m) + x_j, 
                    r_m*np.sin(theta_m) + y_j]),
        mean=np.concatenate([loc_j, loc_i]),
        cov=radius_gps
    )
    return prob

def inclusion_prob_gaus(loc_j, loc_i, radius_gps, radius_measure):
    max_dist = (2*radius_gps+radius_measure) # limit +/-infty to a max distance
    result, __ = nquad(
        integrand_gaus,
        args=(loc_j, loc_i, radius_gps),
        ranges=[
            (loc_j[0]-max_dist, loc_j[0]+max_dist),
            (loc_j[1]-max_dist, loc_j[1]+max_dist),
            (0, radius_measure),
            (0, 2*np.pi)],
        # opts={'epsabs': 1.49e-08, 'epsrel': 1.49e-08, 'limit':50}
        opts={'epsabs': 1e-01, 'epsrel':5e-01, 'limit':4}
        )
    return result 

# INCLUSION PROBABILITIES ASSUMING UNIFORM ERRORS
def integrand_unif(r_j, theta_j, r_m, theta_m, loc_j, loc_i, radius_gps):
    '''integrand for inclusion probabilities, assuming uniform distribution'''
    x_diff = r_m*np.cos(theta_m) + r_j*np.cos(theta_j) + loc_j[0] - loc_i[0]
    y_diff = r_m*np.sin(theta_m) + r_j*np.sin(theta_j) + loc_j[1] - loc_i[1]
    indicator = x_diff**2 + y_diff**2 <= radius_gps**2
    return r_j * r_m * indicator

def inclusion_prob_unif(loc_j, loc_i, radius_gps, radius_measure):
    scale = 1 / (np.pi * radius_gps**2)**2
    result, __ = nquad(
        integrand_unif,
        args=(loc_j, loc_i, radius_gps),
        ranges=[
            (0, radius_gps),
            (0, 2*np.pi), 
            (0, radius_measure),
            (0, 2*np.pi)],
        # opts={'epsabs': 1.49e-08, 'epsrel': 1.49e-08, 'limit':50}
        opts={'epsabs': 1e-01/scale, 'epsrel':5e-01, 'limit':4}
        )
    return result * scale


def get_inclusion_probs(inclusion_prob_method, locs, radius_gps, radius_measure):
    N = len(locs)
    inc_probs = np.zeros((N, N))
    # iterate through all pairs only once; skip when distance is too far
    for i in tqdm(range(N), disable=False):
        for j in tqdm(range(i, N), disable=True):
            dist = np.linalg.norm(locs[i] - locs[j])
            if dist > (radius_gps*2 + radius_measure):
                continue
            inc_probs[i,j] = inclusion_prob_method(locs[i], locs[j], radius_gps, radius_measure)
    # make symmetric
    inc_probs = inc_probs + inc_probs.T - np.diag(inc_probs.diagonal())
    return inc_probs


def get_distance_matrix(locs1, locs2):
    N1, N2 = len(locs1), len(locs2)
    dist = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            dist[i,j] = np.linalg.norm(locs1[i] - locs2[j])
    return dist


def get_rough_inclusion_probs(locs, eps=1e-1):
    inc_probs = get_distance_matrix(locs, locs)
    inc_probs[inc_probs == 0] = eps
    inc_probs = 1 / inc_probs
    return inc_probs


if __name__ == '__main__':
    if "snakemake" not in globals():
        snakemake = mock_snakemake('get_inclusion_probability')
    
    radius_gps = int(snakemake.params.radius_gps)
    radius_measure = int(snakemake.wildcards.radius_measure)
    gps_error_type = snakemake.wildcards.gps_error_type
    tree_locs = np.loadtxt(snakemake.input.tree_locs, delimiter=',')
    cluster_locs = np.loadtxt(snakemake.input.cluster_locs, delimiter=',')


    if gps_error_type == 'gaussian':
        inclusion_probs = get_inclusion_probs(
            inclusion_prob_gaus, cluster_locs, radius_gps, radius_measure)
    elif gps_error_type == 'uniform':
        inclusion_probs = get_inclusion_probs(
            inclusion_prob_unif, cluster_locs, radius_gps, radius_measure)
    elif gps_error_type == 'rough':
        inclusion_probs = get_rough_inclusion_probs(cluster_locs)

    
    np.savetxt(snakemake.output.inc_probs, inclusion_probs, delimiter=',')
    
# %%
