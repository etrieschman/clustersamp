import numpy as np
from scipy.integrate import nquad
from tqdm import tqdm

# Define parameters
GPS_ERROR_VARIANCE = 5
MEASUREMENT_RADIUS = 0.1

# Define the indicator function
def indicator_function(xj, yj, xm, ym, R):
    return ((xj - xm)**2 + (yj - ym)**2 <= R**2)

def normal_distribution(x, mean, cov):
    k = len(x)
    coef = 1 / (2*np.pi*cov)**(k/2)
    exp = np.exp(-(1/ (2*cov)) * np.dot(x - mean, x - mean))
    return coef * exp

# def inclusion_prob_gaus(loc_i, loc_j, cov, R, interval):
#     '''inclusion of tree j in cluster i'''
#     result, _ = nquad(
#         lambda xj, yj, r, theta: 
#         r*normal_distribution(
#             x=np.array([xj, yj, xj+r*np.cos(theta), yj+r*np.sin(theta)]),
#             mean=np.concatenate([loc_j, loc_i]), 
#             cov=cov),
#         ranges=[
#             (loc_j[0]-interval, loc_j[0]+interval),
#             (loc_j[1]-interval, loc_j[1]+interval), 
#             (0, R),
#             (0, 2*np.pi)
#         ],
#         )
#     return result

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


def get_inclusion_probs(locs, radius_gps, radius_measure):
    N = len(locs)
    inc_probs_ur = np.zeros((N, N))
    # iterate through all pairs only once; skip when distance is too far
    for i in tqdm(range(N), disable=False):
        for j in tqdm(range(i, N), disable=True):
            dist = np.linalg.norm(locs[i] - locs[j])
            if dist > (radius_gps*2 + radius_measure):
                continue
            inc_probs_ur[i,j] = inclusion_prob_unif(locs[i], locs[j], radius_gps, radius_measure)
    # make symmetric
    inc_probs = inc_probs_ur + inc_probs_ur.T - np.diag(inc_probs_ur.diagonal())
    return inc_probs, inc_probs_ur


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


