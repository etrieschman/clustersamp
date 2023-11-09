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

def inclusion_prob(loc_i, loc_j, cov, R, interval):
    '''inclusion of tree j in cluster i'''
    result, _ = nquad(
        lambda xj, yj, r, theta: 
        r*normal_distribution(
            x=np.array([xj, yj, xj+r*np.cos(theta), yj+r*np.sin(theta)]),
            mean=np.concatenate([loc_j, loc_i]), 
            cov=cov),
        ranges=[
            (loc_j[0]-interval, loc_j[0]+interval),
            (loc_j[1]-interval, loc_j[1]+interval), 
            (0, R),
            (0, 2*np.pi)
        ],
        )
    return result


def get_inclusion_probs(locs, cov, R, interval=0.005, distance_threshold=100):
    N = len(locs)
    inc_probs = np.zeros((N, N))
    for i in tqdm(range(N), disable=False):
        for j in tqdm(range(N), disable=True):
            dist = np.linalg.norm(locs[i] - locs[j])
            if dist > distance_threshold:
                continue
            inc_probs[i,j] = inclusion_prob(locs[i], locs[j], cov, R, interval)
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


