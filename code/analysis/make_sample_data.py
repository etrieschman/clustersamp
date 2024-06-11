# %%
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(parent_dir, 'code'))
from make_data import get_real_tree_data
from sample import SRS, PPSWR_SRS

# %%
n_primary_units = 500
radius_gps = 5
EMs = pd.DataFrame(columns=['index', 'EM'])

for gps_error_type in ['gaussian', 'uniform']:
    for radius_measurement in tqdm([10, 15, 20]):
        # readin data and get inclusion probabilities
        infile = f'{parent_dir}/data/raw/wt_kentland_data.csv'
        tree_locs, cluster_locs, tree_bm = get_real_tree_data(infile, gps_error_type, radius_gps)
        inc_probs = np.loadtxt(f'{parent_dir}/resources/{gps_error_type}/ip_rad{radius_measurement}.txt', delimiter=',')
        N = len(tree_bm)
        ENis = inc_probs.sum(axis=1)
        EM = ENis.sum()
        Zcoef = (EM / (N * ENis))
        EMs = pd.concat([EMs, pd.DataFrame({'index': [f'{gps_error_type}_rad{radius_measurement}'], 'EM': [EM]})], ignore_index=True)

        # sample data
        sample_params = {
            'sunits':tree_bm, 'sunit_locs':tree_locs, 
            'punit_weights':ENis/EM, 'punit_scales':Zcoef,
            'punit_locs':cluster_locs, 
            'gps_error_type':gps_error_type,
            'gps_error':radius_gps, 
            'measurement_rad':radius_measurement, 
            'replace':True, 
            'n_bootstraps':100
            }

        sampler = PPSWR_SRS(**sample_params)
        samples = sampler.sample(n_primary_units)

        # format output
        sunit_idxs = samples['sunit_idxs']
        punit_idxs = samples['punit_idxs']
        punit_ENis = ENis[punit_idxs]
        sunit_bm = [tree_bm[i] if len(i) > 0 else np.array([]) for i in sunit_idxs]
        # Find the maximum shape along each dimension
        max_sunits = np.max([len(i) for i in sunit_idxs])
        # Create an array of np.nan with the maximum shape and correct dtype
        sample_measurements = np.full((len(sunit_idxs), max_sunits), np.nan, dtype=tree_bm.dtype)
        for i, s in enumerate(sunit_bm):
            sample_measurements[i, :len(s)] = s

        df = pd.concat([
            pd.Series(punit_idxs, name='punit_idx'), pd.Series(punit_ENis, name='E[N]'),
            pd.DataFrame(sample_measurements, columns=[f'{i+1}' for i in range(max_sunits)])], axis=1)

        # write to file
        df.to_csv(f'{parent_dir}/results/analysis/make_sample_data/sample_data_{gps_error_type}_rad{radius_measurement}.csv', index=False, na_rep='')


EMs.to_csv(f'{parent_dir}/results/analysis/make_sample_data/EMs.csv', index=True, na_rep='')

# %%
