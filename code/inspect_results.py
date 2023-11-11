# %%
# SETUP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH_RESULTS = '../results/'
PATH_RESOURCES = '../resources/'
PATH_DATA = '../data/'

# set conditions
radius_gps = 5
radius_measures = [15, 10, 5]
sample_designs = ['PPSWR-SRSWR', 'SRS']
gps_error_types = ['uniform', 'gaussian']

summ = pd.DataFrame()

for gps_error_type in gps_error_types:
    for radius_measure in radius_measures:
        for sample_design in sample_designs:

            # readin data
            try:
                results = np.load(PATH_RESULTS + f'{gps_error_type}/results_rad{radius_measure}_{sample_design}.npz')
                inc_probs = np.loadtxt(PATH_RESOURCES + f'{gps_error_type}/inclusion_probability_rad{radius_measure}.txt', delimiter=',')
                tree_bm = np.loadtxt(PATH_DATA + f'processed/{gps_error_type}/tree_bm.txt', delimiter=',')[:len(inc_probs)]
            except:
                continue

            # get weighting terms
            EN = inc_probs.sum(0)
            EM = EN.sum()
            EF = inc_probs.sum(1)
            N = len(tree_bm)
            Z = EM / (N*EF) * tree_bm

            # Make summary table
            nmin = 10
            n = 50
            nindex = n - nmin
            round = 4

            summiter = {
                'True mean':tree_bm.mean(),
                'Estimator':results['means'][nindex].mean(),
                'Estimator variance':results['vars'][nindex].mean(),
                'N':N,
                'E[M]': EM.copy(),
                'n primary':n,
                'n secondary mean':results['n_samples'][nindex].mean(),
                'n secondary variance':results['n_samples'][nindex].var(),
                
                '$P(i \in i)$':inc_probs[0,0]
            }
            index = pd.MultiIndex.from_tuples(
                [(gps_error_type, radius_measure, sample_design)],
                names=['GPS noise type', 'Measurement radius', 'Design'])
            summiter = pd.DataFrame(summiter, index=index).T
            summ = pd.concat([summiter, summ], axis=1)

summ.to_latex(buf=PATH_RESULTS + 'tab_results_summary.tex', 
                 index=True, bold_rows=True, index_names=False, 
                #  column_format='ll|c|ccc'
                 )
summ


# %%
