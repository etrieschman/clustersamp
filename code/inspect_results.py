# %%
# SETUP
import numpy as np
import pandas as pd
import scipy.stats as sps
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
ns = [25, 45, 65]
ns = [25, 45]

# helper functions
def get_mean_pvalue(results, nindex, true_mean):
    tstats = (results['means'][nindex] - true_mean) / np.sqrt(results['vars'][nindex])
    p_values_one_sided = sps.t.sf(abs(tstats), n)  # one-sided p-value
    p_values_two_sided = p_values_one_sided * 2 
    return f"{p_values_two_sided.mean():0.5}{'**' if p_values_two_sided.mean() < 0.05 else ''}"

for n in ns:
    nmin = 10
    nindex = n - nmin
    summ = pd.DataFrame()
    for gps_error_type in gps_error_types:
        tree_bm = np.loadtxt(PATH_DATA + f'processed/{gps_error_type}/tree_bm.txt', delimiter=',')
        for radius_measure in radius_measures:
            # get sample results
            try:
                inc_probs = np.loadtxt(PATH_RESOURCES + f'{gps_error_type}/ip_rad{radius_measure}.txt', delimiter=',')
                results_srs = np.load(PATH_RESULTS + f'{gps_error_type}/results_rad{radius_measure}_SRS.npz', allow_pickle=True)
                results_pps = np.load(PATH_RESULTS + f'{gps_error_type}/results_rad{radius_measure}_PPSWR-SRSWR.npz', allow_pickle=True)
            except:
                continue
            
            # get weighting terms
            EN = inc_probs.sum(0)
            EM = EN.sum()
            EF = inc_probs.sum(1)
            N = len(tree_bm)

            # Make summary table
            ttest = sps.ttest_1samp(results_pps['means'][nindex], tree_bm.mean())
            ttest = f"{ttest[1]:0.5}{'**' if ttest[1] < 0.05 else ''}"
            ttest2samp = sps.ttest_ind(results_pps['means'][nindex], results_srs['means'][nindex])
            ttest2samp = f"{ttest2samp[1]:0.5}{'**' if ttest2samp[1] < 0.05 else ''}"
            # estimator values
            summiter_pps = {
                'True mean':f'{tree_bm.mean(): 0.5f}',
                'Estimator mean':f"{results_pps['means'][nindex].mean():0.5f}",
                'Estimator variance':f"{results_pps['vars'][nindex].mean():0.5f}",
                'N':str(N),
                'n primary units':str(n),
                't-test of estimator mean (vs. True)':ttest,
                'Mean of repeat t-tests (vs. True)':get_mean_pvalue(results_pps, nindex, tree_bm.mean()),
                'Two-sample t-test (vs. SRS)':ttest2samp,
                'Mean of two-sample t-tests (vs. SRS)':None,
                'E[M]':f'{EM.copy():0.5f}',
                'Mean n secondary units':f"{results_pps['n_samples'][nindex].mean():0.5f}",
                '     per primary unit': f"{results_pps['n_samples'][nindex].mean()/n:0.5f}",
                '$P(i \in C_i)$':f"{inc_probs[0,0]:0.5f}",
            }

            index_pps = pd.MultiIndex.from_tuples(
                [(gps_error_type, radius_measure, 'PPSWR')],
                names=['GPS noise type', 'Measurement radius', 'Design'])
            summiter_pps = pd.DataFrame(summiter_pps, index=index_pps).T
            
            summ = pd.concat([summiter_pps, summ], axis=1)

            if radius_measure == radius_measures[-1]:
                ttest = sps.ttest_1samp(results_srs['means'][nindex], tree_bm.mean())
                ttest = f"{ttest[1]:0.5}{'**' if ttest[1] < 0.05 else ''}"
                summiter_srs = {
                    'True mean':f'{tree_bm.mean(): 0.5f}',
                    'Estimator':f"{results_srs['means'][nindex].mean():0.5f}",
                    'Estimator variance':f"{results_srs['vars'][nindex].mean():0.5f}",
                    'N':str(N),
                    'n primary units':str(n),
                    't-test of mean estimator (vs. True)':ttest,
                    'Mean of repeat t-tests (vs. True)':get_mean_pvalue(results_srs, nindex, tree_bm.mean())
                }
                index_srs = pd.MultiIndex.from_tuples(
                    [(gps_error_type, '', 'SRS')],
                    names=['GPS noise type', 'Measurement radius', 'Design'])

                summiter_srs = pd.DataFrame(summiter_srs, index=index_srs).T
                summ = pd.concat([summiter_srs, summ], axis=1)

            # # plot tstats
            # tstats = (results_pps['means'] - tree_bm.mean()) / np.sqrt(results_pps['vars'] / n)
            # p_values_one_sided = sps.t.sf(abs(tstats), n)  # one-sided p-value
            # p_values_two_sided = p_values_one_sided * 2 
            # plt.plot(p_values_two_sided.mean(1))
            # plt.title(f'N{n}_{gps_error_type}_rad{radius_measure}')
            # plt.show()

    summ = summ.fillna('-').round(5)
    summ.to_latex(buf=PATH_RESULTS + f'tab_results_summary_{n}.tex', 
                    index=True, bold_rows=True, index_names=False, 
                    column_format='l|r|rrr|r|rrr'
                    )
    display(summ)


# %%

# plt.yscale('log')

# %%
