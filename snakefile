from snakemake.utils import min_version
min_version('6.0')

# --------------------------- Workflow constants --------------------------- #
configfile: "config.yaml"

PATH_DATA = 'data/'
PATH_RESOURCES = 'resources/'
PATH_RESULTS = 'results/'


# --------------------------- Rules --------------------------- #

rule all:
    input:
        expand(PATH_RESULTS + '{gps_error_type}/results_rad{radius_measure}_{sample_design}.npz',
               gps_error_type=config['gps_error_type'],
               radius_measure=config['radius_measure'],
               sample_design=config['sample_design'])

rule make_data:
    params:
        radius_gps = config['radius_gps'],
        take_subsample = config['take_subsample']
    input:
        raw_data = PATH_DATA + 'raw/wt_kentland_data.csv',
    output:
        tree_locs = PATH_DATA + 'processed/{gps_error_type}/tree_locs.txt',
        cluster_locs = PATH_DATA + 'processed/{gps_error_type}/cluster_locs.txt',
        tree_bm = PATH_DATA + 'processed/{gps_error_type}/tree_bm.txt',
        fig = PATH_RESULTS + '{gps_error_type}/fig_tree_locations.png'
    log:
        "logs/make_data_{gps_error_type}.log",
    script:
        'code/make_data.py'



rule get_inclusion_probability:
    params:
        radius_gps = config['radius_gps'],
        epsabs = config['ip_epsabs'],
        epsrel = config['ip_epsrel'],
        limit = config['ip_limit'],
    input: 
        tree_locs = PATH_DATA + 'processed/{gps_error_type}/tree_locs.txt',
        cluster_locs = PATH_DATA + 'processed/{gps_error_type}/cluster_locs.txt',
    output:
        inc_probs = PATH_RESOURCES + '{gps_error_type}/ip_rad{radius_measure}.txt',
    log:
        "logs/get_inclusion_probability_{gps_error_type}_rad{radius_measure}.log",
    script:
        "code/get_inclusion_probability.py"



rule simulate_sample_design:
    params:
        radius_gps = config['radius_gps'],
        bootstraps_for_variance = config['bootstraps_for_variance'],
        n_samples_min = config['n_samples_min'],
        n_samples_inc = config['n_samples_inc'],
        n_repeats = config['n_repeats'],
    input:
        tree_locs = PATH_DATA + 'processed/{gps_error_type}/tree_locs.txt',
        cluster_locs = PATH_DATA + 'processed/{gps_error_type}/cluster_locs.txt',
        tree_bm = PATH_DATA + 'processed/{gps_error_type}/tree_bm.txt',
        inc_probs = PATH_RESOURCES + '{gps_error_type}/ip_rad{radius_measure}.txt'
    output:
        results = PATH_RESULTS + '{gps_error_type}/results_rad{radius_measure}_{sample_design}.npz',
        fig = PATH_RESULTS + '{gps_error_type}/fig_rad{radius_measure}_{sample_design}.png'
    log:
        'logs/simulate_sample_design_{gps_error_type}_rad{radius_measure}_{sample_design}.log'
    script:
        'code/simulate_sample_design.py'

