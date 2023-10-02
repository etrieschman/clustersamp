import numpy as np

from inclusion_prob import get_distance_matrix
from make_data import record_noisy_gps_locs

class Sample:
    def __init__(self, units:np.array, n_bootstraps=None):
        self.units = units
        self.true_mean = units.mean()
        self.n_bootstraps = 200 if n_bootstraps is None else n_bootstraps
        self.N = len(units)
        self.unit_idx_samples = None
        self.n = None
        self.sample_mean = None
        self.sample_mean_variance = None
        self.sample_mean_variance_boot = None

    def sample(self, n):
        pass

    def get_stats(self):
        self._calculate_sample_mean()
        self._calculate_sample_variance()
        self._calculate_sample_variance_boot()
        stats = {
            'true_mean':self.true_mean,
            'sample_mean':self.sample_mean,
            'sample_mean_var': self.sample_mean_variance,
            'sample_mean_var_boot': self.sample_mean_variance_boot
        }
        return stats
    
    def _calculate_sample_mean(self):
        pass
    def _calculate_sample_variance(self):
        pass
    def _calculate_sample_variance_boot(self):
        pass


class SRS(Sample):
    def __init__(self, units, n_bootstraps=None):
        super().__init__(units, n_bootstraps)
    
    def sample(self, n):
        self.n = n
        self.unit_idx_samples = np.random.choice(self.N, size=n, replace=True)
        self.get_stats()
        return self.unit_idx_samples

    def _calculate_sample_mean(self):
        self.sample_mean = self.units[self.unit_idx_samples].mean()

    def _calculate_sample_variance(self):
        self.sample_mean_variance = self.units[self.unit_idx_samples].var() / self.n

    def _calculate_sample_variance_boot(self):
        self.sample_mean_variance_boot = np.array(
            [np.random.choice(self.units[self.unit_idx_samples], size=self.n, replace=True).mean() 
             for i in range(self.n_bootstraps)]
             ).var() 
        
class PPSWR_SRS(Sample):
    def __init__(self, sunits_scaled, sunit_locs, punit_weights, punit_locs, 
                 gps_error_var, measurement_rad, replace=True, n_bootstraps=None):
        super().__init__(sunits_scaled, n_bootstraps)
        self.punit_weights = punit_weights # weights for pps
        self.punit_locs = punit_locs # noisy gps locations
        self.sunit_locs = sunit_locs # true tree locations
        self.gps_error_var = gps_error_var
        self.measurement_rad = measurement_rad
        self.replace = replace

        self.punit_idx_samples = None
        self.sunit_idx_samples = None
        self.cluster_size = None
        self.cluster_mean = None
    
    def sample(self, k):
        self.k = k
        # FIRST STAGE (ppswr) -- get primary unit indices
        self.punit_idx_samples = np.random.choice(self.N, size=self.k, replace=True, p=self.punit_weights)
        punit_measurement_locs = record_noisy_gps_locs(self.punit_locs[self.punit_idx_samples], self.gps_error_var)
        dist = get_distance_matrix(punit_measurement_locs, self.sunit_locs)
        # SECOND STAGE (SRSWOR / SRSWR)
        self.sunit_in_punit = (dist < self.measurement_rad) # inclusion mask
        cluster_size = self.sunit_in_punit.sum(1)
        cluster_prob = self.sunit_in_punit / cluster_size.reshape((-1,1))
        self.sunit_idx_samples = []
        for i in range(k):
            self.sunit_idx_samples += [np.random.choice(self.N, size=cluster_size[i], replace=self.replace, p=cluster_prob[i])]
        self.get_stats()
        return self.punit_idx_samples, self.sunit_idx_samples

    def _calculate_sample_mean(self):
        self.cluster_mean = np.array([self.units[idxs].mean() for idxs in self.sunit_idx_samples])
        self.sample_mean = self.cluster_mean.mean()

    def _calculate_sample_variance(self):
        self.sample_mean_variance = self.cluster_mean.var() / self.k

    def _calculate_sample_variance_boot(self):
        bootstrap_means = np.zeros(self.n_bootstraps)
        for i in range(self.n_bootstraps):
            # recalculate mean
            bootstrap_cluster_means = np.array(
                [np.random.choice(self.units[idxs], size=len(idxs), replace=True).mean()
                 for idxs in self.sunit_idx_samples])
            bootstrap_means[i] = bootstrap_cluster_means.mean()
        self.sample_mean_variance_boot = bootstrap_means.var()
        
