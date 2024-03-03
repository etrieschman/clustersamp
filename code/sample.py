import numpy as np

from get_inclusion_probability import get_distance_matrix
from make_data import record_noisy_gps_locs_gaus, record_noisy_gps_locs_unif

class Sample:
    def __init__(self, units:np.array, n_bootstraps):
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
    def __init__(self, units, n_bootstraps):
        super().__init__(units, n_bootstraps)
    
    def sample(self, n):
        self.n = n
        self.unit_idx_samples = np.random.choice(self.N, size=n, replace=True)
        self.get_stats()
        return {'unit_idxs': self.unit_idx_samples}

    def _calculate_sample_mean(self):
        self.sample_mean = self.units[self.unit_idx_samples].mean()

    def _calculate_sample_variance(self):
        self.sample_mean_variance = self.units[self.unit_idx_samples].var() / self.n

    def _calculate_sample_variance_boot(self):
        self.sample_mean_variance_boot = np.array(
            [np.random.choice(self.units[self.unit_idx_samples], 
                              size=self.n, replace=True).mean() 
             for i in range(self.n_bootstraps)]
             ).var() 
        
class PPSWR_SRS(Sample):
    def __init__(self, sunits, sunit_locs, punit_weights, punit_scales, punit_locs, 
                 gps_error_type, gps_error, measurement_rad, replace, n_bootstraps):
        super().__init__(sunits, n_bootstraps)
        self.punit_weights = punit_weights # weights for pps
        self.punit_scales = punit_scales # what to scale each secondary unit mean by
        self.punit_locs = punit_locs # noisy gps locations
        self.sunit_locs = sunit_locs # true tree locations
        self.gps_error_type = gps_error_type
        self.gps_error = gps_error
        self.measurement_rad = measurement_rad
        self.replace = replace

        self.punit_idx_samples = None
        self.sunit_idx_samples = None
        self.cluster_size = None
        self.cluster_mean = None
    
    def sample(self, k):
        self.k = k
        # FIRST STAGE (ppswr) -- get primary unit indices
        self.punit_idx_samples = np.random.choice(
            self.N, size=self.k, replace=True, p=self.punit_weights)
        if self.gps_error_type == 'gausian':
            punit_measurement_locs = record_noisy_gps_locs_gaus(
                self.punit_locs[self.punit_idx_samples], self.gps_error)
        else:
            punit_measurement_locs = record_noisy_gps_locs_unif(
                self.punit_locs[self.punit_idx_samples], self.gps_error)

        # loop first-stage units to get second stage units
        self.sunit_idx_samples = []
        for i in range(self.k):
            dist = get_distance_matrix(punit_measurement_locs[[i]], self.sunit_locs)
            if (dist >= self.measurement_rad).all():
                self.sunit_idx_samples += [np.array([])]
            else:
                samples = np.where(dist < self.measurement_rad)[1]
                self.sunit_idx_samples += [
                    np.random.choice(samples, size=len(samples), replace=self.replace)]
        self.get_stats()
        return {
            'punit_idxs': self.punit_idx_samples, 
            'sunit_idxs': self.sunit_idx_samples}

    def _calculate_sample_mean(self):
        self.cluster_mean = np.array(
            [self.units[idxs].mean() if len(idxs) > 0 else 0 for idxs in self.sunit_idx_samples])
        self.sample_mean = (self.punit_scales[self.punit_idx_samples]*self.cluster_mean).mean()

    def _calculate_sample_variance(self):
        self.sample_mean_variance = (
            self.punit_scales[self.punit_idx_samples]*self.cluster_mean).var() / self.k

    def _calculate_sample_variance_boot(self):
        bootstrap_means = np.zeros(self.n_bootstraps)
        for i in range(self.n_bootstraps):
            # recalculate mean
            bootstrap_cluster_means = np.array(
                [np.random.choice(self.units[idxs], size=len(idxs), replace=True).mean()
                 if (len(idxs) != 0) else 0 for idxs in self.sunit_idx_samples])
            means_idx = np.random.choice(np.arange(len(bootstrap_cluster_means)),
                                         size=self.k, replace=True)
            bootstrap_means[i] = (self.punit_scales[means_idx]*bootstrap_cluster_means[means_idx]).mean()
        self.sample_mean_variance_boot = bootstrap_means.var()
        
