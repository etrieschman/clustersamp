import numpy as np
from scipy.spatial import cKDTree

from make_data import record_noisy_gps_locs

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
    '''Benchmark: every tree is tagged, so we can sample trees directly.'''
    def __init__(self, units, n_bootstraps):
        super().__init__(units, n_bootstraps)

    def sample(self, n):
        self.n = n
        self.unit_idx_samples = np.random.choice(self.N, size=n, replace=True)
        self.get_stats()
        return {'unit_idxs': self.unit_idx_samples, 'n_measured': n}

    def _calculate_sample_mean(self):
        self.sample_mean = self.units[self.unit_idx_samples].mean()

    def _calculate_sample_variance(self):
        self.sample_mean_variance = self.units[self.unit_idx_samples].var(ddof=1) / self.n

    def _calculate_sample_variance_boot(self):
        boot_idx = np.random.randint(self.n, size=(self.n_bootstraps, self.n))
        self.sample_mean_variance_boot = self.units[self.unit_idx_samples][boot_idx].mean(1).var()


class PPSWR_SRS(Sample):
    '''
    The 2024 design. Pins are selected with probability proportional to expected
    plot size and each plot contributes its scaled mean. Kept as the baseline:
    its expectation is the average over pins of the expected plot mean, which is
    not the population mean when tree size is correlated with local density.
    '''
    def __init__(self, sunits, sunit_locs, punit_weights, punit_scales, punit_locs,
                 gps_error_type, gps_error, measurement_rad, replace, n_bootstraps):
        super().__init__(sunits, n_bootstraps)
        self.punit_weights = punit_weights # weights for pps
        self.punit_scales = punit_scales # what to scale each secondary unit mean by
        self.punit_locs = punit_locs # noisy gps locations
        self.sunit_locs = sunit_locs # true tree locations
        self.sunit_tree = cKDTree(sunit_locs)
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
        punit_measurement_locs = record_noisy_gps_locs(
            self.punit_locs[self.punit_idx_samples], self.gps_error_type, self.gps_error)

        # loop first-stage units to get second stage units
        self.sunit_idx_samples = []
        for samples in self.sunit_tree.query_ball_point(punit_measurement_locs, self.measurement_rad):
            samples = np.array(samples, dtype=int)
            if len(samples) > 0:
                samples = np.random.choice(samples, size=len(samples), replace=self.replace)
            self.sunit_idx_samples += [samples]
        self.get_stats()
        return {
            'punit_idxs': self.punit_idx_samples,
            'sunit_idxs': self.sunit_idx_samples,
            'n_measured': sum(len(idxs) for idxs in self.sunit_idx_samples)}

    def _calculate_sample_mean(self):
        self.cluster_mean = np.array(
            [self.units[idxs].mean() if len(idxs) > 0 else 0 for idxs in self.sunit_idx_samples])
        self.sample_mean = (self.punit_scales[self.punit_idx_samples]*self.cluster_mean).mean()

    def _calculate_sample_variance(self):
        self.sample_mean_variance = (self.punit_scales[self.punit_idx_samples]*self.cluster_mean).var(ddof=1) / self.k

    def _calculate_sample_variance_boot(self):
        # resample trees within each cluster, then resample clusters
        # (clusters are padded to a common length so this runs without a loop)
        cluster_size = np.array([len(idxs) for idxs in self.sunit_idx_samples])
        max_size = max(cluster_size.max(), 1)
        cluster_units = np.zeros((self.k, max_size))
        for i, idxs in enumerate(self.sunit_idx_samples):
            cluster_units[i, :len(idxs)] = self.units[idxs]
        pos = np.floor(np.random.uniform(size=(self.n_bootstraps, self.k, max_size))
                       * cluster_size[np.newaxis, :, np.newaxis]).astype(int)
        in_cluster = np.arange(max_size)[np.newaxis, np.newaxis, :] < cluster_size[np.newaxis, :, np.newaxis]
        boot_units = np.take_along_axis(
            np.broadcast_to(cluster_units, pos.shape), pos, axis=2) * in_cluster
        bootstrap_cluster_means = boot_units.sum(2) / np.maximum(cluster_size, 1)
        means_idx = np.random.randint(self.k, size=(self.n_bootstraps, self.k))
        bootstrap_means = (
            self.punit_scales[self.punit_idx_samples][means_idx] *
            np.take_along_axis(bootstrap_cluster_means, means_idx, axis=1)).mean(1)
        self.sample_mean_variance_boot = bootstrap_means.var()


def measure_plots(plot_locs, punit_locs_sampled, sunit_locs, sunit_tree, punit_tree, measurement_rad,
                  offset_error=0., bearing_error=0., inclusion_rule='recorded'):
    '''
    The field work at each visited pin, for the pin-count design.

    The surveyor stands at plot_locs (the pin plus navigation error) and records
    each nearby tree's offset from where they stand (distance and bearing). A
    tree's apparent location is the pin plus its recorded offset, and its pin
    count is the number of pins in the frame within the measurement radius of
    that apparent location.

    offset_error: std (m) of additive error in each coordinate of the recorded offset
    bearing_error: std (degrees) of error in the recorded bearing
    inclusion_rule: 'recorded' keeps a tree if its recorded offset is inside the
        measurement radius (the rule the estimator needs); 'true' keeps it if its
        true distance is, which is inconsistent with the pin count when offsets
        are recorded with error

    Returns one entry per (plot, tree) measurement: plot index, tree index, pin count.
    '''
    k = len(plot_locs)
    search_rad = measurement_rad * (1 + 5 * np.deg2rad(bearing_error)) + 5 * offset_error
    neighbors = sunit_tree.query_ball_point(plot_locs, search_rad)
    plot_idx = np.repeat(np.arange(k), [len(idxs) for idxs in neighbors])
    sunit_idx = np.array([idx for idxs in neighbors for idx in idxs], dtype=int)
    if len(sunit_idx) == 0:
        return plot_idx, sunit_idx, np.array([])

    offsets = sunit_locs[sunit_idx] - plot_locs[plot_idx]
    offsets_recorded = offsets.copy()
    if bearing_error > 0:
        theta = np.deg2rad(np.random.normal(scale=bearing_error, size=len(offsets)))
        offsets_recorded = np.array([
            np.cos(theta)*offsets[:,0] - np.sin(theta)*offsets[:,1],
            np.sin(theta)*offsets[:,0] + np.cos(theta)*offsets[:,1]]).T
    if offset_error > 0:
        offsets_recorded = offsets_recorded + np.random.normal(scale=offset_error, size=offsets.shape)

    if inclusion_rule == 'recorded':
        in_plot = np.linalg.norm(offsets_recorded, axis=1) < measurement_rad
    else:
        in_plot = np.linalg.norm(offsets, axis=1) < measurement_rad
    plot_idx, sunit_idx, offsets_recorded = plot_idx[in_plot], sunit_idx[in_plot], offsets_recorded[in_plot]

    apparent_locs = punit_locs_sampled[plot_idx] + offsets_recorded
    pin_counts = punit_tree.query_ball_point(apparent_locs, measurement_rad, return_length=True)
    # under the 'recorded' rule the visited pin is always counted, so pin_counts >= 1
    pin_counts = np.maximum(pin_counts, 1).astype(float)
    return plot_idx, sunit_idx, pin_counts


def choose_one_per_plot(plot_idx, pin_counts):
    '''Choose one tree in each plot with probability proportional to 1 / pin count.
    Returns positions into the measurement arrays.'''
    if len(plot_idx) == 0:
        return np.array([], dtype=int)
    key = -np.log(pin_counts) + np.random.gumbel(size=len(pin_counts))
    order = np.lexsort((key, plot_idx))  # by plot, then key; the last of each plot wins
    last = np.r_[plot_idx[order][1:] != plot_idx[order][:-1], True]
    return order[last]


class PinCount(Sample):
    '''
    The pin-count design. Pins are selected with equal probability. Every tree in
    a plot is weighted by one over its pin count (see measure_plots), which is the
    tree's multiplicity in the frame given the navigation error.

    ratio: False gives the Hansen-Hurwitz form, mean of the plot totals of y / count.
        True divides by the plot totals of 1 / count (times the known survival rate),
        which needs the number of live trees to be known.
    subsample: None measures biomass on every tree in the plot. 1 locates every
        tree but measures biomass on one, chosen with probability proportional to
        1 / count.
    '''
    def __init__(self, sunits, sunit_locs, punit_locs, gps_error_type, gps_error,
                 measurement_rad, ratio, subsample, n_bootstraps,
                 offset_error=0., bearing_error=0., inclusion_rule='recorded', survival=1.):
        super().__init__(sunits, n_bootstraps)
        self.N = len(punit_locs) # the frame is the pins, one per planted tree
        self.true_mean = sunits.sum() / self.N
        self.punit_locs = punit_locs # noisy gps locations
        self.sunit_locs = sunit_locs # true tree locations
        self.punit_tree = cKDTree(punit_locs)
        self.sunit_tree = cKDTree(sunit_locs)
        self.gps_error_type = gps_error_type
        self.gps_error = gps_error
        self.measurement_rad = measurement_rad
        self.ratio = ratio
        self.subsample = subsample
        self.offset_error = offset_error
        self.bearing_error = bearing_error
        self.inclusion_rule = inclusion_rule
        self.survival = survival

        self.punit_idx_samples = None
        self.sunit_idx_samples = None
        self.cluster_total = None # per plot: sum of y / pin count
        self.cluster_weight = None # per plot: sum of 1 / pin count

    def sample(self, k):
        self.k = k
        # FIRST STAGE (srswr) -- get primary unit indices and walk to them
        self.punit_idx_samples = np.random.choice(self.N, size=self.k, replace=True)
        punit_locs_sampled = self.punit_locs[self.punit_idx_samples]
        punit_measurement_locs = record_noisy_gps_locs(
            punit_locs_sampled, self.gps_error_type, self.gps_error)

        # SECOND STAGE -- locate every tree in each plot and count its pins
        plot_idx, sunit_idx, pin_counts = measure_plots(
            punit_measurement_locs, punit_locs_sampled, self.sunit_locs,
            self.sunit_tree, self.punit_tree, self.measurement_rad,
            self.offset_error, self.bearing_error, self.inclusion_rule)
        self.cluster_size = np.bincount(plot_idx, minlength=self.k)
        self.sunit_idx_samples = np.split(sunit_idx, np.cumsum(self.cluster_size)[:-1])
        self.cluster_weight = np.bincount(plot_idx, weights=1/pin_counts, minlength=self.k)
        if self.subsample is None:
            self.cluster_total = np.bincount(
                plot_idx, weights=self.units[sunit_idx]/pin_counts, minlength=self.k)
            n_measured = len(sunit_idx)
        else:
            chosen = choose_one_per_plot(plot_idx, pin_counts)
            self.cluster_total = np.zeros(self.k)
            self.cluster_total[plot_idx[chosen]] = (
                self.cluster_weight[plot_idx[chosen]] * self.units[sunit_idx[chosen]])
            n_measured = len(chosen)
        self.get_stats()
        return {
            'punit_idxs': self.punit_idx_samples,
            'sunit_idxs': self.sunit_idx_samples,
            'n_measured': n_measured}

    def _estimate(self, totals, weights):
        # totals, weights: (..., k) arrays of plot totals
        if not self.ratio:
            return totals.mean(-1)
        return self.survival * totals.sum(-1) / np.maximum(weights.sum(-1), 1e-12)

    def _calculate_sample_mean(self):
        self.sample_mean = self._estimate(self.cluster_total, self.cluster_weight)

    def _calculate_sample_variance(self):
        if not self.ratio:
            self.sample_mean_variance = self.cluster_total.var(ddof=1) / self.k
        else:
            # linearization of the ratio
            ratio = self.sample_mean / self.survival
            resid = self.cluster_total - ratio * self.cluster_weight
            self.sample_mean_variance = (
                self.survival**2 * (resid**2).sum()
                / (self.k * (self.k - 1) * max(self.cluster_weight.mean(), 1e-12)**2))

    def _calculate_sample_variance_boot(self):
        # resample plots
        means_idx = np.random.randint(self.k, size=(self.n_bootstraps, self.k))
        bootstrap_means = self._estimate(self.cluster_total[means_idx], self.cluster_weight[means_idx])
        self.sample_mean_variance_boot = bootstrap_means.var()
