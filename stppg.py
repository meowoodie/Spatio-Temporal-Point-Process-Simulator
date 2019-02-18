#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STPPG: Spatio-Temporal Point Process Generator

References:
- https://www.jstatsoft.org/article/view/v053i02
- https://www.ism.ac.jp/editsec/aism/pdf/044_1_0001.pdf

Dependencies:
- Python 3.6.7
"""

import sys
import utils
import arrow
import numpy as np

class DiffusionKernel(object):
    """
    Kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    """
    def __init__(self, beta=1., C=1., sigma_x = 1., sigma_y = 1.):
        self.beta    = beta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def nu(self, delta_t, delta_s):
        delta_x = delta_s[:, 0]
        delta_y = delta_s[:, 1]
        return np.exp(- self.beta * delta_t) * \
            (1. / (2 * np.pi * self.sigma_x * self.sigma_y * delta_t)) * \
            np.exp((- 1. / (2 * delta_t)) * \
                ((np.square(delta_x) / np.square(self.sigma_x)) + \
                (np.square(delta_y) / np.square(self.sigma_y))))
        # return np.exp(- self.beta * delta_t) * \
        #     (1. / (2 * np.pi * np.sqrt(self.sigma_x_2) * np.sqrt(self.sigma_y_2) * delta_t)) * \
        #     np.exp((- 1. / (2 * delta_t)) * \
        #         ((np.square(delta_x) / self.sigma_x_2) + \
        #         (np.square(delta_y) / self.sigma_y_2) - \
        #         (2. * np.sqrt(1 - np.square(delta_t)) * (delta_x - delta_y) / (np.sqrt(self.sigma_x_2) * np.sqrt(self.sigma_y_2))))) 

class HawkesLam(object):
    """Intensity of Spatio-temporal Hawkes point process"""
    def __init__(self, mu, kernel, maximum=1e+4):
        self.mu      = mu
        self.kernel  = kernel
        self.maximum = maximum

    def value(self, t, his_t, s, his_s):
        """
        return the intensity value at (t, s).
        The last element of seq_t and seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """
        if len(his_t) > 1:
            val = self.mu + np.sum(self.kernel.nu(t-his_t, s-his_s))
        else:
            val = self.mu
        return val

    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Hawkes processes"

class SpatialTemporalPointProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, lam):
        """
        Params:
        """
        # model parameters
        # self.mu      = tf.get_variable(name="mu", initializer=tf.random_uniform([n_nodes], 0, 1))
        # self.beta    = tf.get_variable(name="beta", initializer=tf.random_uniform([n_nodes, n_nodes], 0, 1))
        self.lam     = lam

    def _homogeneous_poisson_sampling(self, T=[0, 1], S=[[0, 1], [0, 1]]):
        """
        To generate a homogeneous Poisson point pattern in space S X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S X T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.

        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            S:   [(min_t, max_t), (min_x, max_x), (min_y, max_y), ...] indicates the
                range of coordinates regarding a square (or cubic ...) region.
        Returns:
            samples: point process samples:
            [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
        """
        _S     = [T] + S
        # sample the number of events from S
        n      = utils.lebesgue_measure(_S)
        N      = np.random.poisson(size=1, lam=self.lam.upper_bound() * n)
        # simulate spatial sequence and temporal sequence separately.
        points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
        points = np.array(points).transpose()
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points[:, 0].argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        print("[%s] generate %s samples from homogeneous poisson point process" % \
            (arrow.now(), homo_points.shape), file=sys.stderr)
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            t     = homo_points[i, 0]
            s     = homo_points[i, 1:]
            his_t = retained_points[:, 0]
            his_s = retained_points[:, 1:]
            # thinning
            lam_value = self.lam.value(t, his_t, s, his_s)
            lam_bar   = self.lam.upper_bound()
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("intensity %f is greater than upper bound %f." % (lam_value, lam_bar), file=sys.stderr)
                return None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, homo_points[[i], :]], axis=0)
            # monitor the process of the generation
            if i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]), file=sys.stderr)
        # log the final results of the thinning algorithm
        print("[%s] thining samples %s based on %s." % \
            (arrow.now(), retained_points.shape, self.lam), file=sys.stderr)
        return retained_points

    def generate(self, T=[0, 1], S=[[0, 1], [0, 1]], batch_size=10):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        while b < batch_size:
            print("[%s] generating %d-th sequence." % (arrow.now(), b), file=sys.stderr)
            homo_points = self._homogeneous_poisson_sampling(T, S)
            points      = self._inhomogeneous_poisson_thinning(homo_points)
            if points is None:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            b += 1
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len, 3))
        for b in range(batch_size):
            data[b, :points_list[b].shape[0]] = points_list[b]
        return data



if __name__ == "__main__":
    mu     = 1.
    kernel = DiffusionKernel()
    lam    = HawkesLam(mu, kernel, maximum=1e+6)
    pp     = SpatialTemporalPointProcess(lam)

    data = pp.generate(T=[0., 1.], S=[[-1., 1.], [-1., 1.]], batch_size=3)
    print(data)
    print(data.shape)