#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TPPG: Temporal Point Process Generator

Dependencies:
- Python 3.6.7
"""

import sys
import utils
import arrow
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

class ExpKernel(object):
    """
    Kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    """
    def __init__(self, beta=1.):
        self.beta    = beta

    def nu(self, t, his_t):
        delta_t = t - his_t
        return self.beta * np.exp(- self.beta * delta_t)



class HawkesLam(object):
    """Intensity of Spatio-temporal Hawkes point process"""
    def __init__(self, mu, kernel, maximum=1e+4):
        self.mu      = mu
        self.kernel  = kernel
        self.maximum = maximum

    def value(self, t, his_t):
        """
        return the intensity value at t.
        The last element of seq_t is the location t that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """
        if len(his_t) > 0:
            val = self.mu + np.sum(self.kernel.nu(t, his_t))
        else:
            val = self.mu
        return val

    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Hawkes processes"

class TemporalPointProcess(object):
    """
    Marked Temporal Hawkes Process

    A stochastic temporal points generator based on Hawkes process.
    """

    def __init__(self, lam):
        """
        Params:
        """
        # model parameters
        self.lam     = lam

    def _homogeneous_poisson_sampling(self, T=[0, 1]):
        """
        To generate a homogeneous Poisson point pattern in space T, it basically
        takes two steps:
        1. Simulate the number of events n = N(T) occurring in T according to a
        Poisson distribution with mean lam * |T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.

        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            T:   (min_t, max_t) indicates the range of coordinates.
        Returns:
            samples: point process samples:
            [t1, t2, ..., tn]
        """
        # sample the number of events from S
        n      = T[1] - T[0]
        N      = np.random.poisson(size=1, lam=self.lam.upper_bound() * n)
        # simulate spatial sequence and temporal sequence separately.
        points = np.random.uniform(T[0], T[1], N)
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points.argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(t):
        1. Define an upper bound max_lam for the intensity function lam(t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(t)/max_lam for each point (t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0))
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape), file=sys.stderr)
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            t     = homo_points[i]
            his_t = retained_points
            # thinning
            lam_value = self.lam.value(t, his_t)
            lam_bar   = self.lam.upper_bound()
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                if verbose:
                    print("intensity %f is greater than upper bound %f." % (lam_value, lam_bar), file=sys.stderr)
                return None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, homo_points[[i]]], axis=0)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]), file=sys.stderr)
        # log the final results of the thinning algorithm
        if verbose:
            print("[%s] thining samples %s based on %s." % \
                (arrow.now(), retained_points.shape, self.lam), file=sys.stderr)
        return retained_points

    def generate(self, T=[0, 1], batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        pbar = tqdm(total = batch_size, desc="Generating ...")
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling(T)
            points      = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            pbar.update(1)
            b += 1
            if verbose:
                print("[%s] %d-th sequence is generated." % (arrow.now(), b+1), file=sys.stderr)
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len))
        for b in range(batch_size):
            data[b, :len(points_list[b])] = points_list[b]
        return data, sizes



def plot_1d_pointprocess(points, lam, T, ngrid=100):
    """
    visualize 1 dimensional point process
    """
    ts      = np.linspace(T[0], T[1], ngrid)
    lamvals = []
    for t in ts:
        his_t  = points[(points <= t) * (points > 0)]
        lamval = lam.value(t, his_t)
        lamvals.append(lamval)
    
    evals   = []
    print(points)
    for t in points:
        his_t  = points[(points <= t) * (points > 0)]
        lamval = lam.value(t, his_t)
        evals.append(lamval)

    fig, ax = plt.subplots()
    ax.plot(ts, lamvals, linestyle="--", color="grey")
    ax.scatter(points, evals, c="r")
    plt.show()
    
    

if __name__ == "__main__":

    # np.random.seed(0)
    # np.set_printoptions(suppress=True)

    # parameters initialization
    mu     = 10
    T      = [0., 1.]
    beta   = 100.
    kernel = ExpKernel(beta=beta)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = TemporalPointProcess(lam)

    # generate points
    points, sizes = pp.generate(
        T=T, batch_size=1000, verbose=False)

    np.save("data-1d-mu%.1f-beta%.1f-points.npy" % (mu, beta), points)
    np.save("data-1d-mu%.1f-beta%.1f-sizes.npy" % (mu, beta), sizes)

    print(sizes)

    plot_1d_pointprocess(points[0], lam, T=T, ngrid=1000)
