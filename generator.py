#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Lam(object):
    '''An abstract class for defining the intensity of point process'''
    __metaclass__ = abc.ABCMeta

class HawkesLam(Lam):
    '''Intensity of Spatio-temporal Hawkes point process'''
    def __init__(self, mu, alpha, beta):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta

    def value(self, t, s):
        '''return the intensity value at (s, t)'''
        pass

    def upper_bound(self):
        '''return the upper bound of the intensity value'''
        pass

def homogeneous_poisson_process(lam, T, S):
    '''
    To generate a homogeneous Poisson point pattern in S ✕ T, this function uses
    a two step procedure:
    1. Simulate the number of events n = N(S ✕ T) occurring in S ✕ T according
       to a Poisson distribution with mean lam * |S| * |T|.
    2. Sample each of the n location and n times according to a uniform
       distribution on S and on T respectively.

    Args:
        lam: intensity (or maximum intensity when used by thining algorithm)
        T:   (t0, tn) indicates the range of time
        S:   [(min_x, max_x), (min_y, max_y), ...] indicates the range of
             coordinates regarding a square (or cubic ...) region.
    Returns:
        samples: point process samples [(t1, s1), (t2, s2), ..., (tn, sn)]
    '''

    # A helper function for calculating the Lebesgue measure for a space.
    # It actually is the length of an one-dimensional space, and the area of
    # a two-dimensional space.
    def lebesgue_measure(S):
        sub_lebesgue_ms = [ sub_space[1] - sub_space[0] for sub_space in S ]
        return np.prod(sub_lebesgue_ms)

    # calculate the number of events in S ✕ T
    n     = lebesgue_measure(S) * lebesgue_measure([T])
    N     = np.random.poisson(size=1, lam=lam * n)
    # simulate spatial sequence and temporal sequence separately.
    seq_t = [ np.random.uniform(T[0], T[1], N) ]
    seq_s = [ np.random.uniform(S[i][0], S[i][1], N) for i in range(len(S)) ]
    seq_t = np.array(seq_t).transpose()
    seq_s = np.array(seq_s).transpose()
    # concatenate spatial sequence and temporal sequence
    samples = np.concatenate([seq_t, seq_s], axis=1)
    # sort the sequence regarding the ascending order of the temporal sample.
    samples = samples[samples[:, 0].argsort()]
    return samples

def inhomogeneous_poisson_process(lam, T, S):
    '''
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
    '''

    # simulate a homogeneous Poisson process with intensity max_lam
    homo_samples     = homogeneous_poisson_process(lam.upper_bound(), T, S).tolist()
    retained_samples = [ point
        for point in homo_samples
        if lam.value(point[0], point[1:]) / lam.upper_bound() >= np.random.uniform(0, 1, 1)]
    return retained_samples

if __name__ == '__main__':
    lam = 1
    T   = (0, 10)
    S   = [(0, 1), (0, 1)]
    print(generate_homogeneous_poisson_process(lam, T, S))
