#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import sys
import arrow
import numpy as np

class Lam(object):
    '''An abstract class for defining the intensity of point process'''
    __metaclass__ = abc.ABCMeta

class SpatioTemporalHawkesLam(Lam):
    '''Intensity of Spatio-temporal Hawkes point process'''
    def __init__(self, mu, alpha, beta, sigma, maximum=500.):
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta
        self.sigma = sigma # same length as the dimension of the S
        self.maximum = maximum

    def value(self, seq_t, seq_s):
        '''
        return the intensity value at (t, s).
        The last element of seq_t and seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        '''
        # kernel function (clustering density)
        # t is a scalar or a vector.
        def nu(t, s, C=1.):
            return (C/(2*np.pi*np.prod(self.sigma)*t)) * \
                   np.exp(-1*self.beta*t - np.sum((np.power(s, 2) * 1/np.power(self.sigma, 2)), axis=1) / (2*t))
        if len(seq_t) > 0:
            # get current time, spatial values and historical time, spatial values.
            cur_t, his_t = seq_t[-1], seq_t[:-1]
            cur_s, his_s = seq_s[-1], seq_s[:-1]
            val = self.mu + np.sum(nu(cur_t-his_t, cur_s-his_s))
        else:
            val = self.mu
        return val

    def upper_bound(self, ):
        '''return the upper bound of the intensity value'''
        return self.maximum

    def __str__(self):
        return 'Spatio-temporal Hawkes point process intensity'

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
    points          = homogeneous_poisson_process(lam.upper_bound(), T, S)
    retained_points = []
    print('[%s] generate samples %s from homogeneous poisson point process' % \
          (arrow.now(), points.shape), file=sys.stderr)
    # thining samples by acceptance rate.
    for idx in range(len(points)-1):
        lam_value   = lam.value(points[:idx+2][:, 0], points[:idx+2][:, 1:])
        lam_maximum = lam.upper_bound()
        accept_rate = np.random.uniform(0, 1, 1)
        # if acceptance rate is greater than 1, then raise exceptionself.
        # and upper bound of intensity should be raised accordingly.
        assert lam_value / lam_maximum <= 1, \
               'intensity %f is greater than upper bound %f.' % (lam_value, lam_maximum)
        if lam_value / lam_maximum >= accept_rate:
            retained_points.append(points[idx])
    retained_points = np.array(retained_points)
    print('[%s] thining samples %s based on %s' % \
          (arrow.now(), retained_points.shape, lam), file=sys.stderr)
    return retained_points

if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(suppress=True)

    # lam = 10
    T   = (0, 10)
    S   = [(0, 1), (0, 1)]
    # print(generate_homogeneous_poisson_process(lam, T, S))
    lam = SpatioTemporalHawkesLam(mu=5., alpha=1., beta=1., sigma=[1., 2.])
    print(inhomogeneous_poisson_process(lam, T, S))
