#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
MVPPG: MultiVariate Point Process Generator.
'''

import abc
import sys
import arrow
import numpy as np
# from stppg import homogeneous_poisson_process

class ExpKernel(object):
    '''
    Exponential Kernel function
    beta * exp(-beta * (t - t_i))
    '''
    def __init__(self, beta=1.):
        self.beta = beta

    def nu(self, delta_t):
        return self.beta * np.exp(-1 * self.beta * delta_t)

    def __str__(self):
        return 'Exponential Kernel'

class MultiVariateLam(object):
    '''Intensity of Multivariate Point Process'''
    def __init__(self, D, Mu, A, kernel, maximum=500.):
        self.D     = D
        self.Mu    = Mu
        self.A     = A
        self.maximum = maximum
        self.kernel  = kernel

    def value(self, seq_t, seq_d):
        '''
        return the intensity value at (d, t), where t means the time and d means
        the i-th component of the multivariate point process.
        The last elements of seq_d and seq_t are the location (t, d) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        '''
        # get current time, spatial values and historical time, spatial values.
        cur_t, his_t = seq_t[-1], seq_t[:-1]
        cur_d, his_d = seq_d[-1], seq_d[:-1]
        cur_d = cur_d.astype(np.int32)
        his_d = his_d.astype(np.int32)
        if len(seq_t) > 1:
            val = self.Mu[cur_d] + np.sum(self.A[cur_d, his_d] * self.kernel.nu(cur_t-his_t))
        else:
            val = self.Mu[cur_d]
        return val

    def upper_bound(self):
        '''return the upper bound of the intensity value'''
        return self.maximum

    def __str__(self):
        return '%dD Multivariate point process intensity Mu=%s and %s.' \
            % (self.D, self.Mu, self.kernel)

def inhomogeneous_multivariate_poisson_process(lam, D, T=(0, 1)):
    T = [T] # there is only one dimension (time) for each of the point process
            # since the spatio information has been represented in dfferent
            # variate (point process).
    # simulate D homogeneous Poisson process with intensity max_lam
    multi_points = [
        homogeneous_poisson_process(lam.upper_bound(), T).flatten()
        for d in range(D) ]
    retained_points     = []
    retained_components = []
    samples_size        = [ len(points) for points in multi_points ]
    print('[%s] generate samples %s (%d in total) from %dD homogeneous poisson point process' % \
          (arrow.now(), samples_size, sum(samples_size), D), file=sys.stderr)
    # reorganize the multi-points into a 1D array sorted by their time.
    points     = np.concatenate(multi_points)
    components = np.concatenate([
        np.array([ d for i in range(len(multi_points[d])) ]) # list of component index
        for d in range(D)])                                  # for each of component
    sorted_points     = points[points.argsort()]
    sorted_components = components[points.argsort()]
    # thining samples by acceptance rate.
    for idx in range(len(sorted_points) - 1):
        lam_value   = lam.value(sorted_points[:idx+1], sorted_components[:idx+1])
        lam_maximum = lam.upper_bound()
        accept_rate = np.random.uniform(0, 1, 1)
        # if acceptance rate is greater than 1, then raise exception
        # and upper bound of intensity should be raised accordingly.
        # assert lam_value / lam_maximum <= 1, \
        #        'intensity %f is greater than upper bound %f.' % (lam_value, lam_maximum)
        if lam_value / lam_maximum >= 1:
            break
        if lam_value / lam_maximum >= accept_rate:
            retained_points.append(sorted_points[idx])
            retained_components.append(sorted_components[idx])
        # show the process of the generation
        if idx % 1e+3 == 0 and idx != 0:
            print('[%s] %d raw samples have been checked. %d samples have been retained.' % \
                  (arrow.now(), idx, len(retained_points)), file=sys.stderr)
    retained_points     = np.array(retained_points)
    retained_components = np.array(retained_components)
    print('[%s] thining samples %s based on %s' % \
          (arrow.now(), len(retained_points), lam), file=sys.stderr)
    return retained_points, retained_components

if __name__ == '__main__':
    D      = 4
    T      = (0, 1)
    Mu     = np.array([.5, .1, .5, .1])
    A      = np.array([[1, 0, 1, 0], [2, 1, 2, 1], [1, 0, 1, 0], [2, 1, 2, 1]])
    kernel = ExpKernel(beta=.1)
    lam    = MultiVariateLam(D, Mu=Mu, A=A, kernel=kernel, maximum=20.)
    points, components = inhomogeneous_multivariate_poisson_process(lam, D, T)
    print(points)
    print(components)
