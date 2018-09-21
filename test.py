#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from stppg import inhomogeneous_poisson_process, SpatioTemporalHawkesLam, DiffusionKernel
from mvppg import ExpKernel, MultiVariateLam, inhomogeneous_multivariate_poisson_process
from utils import plot_spatio_temporal_points, plot_spatial_intensity, plot_multivariate_intensity, GaussianInfluentialMatrixSimulator, multi2spatial

def test_stppg():
    '''Test Spatio-Temporal Point Process Generator'''
    # define time and spatial space
    S = [(0, 1), (0, 1), (0, 1)]

    # define kernel function and intensity function
    kernel = DiffusionKernel(beta=1., C=1., sigma=[1., 1.])
    lam    = SpatioTemporalHawkesLam(mu=.1, alpha=.1, beta=1., kernel=kernel, maximum=1e+4)
    points = inhomogeneous_poisson_process(lam, S)
    print(points)

    # read or save to local txt file.
    # points = np.loadtxt('hpp_sept_20.txt', delimiter=',')
    # np.savetxt('hpp_sept_20.txt', points, delimiter=',')

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points, S,
        t_slots=1000, grid_size=50, interval=50)

def test_mvppg():
    d      = 20
    cov    = [[.1, 0.], [0., .1]]
    beta   = 1e-5
    D      = d * d
    T      = (0, 1)
    Mu     = np.zeros(D)
    ims = GaussianInfluentialMatrixSimulator(
        length=1., grid_size=[d, d], mu=[0., 0.], cov=cov)
    A      = ims.A
    kernel = ExpKernel(beta=beta)
    lam    = MultiVariateLam(D, Mu=Mu, A=A, kernel=kernel, maximum=100.)
    ts, ds = inhomogeneous_multivariate_poisson_process(lam, D, T)
    points = multi2spatial(ts, ds, ims)
    # plot intensity of the process over the time
    plot_multivariate_intensity(lam, points, S=[T, (0, 1), (0, 1)],
        t_slots=1000, grid_size=d, interval=50)

if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(suppress=True)

    # test_stppg()
    test_mvppg()
