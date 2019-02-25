#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from stppg import DiffusionKernel, HawkesLam, SpatialTemporalPointProcess
from mvppg import ExpKernel, MultiVariateLam, inhomogeneous_multivariate_poisson_process
from utils import plot_spatio_temporal_points, plot_spatial_intensity, plot_multivariate_intensity, GaussianInfluentialMatrixSimulator, multi2spatial

def test_stppg():
    '''Test Spatio-Temporal Point Process Generator'''
    mu     = .1
    kernel = DiffusionKernel(beta=1., C=1., sigma_x = .1, sigma_y = .1)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    points, sizes = pp.generate(T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
                         batch_size=100, upper_len=10, verbose=False)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('hpp_Feb_18.npy')
    np.save('results/hpp_Feb_25.npy', points)

    # # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0, :, :], S=[[0., 10.], [-1., 1.], [-1., 1.]],
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

    test_stppg()

    # points = np.array([
    #     [ 0.36997303, -0.05792566,  0.68023894],
    #     [ 0.39260438,  0.97531302, -0.96479512],
    #     [ 0.53247704,  0.1610365,   0.79696197],
    #     [ 0.62975171,  0.42994964,  0.72846005],
    #     [ 0.64687623, -0.07139027,  0.87994606],
    #     [ 0.79895546, -0.036986,   -0.84665047],
    #     [ 0.85706017, -0.7623167,   0.02126256],
    #     [ 0.93949473, -0.72242577,  0.1817626 ],
    #     [ 0.97328026, -0.88536129,  0.91589099]
    # ])
    # S = [(0, 1.), (-1., 1.), (-1., 1.)]

    # kernel = DiffusionKernel(beta=1., C=1., sigma=[1., 1.])
    # lam    = SpatioTemporalHawkesLam(mu=1., alpha=1., beta=1., kernel=kernel, maximum=1e+6)

    # plot_spatial_intensity(lam, points, S,
    #     t_slots=1000, grid_size=50, interval=50)
