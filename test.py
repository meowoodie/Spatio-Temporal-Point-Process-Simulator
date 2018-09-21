#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from stppg import inhomogeneous_poisson_process, SpatioTemporalHawkesLam, DiffusionKernel
from utils import plot_spatio_temporal_points, plot_spatial_intensity

if __name__ == '__main__':
    # np.random.seed(0)
    np.set_printoptions(suppress=True)

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
