#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from generator import inhomogeneous_poisson_process, SpatioTemporalHawkesLam
from utils import plot_spatio_temporal_points, plot_spatial_intensity

if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(suppress=True)

    # lam = 10
    T   = (0, 1)
    S   = [(0, 1), (0, 1)]
    # print(generate_homogeneous_poisson_process(lam, T, S))
    lam    = SpatioTemporalHawkesLam(mu=1., alpha=3., beta=1., sigma=[1., 1.])
    points = inhomogeneous_poisson_process(lam, T, S)
    print(points)
    plot_spatial_intensity(lam, points, S, T,
        t_slots=1000, grid_size=50, interval=50)
