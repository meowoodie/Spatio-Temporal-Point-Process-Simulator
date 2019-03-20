#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from stppg import HawkesLam, SpatialTemporalPointProcess, StdDiffusionKernel, GaussianDiffusionKernel, GaussianMixtureDiffusionKernel
from utils import plot_spatio_temporal_points, plot_spatial_intensity, plot_spatial_kernel

def test_std_diffusion():
    '''Test Spatio-Temporal Point Process Generator equipped with standard diffusion kernel'''
    # parameters initialization
    mu     = .1
    kernel = StdDiffusionKernel(C=1., beta=1., sigma_x=.1, sigma_y=.1)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    # generate points
    points, sizes = pp.generate(
        T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
        batch_size=500, verbose=True)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('results/tf_thining_samples.npy')
    np.save('results/hpp_Feb_25.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)

def test_gaussian_diffusion():
    '''Test Spatio-Temporal Point Process Generator equipped with Gaussian diffusion kernel'''
    mu     = .4
    kernel = GaussianDiffusionKernel(
        layers=[5, 5], C=1., beta=1., 
        SIGMA_SHIFT=.2, SIGMA_SCALE=.05, MU_SCALE=.1, is_centered=True)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)
    print(kernel.Ws)
    print(kernel.bs)

    # plot kernel parameters over the spatial region.
    plot_spatial_kernel("results/kernel.pdf", kernel, S=[[-1., 1.], [-1., 1.]], grid_size=50)

    # generate points
    points, sizes = pp.generate(
        T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
        batch_size=2, verbose=True)
    print(points.shape)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('results/free_hpp_Mar_15_layer_5.npy')
    # np.save('results/gaussian_hpp_Mar_15_layer_5.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[1], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)

def test_gaussian_mixture_diffusion():
    '''Test Spatio-Temporal Point Process Generator equipped with Gaussian mixture diffusion kernel'''
    mu     = .2
    kernel = GaussianMixtureDiffusionKernel(
        n_comp=5, layers=[5, 5], C=1., beta=2., 
        SIGMA_SHIFT=.2, SIGMA_SCALE=.05, MU_SCALE=.1)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

#     # plot kernel parameters over the spatial region.
#     plot_spatial_kernel("results/kernel.pdf", kernel, S=[[-1., 1.], [-1., 1.]], grid_size=50)

    # generate points
    points, sizes = pp.generate(
        T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
        batch_size=2, verbose=True)
    print(points.shape)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('results/free_hpp_Mar_15_layer_5.npy')
    # np.save('results/gaussian_hpp_Mar_15_layer_5.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)


if __name__ == '__main__':
    np.random.seed(11)
    np.set_printoptions(suppress=True)

    test_gaussian_mixture_diffusion()
