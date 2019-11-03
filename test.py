#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from stppg import HawkesLam, SpatialTemporalPointProcess, \
    StdDiffusionKernel, GaussianDiffusionKernel, GaussianMixtureDiffusionKernel, \
    SpatialVariantGaussianDiffusionKernel, SpatialVariantGaussianMixtureDiffusionKernel
from utils import plot_spatio_temporal_points, plot_spatial_intensity, plot_spatial_kernel, DataAdapter

def test_std_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    standard diffusion kernel
    '''
    # parameters initialization
    mu     = .1
    kernel = StdDiffusionKernel(C=1., beta=1., sigma_x=.1, sigma_y=.1)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    # generate points
    points, sizes = pp.generate(
        T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
        batch_size=100, verbose=True)
    print(points)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('results/tf_thining_samples.npy')
    np.save('results/hpp_Feb_25.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)

def test_gaussian_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    Gaussian diffusion kernel
    '''
    mu     = .1
    kernel = GaussianDiffusionKernel(
        mu_x=0., mu_y=0., sigma_x=.1, sigma_y=.1, rho=0., beta=1., C=1.)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    # plot kernel parameters over the spatial region.
    # plot_spatial_kernel("results/gau.pdf", kernel, S=[[-1., 1.], [-1., 1.]], grid_size=50)

    # generate points
    points, sizes = pp.generate(
        T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
        batch_size=2, verbose=True)
    print(points)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('results/free_hpp_Mar_15_layer_5.npy')
    # np.save('results/gaussian_hpp_Mar_15_layer_5.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)

def test_gaussian_mixture_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    random Gaussian mixture diffusion kernel
    '''
    mu     = .2
    kernel = GaussianMixtureDiffusionKernel(
        n_comp=2, w=[0.5, 0.5], 
        mu_x=[0., 0.], mu_y=[0., 0.], 
        sigma_x=[1., 0.5], sigma_y=[0.5, 1.], 
        rho=[0., 0.], beta=1., C=1.)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

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

def test_spatial_variant_gaussian_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    Gaussian diffusion kernel
    '''
    mu     = .1
    kernel = SpatialVariantGaussianDiffusionKernel(
        f_mu_x=lambda x, y: 0., f_mu_y=lambda x, y: 0., 
        f_sigma_x=lambda x, y: (x + y) / 10 + .3, 
        f_sigma_y=lambda x, y: .3 - (x + y) / 10, 
        f_rho=lambda x, y: (x + y) / 4, 
        beta=1., C=1.)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    # plot kernel parameters over the spatial region.
    plot_spatial_kernel("results/kernel-svgau.pdf", kernel, S=[[-1., 1.], [-1., 1.]], grid_size=50)

    # generate points
    points, sizes = pp.generate(
        T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
        batch_size=1000, verbose=True)
    print(points)
    print(sizes)

    # read or save to local npy file.
    points = np.load('results/spatial-variant-gaussian.npy')
    np.save('results/spatial-variant-gaussian.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)

def test_spatial_variant_gaussian_mixture_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    Gaussian diffusion kernel
    '''
    mu     = .1
    kernel = SpatialVariantGaussianMixtureDiffusionKernel(
        n_comp=2, w=[0.5, 0.5],
        f_mu_x=[lambda x, y: 0., lambda x, y: 0.], f_mu_y=[lambda x, y: 0., lambda x, y: 0.], 
        f_sigma_x=[lambda x, y: (x + y) / 10 + .3, lambda x, y: .3 - (x + y) / 10], 
        f_sigma_y=[lambda x, y: .3 - (x + y) / 10, lambda x, y: (x + y) / 10 + .3],
        f_rho=[lambda x, y: (x + y) / 4, lambda x, y: - (x + y) / 4], 
        beta=1., C=1.)
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    # plot intensity of the process over the time
    test_point = np.array([
        [1., -1., -1.], 
        [2., -.75, -.75], 
        [3., -.5, -.5], 
        [4., -.25, -.25], 
        [5., 0., 0.], 
        [6., .25, .25], 
        [7., .5, .5],
        [8., .75, .75],
        [9., 1., 1.]])
    plot_spatial_intensity(lam, test_point, S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)
        


if __name__ == '__main__':
    # np.random.seed(1)
    np.set_printoptions(suppress=True)

    test_gaussian_mixture_diffusion()

    # T = [0., 10.]
    # S = [[-1., 1.], [-1., 1.]]

    # mu     = .1
    # # kernel = StdDiffusionKernel(C=1., beta=1., sigma_x=.08, sigma_y=.08)
    # kernel = GaussianMixtureDiffusionKernel(
    #     n_comp=5, layers=[5], 
    #     beta=1., C=1., SIGMA_SHIFT=.1, SIGMA_SCALE=.15, MU_SCALE=.1,
    #     Wss=None, bss=None, Wphis=None)
    # lam    = HawkesLam(mu, kernel, maximum=1e+3)
    # pp     = SpatialTemporalPointProcess(lam)

    # points = np.load('data/apd.crime.perday.npy')
    # print(len(np.nonzero(points[0, :, 0])[0]))
    # print(points[0])

    # plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
    #     t_slots=1000, grid_size=50, interval=100)

    # # t = np.array([10])
    # # s = np.array([1,1])
    # # his_t = np.array([1, 2, 3])
    # # his_s = np.array([[.1, .2] ,[.2, .3], [.3, .4]])
    # # kernel.nu(t, s, his_t, his_s)