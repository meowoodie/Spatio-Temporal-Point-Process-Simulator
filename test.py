#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from stppg import HawkesLam, SpatialTemporalPointProcess, StdDiffusionKernel, CustomizedDiffusionKernel, GaussianDiffusionKernel, GaussianMixtureDiffusionKernel
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
    print(points)
    print(sizes)

    # read or save to local npy file.
    # points = np.load('results/free_hpp_Mar_15_layer_5.npy')
    # np.save('results/gaussian_hpp_Mar_15_layer_5.npy', points)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)

def test_random_gaussian_mixture_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    random Gaussian mixture diffusion kernel
    '''
    mu     = .2
    kernel = GaussianMixtureDiffusionKernel(
        n_comp=5, layers=[5, 5], C=1., beta=1., 
        SIGMA_SHIFT=.2, SIGMA_SCALE=.05, MU_SCALE=.05)
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

def test_pretrain_gaussian_mixture_diffusion():
    '''
    Test Spatio-Temporal Point Process Generator equipped with 
    pretrained Gaussian mixture diffusion kernel
    '''
    params = np.load('data/ambulance_mle_gaussian_mixture_params.npz')
    mu     = params['mu']
    beta   = params['beta']
    kernel = GaussianMixtureDiffusionKernel(
        n_comp=5, layers=[5], C=1., beta=beta, 
        SIGMA_SHIFT=.05, SIGMA_SCALE=.2, MU_SCALE=.01,
        Wss=params['Wss'], bss=params['bss'], Wphis=params['Wphis'])
    lam    = HawkesLam(mu, kernel, maximum=1e+3)
    pp     = SpatialTemporalPointProcess(lam)

    # # generate points
    # points, sizes = pp.generate(
    #     T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
    #     batch_size=2, verbose=True)
    # print(points.shape)
    # print(sizes)

    # read or save to local npy file.
    points = np.load('data/ambulance.perday.npy')
    da     = DataAdapter(init_data=points)
    points = da.normalize(points)
    # np.save('results/gaussian_hpp_Mar_15_layer_5.npy', points)
    print(points[0].shape)

    # plot intensity of the process over the time
    plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
        t_slots=1000, grid_size=50, interval=50)


if __name__ == '__main__':
    # np.random.seed(1)
    np.set_printoptions(suppress=True)

    test_std_diffusion()

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