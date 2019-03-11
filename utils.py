#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import sys
import arrow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal

def lebesgue_measure(S):
    """
    A helper function for calculating the Lebesgue measure for a space.
    It actually is the length of an one-dimensional space, and the area of
    a two-dimensional space.
    """
    sub_lebesgue_ms = [ sub_space[1] - sub_space[0] for sub_space in S ]
    return np.prod(sub_lebesgue_ms)

def plot_spatial_kernel(path, kernel, S, grid_size, 
        sigma_x_clim=[0.1, 0.35], sigma_y_clim=[0.1, 0.35], rho_clim=[-1, 1]):
    """
    Plot spatial kernel parameters over the spatial region, including 
    sigma_x, sigma_x, and rho. 
    """
    assert len(S) == 2, '%d is an invalid dimension of the space.' % len(S)
    # define the span for space region
    x_span = np.linspace(S[0][0], S[0][1], grid_size+1)[:-1]
    y_span = np.linspace(S[1][0], S[1][1], grid_size+1)[:-1]
    # map initialization
    sigma_x_map = np.zeros((grid_size, grid_size))
    sigma_y_map = np.zeros((grid_size, grid_size))
    rho_map     = np.zeros((grid_size, grid_size))
    # grid entris calculation
    for x_idx in range(grid_size):
        for y_idx in range(grid_size):
            s                         = [x_span[x_idx], y_span[y_idx]]
            sigma_x, sigma_y, rho     = kernel.nonlinear_mapping(s)
            sigma_x_map[x_idx][y_idx] = sigma_x
            sigma_y_map[x_idx][y_idx] = sigma_y
            rho_map[x_idx][y_idx]     = rho
    # plotting
    plt.rc('text', usetex=True)
    plt.rc("font", family="serif")
    # plot as a pdf file
    with PdfPages(path) as pdf:
        fig, axs = plt.subplots(1, 3)
        cmap     = matplotlib.cm.get_cmap('viridis')
        im_0     = axs[0].imshow(sigma_x_map, interpolation='nearest', origin='lower', cmap=cmap)
        im_1     = axs[1].imshow(sigma_y_map, interpolation='nearest', origin='lower', cmap=cmap)
        im_2     = axs[2].imshow(rho_map, interpolation='nearest', origin='lower', cmap=cmap)
        print(sigma_x_map.min(), sigma_x_map.max())
        print(sigma_y_map.min(), sigma_y_map.max())
        print(rho_map.min(), rho_map.max())
        # ticks for colorbars
        im_0.set_clim(*sigma_x_clim)
        im_1.set_clim(*sigma_y_clim)
        im_2.set_clim(*rho_clim)
        tick_0 = np.linspace(sigma_x_clim[0], sigma_x_clim[1], 5).tolist()
        tick_1 = np.linspace(sigma_y_clim[0], sigma_y_clim[1], 5).tolist()
        tick_2 = np.linspace(rho_clim[0], rho_clim[1], 5).tolist()
        # set x, y labels for subplots
        axs[0].set_xlabel(r'$x$')
        axs[1].set_xlabel(r'$x$')
        axs[2].set_xlabel(r'$x$')
        axs[0].set_ylabel(r'$y$')
        axs[1].set_ylabel(r'$y$')
        axs[2].set_ylabel(r'$y$')
        # remove x, y ticks
        axs[0].get_xaxis().set_ticks([])
        axs[1].get_xaxis().set_ticks([])
        axs[2].get_xaxis().set_ticks([])
        axs[0].get_yaxis().set_ticks([])
        axs[1].get_yaxis().set_ticks([])
        axs[2].get_yaxis().set_ticks([])
        # set subtitle for subplots
        axs[0].set_title(r'$\sigma_x$')
        axs[1].set_title(r'$\sigma_y$')
        axs[2].set_title(r'$\rho$')
        # plot colorbar
        cbar_0 = fig.colorbar(im_0, ax=axs[0], ticks=tick_0, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar_1 = fig.colorbar(im_1, ax=axs[1], ticks=tick_1, fraction=0.046, pad=0.08, orientation="horizontal")
        cbar_2 = fig.colorbar(im_2, ax=axs[2], ticks=tick_2, fraction=0.046, pad=0.08, orientation="horizontal")
        # set font size of the ticks of the colorbars
        cbar_0.ax.tick_params(labelsize=5) 
        cbar_1.ax.tick_params(labelsize=5) 
        cbar_2.ax.tick_params(labelsize=5) 
        # adjust the width of the gap between subplots
        plt.subplots_adjust(wspace=0.2)
        pdf.savefig(fig)

def plot_spatial_intensity(lam, points, S, t_slots, grid_size, interval):
    """
    Plot spatial intensity as the time goes by. The generated points can be also
    plotted on the same 2D space optionally.
    """
    assert len(S) == 3, '%d is an invalid dimension of the space.' % len(S)
    # split points into sequence of time and space.
    seq_t, seq_s = points[:, 0], points[:, 1:]
    # define the span for each subspace
    t_span = np.linspace(S[0][0], S[0][1], t_slots+1)[1:]
    x_span = np.linspace(S[1][0], S[1][1], grid_size+1)[:-1]
    y_span = np.linspace(S[2][0], S[2][1], grid_size+1)[:-1]
    # function for yielding the heatmap over the entire region at a given time
    def heatmap(t):
        _map      = np.zeros((grid_size, grid_size))
        sub_seq_t = seq_t[seq_t < t]
        sub_seq_s = seq_s[:len(sub_seq_t)]
        for x_idx in range(grid_size):
            for y_idx in range(grid_size):
                s = [x_span[x_idx], y_span[y_idx]]
                # _seq_t = np.array(sub_seq_t.tolist() + [t])
                # _seq_s = np.array(sub_seq_s.tolist() + [[x_span[x_idx], y_span[y_idx]]])
                _map[x_idx][y_idx] = lam.value(
                    t, sub_seq_t, 
                    s, sub_seq_s)
        return _map
    # prepare the heatmap data in advance
    print('[%s] preparing the dataset %d × (%d, %d) for plotting.' %
        (arrow.now(), t_slots, grid_size, grid_size), file=sys.stderr)
    data = np.array([ heatmap(t_span[i]) for i in range(t_slots) ])

    # initiate the figure and plot
    fig = plt.figure()
    im  = plt.imshow(data[-1], cmap='hot', animated=True) # set last image initially for automatically setting color range.
    # function for updating the image of each frame
    def animate(i):
        # print(t_span[i])
        im.set_data(data[i])
        return im,
    # function for initiating the first image of the animation
    def init():
        im.set_data(data[0])
        return im,
    # animation
    print('[%s] start animation.' % arrow.now(), file=sys.stderr)
    anim = animation.FuncAnimation(fig, animate,
        init_func=init, frames=t_slots, interval=interval, blit=True)
    # show the plot
    plt.show()
    # # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Woody'), bitrate=1800)
    # anim.save('hpp.mp4', writer=writer)

def plot_spatio_temporal_points(points):
    """
    Plot points in a 2D space by their spatial location, as well as coloring the
    points with their corresponding time.
    """
    assert points.shape[1] == 3, 'Unable to plot spatio-temporal points with dimension >= 3'
    # We have three dimensions of data. x and y will be plotted on the x and y
    # axis, while z will be represented with color.
    # If z is a numpy array, matplotlib refuses to plot this.
    t, x, y = points[:, 0], points[:, 1], points[:, 2]

    # cmap will generate a tuple of RGBA values for a given number in the range
    # 0.0 to 1.0 (also 0 to 255 - not used in this example).
    # To map our z values cleanly to this range, we create a Normalize object.
    cmap = matplotlib.cm.get_cmap('viridis')
    normalize = matplotlib.colors.Normalize(vmin=min(t), vmax=max(t))
    colors = [cmap(normalize(value)) for value in t]

    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x, y, color=colors)

    # Optionally add a colorbar
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    plt.show()

class InfluentialMatrixSimulator(object):
    """An abstract class for simulating the influential matrix"""
    __metaclass__ = abc.ABCMeta

class GaussianInfluentialMatrixSimulator(InfluentialMatrixSimulator):
    """
    A simulator for Gaussian Influence Matrix
    An area can be represented by a fixed-length square separated by a specific
    grid. In this influential matrix, the influence of a given point in the area
    will be depicted by a gaussian kernel, which means, the given point (grid)
    have impact on surronding grids w with the value of a gaussian function
    depended on the their locations.
    """
    def __init__(self, length, grid_size, mu=[0, 0], cov=[[1,0],[0,1]]):
        assert len(grid_size) == 2, 'Invalid grid size %s' % grid_size
        self.length    = length    # the actual length of the square area
        self.grid_size = grid_size # the size of the grid (x_size, y_size)
        self.mu        = mu        # the offset of the influential location
        self.cov       = cov       # the covariance of the influential gaussian kernel
        self._construct_A()

    def _construct_A(self):
        """construct the influential matrix A."""
        matrix_size = self.grid_size[0] * self.grid_size[1]
        self.A      = np.zeros((matrix_size, matrix_size))
        for i in range(matrix_size):
            cur_x, cur_y = self.location(i)
            for j in range(matrix_size):
                sur_x, sur_y = self.location(j)
                self.A[i, j] = self._influence(cur_x, cur_y, sur_x, sur_y)

    def _influence(self, cur_x, cur_y, sur_x, sur_y):
        """calculate the surroundings influence regarding the current coordinates."""
        multi_normal = multivariate_normal(mean=[cur_x, cur_y], cov=self.cov)
        return multi_normal.pdf([sur_x, sur_y])

    def location(self, i):
        """calculate location according to the index of the component."""
        x_idx = int(i / self.grid_size[1])
        y_idx = int(i % self.grid_size[1])
        x = (x_idx / self.grid_size[0]) * self.length
        y = (y_idx / self.grid_size[1]) * self.length
        return x, y

def multi2spatial(seq_t, seq_d, ims):
    """convert multivariate sequence (seq_t, seq_d) to a spatio-temporal point process."""
    seq_s  = np.array([ ims.location(d) for d in seq_d ])
    seq_d  = seq_d.reshape((len(seq_d), 1))
    seq_t  = seq_t.reshape((len(seq_t), 1))
    points = np.concatenate([seq_t, seq_d, seq_s], axis=1)
    return points

def plot_multivariate_intensity(lam, points, S, t_slots, grid_size, interval):
    """Plot multivariate intensity as the time goes by."""
    assert len(S) == 3, '%d is an invalid dimension of the space.' % len(S)
    # split points into sequence of time and space.
    seq_t, seq_d, seq_s = points[:, 0], points[:, 1], points[:, 2:]
    print(seq_t)
    print(seq_d)
    print(seq_s)
    # define the span for each subspace
    t_span = np.linspace(S[0][0], S[0][1], t_slots+1)[1:]
    print(S[0][0])
    print(S[0][1])
    x_span = np.linspace(S[1][0], S[1][1], grid_size+1)[:-1]
    y_span = np.linspace(S[2][0], S[2][1], grid_size+1)[:-1]
    # function for yielding the heatmap over the entire region at a given time
    def heatmap(t):
        _map      = np.zeros((grid_size, grid_size))
        sub_seq_t = seq_t[seq_t < t]
        sub_seq_d = seq_d[:len(sub_seq_t)]
        for d in range(grid_size * grid_size):
            _seq_t = np.array(sub_seq_t.tolist() + [t])
            _seq_d = np.array(sub_seq_d.tolist() + [d])
            x_idx  = int(d / grid_size)
            y_idx  = int(d % grid_size)
            _map[x_idx][y_idx] = lam.value(_seq_t, _seq_d)
        return _map
    # prepare the heatmap data in advance
    print('[%s] preparing the dataset %d × (%d, %d) for plotting.' %
        (arrow.now(), t_slots, grid_size, grid_size), file=sys.stderr)
    data = np.array([ heatmap(t_span[i]) for i in range(t_slots) ])

    # initiate the figure and plot
    fig = plt.figure()
    im  = plt.imshow(data[-1], cmap=plt.get_cmap('hot'), # animated=True,
                     vmin=data.min(), vmax=data.max()) # set last image initially for automatically setting color range.
    # print(data[500])
    # function for updating the image of each frame
    def animate(i):
        # print(t_span[i])
        im.set_data(data[i])
        return im,
    # function for initiating the first image of the animation
    def init():
        im.set_data(data[0])
        return im,
    # animation
    print('[%s] start animation.' % arrow.now(), file=sys.stderr)
    anim = animation.FuncAnimation(fig, animate,
        init_func=init, frames=t_slots, interval=interval, blit=True)
    # show the plot
    plt.show()
