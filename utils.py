#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import arrow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

def plot_spatial_intensity(lam, points, S, t_slots, grid_size, interval):
    '''
    Plot spatial intensity as the time goes by. The generated points can be also
    plotted on the same 2D space optionally.
    '''
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
                _seq_t = np.array(sub_seq_t.tolist() + [t])
                _seq_s = np.array(sub_seq_s.tolist() + [[x_span[x_idx], y_span[y_idx]]])
                _map[x_idx][y_idx] = lam.value(_seq_t, _seq_s)
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
        print(t_span[i])
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
    '''
    Plot points in a 2D space by their spatial location, as well as coloring the
    points with their corresponding time.
    '''
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
