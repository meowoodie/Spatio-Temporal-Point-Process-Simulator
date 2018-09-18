#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import arrow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_temporal_intensity(points):
    pass

def plot_spatio_temporal_points(points):
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
