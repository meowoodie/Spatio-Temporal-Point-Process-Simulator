#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
STPPG: Spatio-Temporal Point Process Generator

References:
- https://www.jstatsoft.org/article/view/v053i02
- https://www.ism.ac.jp/editsec/aism/pdf/044_1_0001.pdf
- https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator

Dependencies:
- Python 3.6.7
"""

import sys
import utils
import arrow
import numpy as np
from scipy.stats import norm

class StdDiffusionKernel(object):
    """
    Kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    """
    def __init__(self, C=1., beta=1., sigma_x=1., sigma_y=1.):
        self.C       = C
        self.beta    = beta
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s
        delta_t = t - his_t
        delta_x = delta_s[:, 0]
        delta_y = delta_s[:, 1]
        return np.exp(- self.beta * delta_t) * \
            (self.C / (2 * np.pi * self.sigma_x * self.sigma_y * delta_t)) * \
            np.exp((- 1. / (2 * delta_t)) * \
                ((np.square(delta_x) / np.square(self.sigma_x)) + \
                (np.square(delta_y) / np.square(self.sigma_y))))



class GaussianDiffusionKernel(object):
    """
    A Gaussian diffusion kernel function based on the standard kernel function proposed 
    by Musmeci and Vere-Jones (1992). The angle and shape of diffusion ellipse is able  
    to vary according to the location.  
    """
    def __init__(self, 
        layers=[20, 20], beta=1., C=1., Ws=None, bs=None,
        SIGMA_SHIFT=.1, SIGMA_SCALE=.25, MU_SCALE=0.1, is_centered=True):
        # constant configuration
        self.MU_SCALE    = MU_SCALE
        self.SIGMA_SHIFT = SIGMA_SHIFT
        self.SIGMA_SCALE = SIGMA_SCALE
        # kernel parameters
        self.C           = C # kernel constant
        self.beta        = beta
        self.Ws          = []
        self.bs          = []
        self.is_centered = is_centered
        # construct multi-layers neural networks
        # where 2 is for x and y; And 5 is for mu_x, mu_y, sigma_x, sigma_y, rho
        self.layers = [2] + layers + [5]
        # construct weight & bias matrix layer by layer
        for i in range(len(self.layers)-1):
            if Ws is None and bs is None:
                W = np.random.normal(scale=5.0, size=[self.layers[i], self.layers[i+1]])
                b = np.random.normal(size=self.layers[i+1])
                self.Ws.append(W)
                self.bs.append(b)
            else: 
                if Ws[i].shape == (self.layers[i], self.layers[i+1]) and len(bs[i]) == self.layers[i+1]:
                    self.Ws.append(Ws[i])
                    self.bs.append(bs[i])
                else:
                    raise Exception("Incompatible shape of the weight matrix W=%s, b=%s at %d-th layer." % (Ws[i].shape, b[i].shape, i))

    def nonlinear_mapping(self, locations):
        """nonlinear mapping from the location space to the parameter space."""
        # construct multi-layers neural networks
        output = locations
        for i in range(len(self.layers)-1):
            output = self.__sigmoid(np.matmul(output, self.Ws[i]) + self.bs[i])
        # project to parameters space
        mu_xs    = (output[:, 0] - 0.5) * 2 * self.MU_SCALE
        mu_ys    = (output[:, 1] - 0.5) * 2 * self.MU_SCALE
        sigma_xs = output[:, 2] * self.SIGMA_SCALE + self.SIGMA_SHIFT # sigma_x spans (SIGMA_SHIFT, SIGMA_SHIFT + SIGMA_SCALE)
        sigma_ys = output[:, 3] * self.SIGMA_SCALE + self.SIGMA_SHIFT # sigma_y spans (SIGMA_SHIFT, SIGMA_SHIFT + SIGMA_SCALE)
        rhos     = output[:, 4] * 1.5 - .75                           # rho spans (-1, 1)
        return mu_xs, mu_ys, sigma_xs, sigma_ys, rhos

    def nu(self, t, s, his_t, his_s):
        delta_s = s - his_s
        delta_t = t - his_t
        delta_x = delta_s[:, 0]
        delta_y = delta_s[:, 1]
        mu_xs, mu_ys, sigma_xs, sigma_ys, rhos = self.nonlinear_mapping(his_s)
        if self.is_centered:
            mu_xs = 0
            mu_ys = 0
        return np.exp(- self.beta * delta_t) * \
            (self.C / (2 * np.pi * sigma_xs * sigma_ys * delta_t * np.sqrt(1 - np.square(rhos)))) * \
            np.exp((- 1. / (2 * delta_t * (1 - np.square(rhos)))) * \
                ((np.square(delta_x - mu_xs) / np.square(sigma_xs)) + \
                (np.square(delta_y - mu_ys) / np.square(sigma_ys)) - \
                (2 * rhos * (delta_x - mu_xs) * (delta_y - mu_ys) / (sigma_xs * sigma_ys))))

    @staticmethod
    def __sigmoid(x):
        """sigmoid activation function for nonlinear mapping"""
        return 1. / (1. + np.exp(-x))



# class GaussianMixtureDiffusionKernel(object):
#     """
#     A Gaussian mixture diffusion kernel function is superposed by multiple Gaussian diffusion 
#     kernel function. The number of the Gaussian components is specified by n_comp. 
#     """
#     def __init__(self, n_comp, layers, beta=1., C=1., SIGMA_SHIFT=.1, SIGMA_SCALE=.25, MU_SCALE=.1):
#         self.n_comp = n_comp
#         self.gdks   = []
#         for k in range(self.n_comp):
#             gdk = GaussianDiffusionKernel(
#                 layers=layers, beta=beta, C=C, Ws=None, bs=None, 
#                 SIGMA_SHIFT=SIGMA_SHIFT, SIGMA_SCALE=SIGMA_SCALE, MU_SCALE=MU_SCALE, is_centered=False)
#             self.gdks.append(gdk)
    
#     def nu(self, t, s, his_t, his_s):
#         nu = 0
#         for k in range(self.n_comp):
#             nu += (1./self.n_comp) * self.gdks[k].nu(t, s, his_t, his_s)
#         return nu



class GaussianMixtureDiffusionKernel(object):
    """
    A Gaussian mixture diffusion kernel function is superposed by multiple Gaussian diffusion 
    kernel function. The number of the Gaussian components is specified by n_comp. 
    """
    def __init__(self, n_comp, layers, 
        beta=1., C=1., SIGMA_SHIFT=.1, SIGMA_SCALE=.25, MU_SCALE=.1,
        Wss=None, bss=None, W_phis=None):
        self.W_phis = W_phis
        self.n_comp = n_comp
        self.gdks   = []
        # Gaussian mixture component initialization
        for k in range(self.n_comp):
            Ws  = Wss[k] if Wss is not None else None
            bs  = bss[k] if bss is not None else None
            gdk = GaussianDiffusionKernel(
                layers=layers, beta=beta, C=C, Ws=Ws, bs=bs, 
                SIGMA_SHIFT=SIGMA_SHIFT, SIGMA_SCALE=SIGMA_SCALE, MU_SCALE=MU_SCALE, is_centered=False)
            self.gdks.append(gdk)
        # Gaussian mixture weighting matrix initialization
        if self.W_phis is None:
            self.W_phis = [ np.random.normal(scale=5.0, size=[2, 1]) for k in range(self.n_comp) ]
    
    def nu(self, t, s, his_t, his_s):
        nu = 0
        for k in range(self.n_comp):
            phi = self._softmax(his_s, k)                   # [ n_his ]
            nu += phi * self.gdks[k].nu(t, s, his_t, his_s) # [ n_his ]
        return nu

    def _softmax(self, locations, k):
        """
        Gaussian mixture components are weighted by phi^k, which are computed by a softmax function, i.e., 
        phi^k(x, y) = e^{[x y]^T w^k} / \sum_{i=1}^K e^{[x y]^T w^i}
        """
        numerator   = np.exp(np.matmul(locations, self.W_phis[k])).flatten()
        denominator = np.concatenate([ 
            np.exp(np.matmul(locations, self.W_phis[i])) 
            for i in range(self.n_comp) ], axis=1).sum(axis=1)
        phis        = numerator / denominator
        return phis


class HawkesLam(object):
    """Intensity of Spatio-temporal Hawkes point process"""
    def __init__(self, mu, kernel, maximum=1e+4):
        self.mu      = mu
        self.kernel  = kernel
        self.maximum = maximum

    def value(self, t, his_t, s, his_s):
        """
        return the intensity value at (t, s).
        The last element of seq_t and seq_s is the location (t, s) that we are
        going to inspect. Prior to that are the past locations which have
        occurred.
        """
        if len(his_t) > 1:
            val = self.mu + np.sum(self.kernel.nu(t, s, his_t, his_s))
        else:
            val = self.mu
        return val

    def upper_bound(self):
        """return the upper bound of the intensity value"""
        return self.maximum

    def __str__(self):
        return "Hawkes processes"

class SpatialTemporalPointProcess(object):
    """
    Marked Spatial Temporal Hawkes Process

    A stochastic spatial temporal points generator based on Hawkes process.
    """

    def __init__(self, lam):
        """
        Params:
        """
        # model parameters
        self.lam     = lam

    def _homogeneous_poisson_sampling(self, T=[0, 1], S=[[0, 1], [0, 1]]):
        """
        To generate a homogeneous Poisson point pattern in space S X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S X T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.

        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            S:   [(min_t, max_t), (min_x, max_x), (min_y, max_y), ...] indicates the
                range of coordinates regarding a square (or cubic ...) region.
        Returns:
            samples: point process samples:
            [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
        """
        _S     = [T] + S
        # sample the number of events from S
        n      = utils.lebesgue_measure(_S)
        N      = np.random.poisson(size=1, lam=self.lam.upper_bound() * n)
        # simulate spatial sequence and temporal sequence separately.
        points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
        points = np.array(points).transpose()
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points[:, 0].argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape), file=sys.stderr)
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            t     = homo_points[i, 0]
            s     = homo_points[i, 1:]
            his_t = retained_points[:, 0]
            his_s = retained_points[:, 1:]
            # thinning
            lam_value = self.lam.value(t, his_t, s, his_s)
            lam_bar   = self.lam.upper_bound()
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("intensity %f is greater than upper bound %f." % (lam_value, lam_bar), file=sys.stderr)
                return None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, homo_points[[i], :]], axis=0)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]), file=sys.stderr)
        # log the final results of the thinning algorithm
        if verbose:
            print("[%s] thining samples %s based on %s." % \
                (arrow.now(), retained_points.shape, self.lam), file=sys.stderr)
        return retained_points

    def generate(self, T=[0, 1], S=[[0, 1], [0, 1]], batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling(T, S)
            points      = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            print("[%s] %d-th sequence is generated." % (arrow.now(), b+1), file=sys.stderr)
            b += 1
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len, 3))
        for b in range(batch_size):
            data[b, :points_list[b].shape[0]] = points_list[b]
        return data, sizes