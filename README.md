Marked Spatio-Temporal Point Process Simulator
===

A set of Python tools called `STPPG` for simulating any form of Marked Spatio-temporal self-exciting point processes. 

### Usage

There are two critical files included in this repo: 

- `stppg.py` includes basic generators for homogeneous and inhomogeneous univariate point process, as well as various types of intensity classes and kernel functions.
- `utils.py` includes plotting functions for visualizing the point process simulation.

For simulating any parametric point process defined in references, we first need to define the `parametric form` of its conditional density and `event space` (time, location, and marks). 

- To define the form of the conditional density of a point process, a `Lam` object defined in `stppg.py` needs to be substantiated accordingly. The most important parameters of a conditional intensity `Lam` is defined in its kernel function. See the example below.
- For spatio-temporal point processes, time is specified by 1D two-elements list `T`, and location & mark space is jointly specified by 2D list `S`, where first two sub-lists indicate the location X and Y, and the following sub-lists indicate the mark space.  

### Examples

A simple example of simulating a spatio-temporal hawkes point process equipped with a standard diffusion kernel by using `stppg`.
```python
from stppg import StdDiffusionKernel, HawkesLam
from utils import plot_spatio_temporal_points, plot_spatial_intensity

np.random.seed(0)
np.set_printoptions(suppress=True)

# parameters initialization
mu     = .1
kernel = StdDiffusionKernel(C=1., beta=1., sigma_x=.1, sigma_y=.1)
lam    = HawkesLam(mu, kernel, maximum=1e+3)
pp     = SpatialTemporalPointProcess(lam)

# generate points
points, sizes = pp.generate(
    T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
    batch_size=500, verbose=True)

# plot intensity of the process over the time
plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
    t_slots=1000, grid_size=50, interval=50)
```

And see the console output below.
```bash
[2018-09-21T08:28:59.974629-04:00] generate samples (10167, 3) from homogeneous poisson point process
[2018-09-21T08:28:59.975163-04:00] 0 raw samples have been checked. 0 samples have been retained.
[2018-09-21T08:29:00.097072-04:00] 1000 raw samples have been checked. 1 samples have been retained.
[2018-09-21T08:29:00.327207-04:00] 2000 raw samples have been checked. 11 samples have been retained.
[2018-09-21T08:29:00.643629-04:00] 3000 raw samples have been checked. 21 samples have been retained.
[2018-09-21T08:29:01.061958-04:00] 4000 raw samples have been checked. 51 samples have been retained.
[2018-09-21T08:29:01.592077-04:00] 5000 raw samples have been checked. 71 samples have been retained.
[2018-09-21T08:29:02.242894-04:00] 6000 raw samples have been checked. 88 samples have been retained.
[2018-09-21T08:29:02.974756-04:00] 7000 raw samples have been checked. 100 samples have been retained.
[2018-09-21T08:29:03.779483-04:00] 8000 raw samples have been checked. 110 samples have been retained.
[2018-09-21T08:29:04.741294-04:00] 9000 raw samples have been checked. 128 samples have been retained.
[2018-09-21T08:29:05.729931-04:00] 10000 raw samples have been checked. 146 samples have been retained.
[2018-09-21T08:29:05.904117-04:00] thining samples (147, 3) based on Spatio-temporal Hawkes point process intensity with mu=0, beta=1 and Diffusion-type Kernel.
[[0.06299644 0.9183208  0.97126597]
 [0.11153312 0.280956   0.05509999]
 [0.121267   0.36991751 0.65443569]
 [0.12390124 0.49035887 0.64267112]
 [0.13272256 0.9463777  0.02484997]
     ... ...
 [0.97093684 0.13255896 0.15690197]
 [0.97663217 0.6115613  0.34787382]
 [0.97863127 0.21207182 0.19613095]
 [0.98051828 0.43338621 0.67951877]
 [0.99105454 0.21976152 0.26218951]]
[2018-09-21T08:29:05.911032-04:00] preparing the dataset 1000 × (50, 50) for plotting.
[2018-09-21T08:32:26.300910-04:00] start animation.
```

The animations below show the progression of conditional intensities through time with different types of kernel functions.

Standard Diffusion Kernel     | Spatial-Variant Gaussian Diffusion Kernel    | Spatial-Variant Gaussian Mixture Diffusion Kernel
:----------------------------:|:----------------------------:|:----------------------------:
![](https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator/blob/master/results/stppg3.gif)  |  ![](https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator/blob/master/results/gaussian_kernel.gif) | ![](https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator/blob/master/results/gaussian_mixture_kernel.gif)

### References

- [Shixiang Zhu, Yao Xie. "Reinforcement Learning of Spatio-Temporal Point Processes."](https://arxiv.org/abs/1906.05467)
- [Y. Ogata. "Space-Time Point-Process Models for Earthquake Occurrences"](https://link.springer.com/article/10.1023/A:1003403601725)
- [F. Musmeci, D. Vere-Jones. "A Space-Time Clustering Model for Historical Earthquakes"](https://link.springer.com/content/pdf/10.1007%2FBF00048666.pdf)
- [S. Zhu and Y. Xie. "Crime Linkage Detection by Spatio-Temporal-Textual Point Processes"](https://arxiv.org/abs/1902.00440)
