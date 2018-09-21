Spatio-Temporal Point Process Simulator
===

Simple Python functions showing how to simulate Spatio-temporal point processes, and marked point processes (there is an example below that shows how to generate and plot a spatio-temporal hawkes point process). In general, there are two ways to model the spatial-temporal events:

- `STPPG`: The most general way is modeling the spatial-temporal points as a univariate point process where each single point consists of time and location coordinates (or even marks).
- `MVPPG`: Another way is modeling the spatial-temporal points as a multivariate point process that views events occurred in different discrete locations as individual point processes (simulates the locations by using different components of the point process), and uses an influential matrix to depict the dependencies between different discrete locations.

### Usage

> Please see the comments in the source code and unittest for the detailed usage.

- `stppg.py` basic generators for homogeneous and inhomogeneous univariate point process, as well as different kinds of intensity classes and kernel functions.
- `mvppg.py` generators for multivariate point process, as well as different kinds of intensity classes and kernel functions.
- `utils.py` Some simple plotting functions for visualizing the point process simulation.

### Examples

A simple example for simulating a spatio-temporal hawkes point process by using `stppg`.
```python
from stppg import inhomogeneous_poisson_process, SpatioTemporalHawkesLam, DiffusionKernel
from utils import plot_spatio_temporal_points, plot_spatial_intensity

np.random.seed(0)
np.set_printoptions(suppress=True)

# define time and spatial space
S = [(0, 1),         # t from 0 to 1 and
     (0, 1), (0, 1)] # x from 0 to 1, y from 0 to 1
# define kernel function and intensity function
kernel = DiffusionKernel(beta=1., C=1., sigma=[1., 1.])
lam    = SpatioTemporalHawkesLam(mu=.1, alpha=.1, beta=1., kernel=kernel, maximum=1e+4)
# generate points
points = inhomogeneous_poisson_process(lam, S)
print(points)
# plot intensity of the process over the time
plot_spatial_intensity(lam, points, S,
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

Here an animation of variation of spatial intensities as time goes by, simulated by a spatio-temporal Hawkes Point process.

<img width="460" height="460" src="https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator/blob/master/results/stppg.gif">

Another example is for simulating a multi-variate point process. Each of the components uniquely indicates a specific discrete region in a 2D space.

```python
from mvppg import ExpKernel, MultiVariateLam, inhomogeneous_multivariate_poisson_process
from utils import plot_spatio_temporal_points, plot_spatial_intensity, plot_multivariate_intensity, GaussianInfluentialMatrixSimulator, multi2spatial

np.random.seed(0)
np.set_printoptions(suppress=True)

d      = 20
cov    = [[.1, 0.], [0., .1]]
beta   = 1e-5
D      = d * d
T      = (0, 1)
Mu     = np.zeros(D)
ims = GaussianInfluentialMatrixSimulator(
    length=1., grid_size=[d, d], mu=[0., 0.], cov=cov)
A      = ims.A
kernel = ExpKernel(beta=beta)
lam    = MultiVariateLam(D, Mu=Mu, A=A, kernel=kernel, maximum=100.)
ts, ds = inhomogeneous_multivariate_poisson_process(lam, D, T)
points = multi2spatial(ts, ds, ims)
# plot intensity of the process over the time
plot_multivariate_intensity(lam, points, S=[T, (0, 1), (0, 1)],
    t_slots=1000, grid_size=d, interval=50)
```

Here an animation of variation of multivariate intensities as time goes by. 

<img width="460" height="460" src="https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator/blob/master/results/mvppg.gif">
