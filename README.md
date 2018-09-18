Spatio-Temporal Point Process Simulator
===

### Examples

A simple example for simulating a spatio-temporal hawkes point process.
```python
from generator import inhomogeneous_poisson_process, SpatioTemporalHawkesLam
from utils import plot_spatio_temporal_points, plot_temporal_intensity, plot_spatial_intensity

np.random.seed(0)
np.set_printoptions(suppress=True)

# define time and spatial space
T = (0, 1)           # from time 0 to 1
S = [(0, 1), (0, 1)] # x from 0 to 1, y from 0 to 1
# define intensity function
lam    = SpatioTemporalHawkesLam(mu=1., alpha=1., beta=1., sigma=[10., 1.])
# generate points
points = inhomogeneous_poisson_process(lam, T, S)
print(points)
```

And see the console output below.
```bash
[2018-09-18T17:21:05.016068-04:00] generate samples (503, 3) from homogeneous poisson point process
[2018-09-18T17:21:05.068921-04:00] thining samples (16, 3) based on Spatio-temporal Hawkes point process intensity with mu=1, beta=1, sigma=[10.0, 1.0]
[[0.13105523 0.3145733  0.13417364]
 [0.24442559 0.98058013 0.08329098]
 [0.32001715 0.7814796  0.99266699]
 [0.38346389 0.75102165 0.11861552]
 [0.42408899 0.33026704 0.13168728]
 [0.45354268 0.40379274 0.43553154]
 [0.57722859 0.60630813 0.92797617]
 [0.58447607 0.85877747 0.57360975]
 [0.59223042 0.73685316 0.11224999]
 [0.76532525 0.87739879 0.69374702]
 [0.82211773 0.9325612  0.99607127]
 [0.86055117 0.00054596 0.82347172]
 [0.87650525 0.92275661 0.30065116]
 [0.8811032  0.99927799 0.96239507]
 [0.9425836  0.99980858 0.14043952]
 [0.96193638 0.45850317 0.95504668]]
```

Here an animation of variation of spatial intensities as time goes by has also been provided, simulated by a Spatio-temporal Hawkes Point process.

<img width="460" height="460" src="https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator/blob/master/results/hpp_clips.gif">
