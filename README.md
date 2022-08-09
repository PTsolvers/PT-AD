# PT-AD

The aim of this repository is to combine solvers using the [Pseudo-Transient Method](https://doi.org/10.5194/gmd-15-5757-2022) with Automatic Differentiation (AD) tools, here [Enzyme.jl](https://enzyme.mit.edu/julia/), on GPUs in order to retrieve necessary objects to perform inversions or parameter optimisation. 

 See our [JuliaCon22 talk](https://youtu.be/K2VtJe9baO4) for details.

As proof of concept, we successfully used AD to:
1. retrieve the residual of the Adjoint variable we can then use in the Pseudo-Transient solving procedure in order to solve for the Adjoint problem, and to;
2. retrieve the gradient of the cost-function with respect to the parameter we are inverting/optimising for.

## Benchmarks
The benchmark codes are available in the [**scripts_ok**](/scripts_ok/) folder and execute on Nvidia GPUs if `cuda` is appended in their name, and on the CPU otherwise. The `adjoint_nonlin_diff_1D_v2.jl` script provides a direct comparison between the CPU and GPU implementation.

### 1D power-law diffusion
We tested our workflow in a 1D configuration in which we try to invert for the power-law exponent of a nonlinear diffusion equation as described in [Reuber et al. 2020 - Appendix](https://doi.org/10.1016/j.jcp.2020.109797) and [Reuber 2021](https://doi.org/10.1007/s13137-021-00186-y). We used a gradient descent approach to minimise the misfit between observed (synthetic) and calculated quantity `H` to retrieve the optimal power-law exponent `n`:

![Inverting for power-law exponent in 1D](/docs/npow_inverse1D.gif)

> code: [`adjoint_nonlin_diff_1D_v2.jl`](/scripts_ok/adjoint_nonlin_diff_1D_v2.jl), [`adjoint_nonlin_diff_1D_v2_cuda.jl`](/scripts_ok/adjoint_nonlin_diff_1D_v2_cuda.jl)

### Shallow-Ice
We tested our approach to invert for the equilibrium line (ELA) or the "mass-balance gradient" in an ice-flow model as described by [Visnjevic et al. 2018](http://www.doi.org/10.1017/jog.2018.82), both in 1D and 2D. For testing purpose, we use a slightly more synthetic configuration:

![Inverting forMB gradient in 2D](/docs/inverse_2D_sia.gif)

> code: [`adjoint_nonlin_diff_react_2D_cuda.jl`](/scripts_ok/adjoint_nonlin_diff_react_2D_cuda.jl)

#### 1D inversion for ELA
The ELA inversion code is available only in 1D and was used to prototype the 2D SIA inversion.

> code: [`adjoint_nonlin_diff_react_1D_cuda_ELA.jl`](/scripts_ok/adjoint_nonlin_diff_react_1D_cuda_ELA.jl)

## Note on performance

### Effective memory throughput
Performance-wise, we compared the efficiency (measuring the effective memory throughput $T_\mathrm{eff}$ in GB/s) of the forward and adjoint solver. In addition, we also report the time per iteration as function of numerical grid resolution for both solvers.

![Effective memory throughput](/docs/Teff_timeit.png)

### Time per iteration
In addition, we also "naively" compared the execution time per forward solve iteration in seconds when executing both on the GPU and on the CPU. There we see about 2 orders of magnitude speed-up.

![CPU vs GPU timing](/docs/timeit_gpu_cpu.png)
