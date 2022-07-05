const USE_GPU = false
using Plots
using ParallelStencil
using ParallelStencil.FiniteDifferences1D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 1)
else
    @init_parallel_stencil(Threads, Float64, 1)
end

@parallel function compute_flux!(qHx, H, n, dx)
    @all(qHx) = - @av(H)^n * @d(H) / dx
    return nothing
end

@parallel function compute_update!(H, qHx, dt, dx)
    @inn(H) = @inn(H) -  dt * @d(qHx) / dx
    return nothing
end

function diffusion1D()
    # Physics
    lx  = 10.0
    n   = 3
    # Numerics
    nx  = 40
    nt  = 100
    dx  = lx/(nx-1)
    # Array initializations
    H   = @zeros(nx)
    qHx = @zeros(nx-1)
    # Initial conditions
    H  .= Data.Array([exp(-(((ix-1)*dx-lx/2)/2)^2) for ix=1:size(H,1)])
    # Time loop
    for it = 1:nt
        dt = dx*dx/maximum(H.^n)/2.1
        @parallel compute_flux!(qHx, H, n, dx)
        @parallel compute_update!(H, qHx, dt, dx)
        display(plot(H,title=it))
    end
    return
end

diffusion1D()
