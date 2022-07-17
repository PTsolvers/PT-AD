using Enzyme,Plots,Printf
using CUDA

@inbounds function residual!(R,H,npow,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(R)-1
        R[ix] = (H[ix-1]^(npow[ix]+1.0) - 2.0*H[ix]^(npow[ix]+1.0) + H[ix+1]^(npow[ix]+1.0))/dx/dx/(npow[ix]+1.0) # fails
        # R[ix] = (H[ix-1]^2 - 2.0*H[ix]^2 + H[ix+1]^2)/dx/dx/(npow[ix]+1.0) # works
        # R[ix] = (H[ix-1]^npow[ix] - 2.0*H[ix]^2 + H[ix+1]^2)/dx/dx/(npow[ix]+1.0) # fails
    end
    return
end

function residual_grad!(R,Jn,H,JVP,npow,dx)
    Enzyme.autodiff_deferred(residual!,Const,Duplicated(R,Jn),Duplicated(H,JVP),Const(npow),Const(dx))
    return
end

function cost_grad!(R,JVP,H,npow,Jn,dx)
    Enzyme.autodiff_deferred(residual!,Const,Duplicated(R,JVP),Const(H),Duplicated(npow,Jn),Const(dx))
    return
end

# function cost(H,H_obs)
#     J = 0.0
#     @inbounds @simd for ix ∈ eachindex(H)
#         J += (H[ix]-H_obs[ix])^2
#     end
#     return 0.5*J
# end

@inbounds function adjoint_residual!(R,Ψ,H,H_obs,npow,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(R)-1
        ∇Ψ_l  = (Ψ[ix  ]-Ψ[ix-1])/dx
        ∇Ψ_r  = (Ψ[ix+1]-Ψ[ix  ])/dx
        Hn_l  = 0.5*(H[ix-1]^npow[ix-1]+H[ix  ]^npow[ix  ])
        Hn_r  = 0.5*(H[ix  ]^npow[ix  ]+H[ix+1]^npow[ix+1])
        R[ix] = 2.0*(Hn_r*∇Ψ_r - Hn_l*∇Ψ_l)/dx - H[ix]^npow[ix]*(∇Ψ_r - ∇Ψ_l)/dx + H[ix] - H_obs[ix]
    end
    return
end

@inbounds function cost_gradient!(Jn,Ψ,H,npow,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(H)-1
        Jn[ix] = H[ix]^npow[ix]*log(H[ix])*0.5*(H[ix+1]-H[ix-1])/dx*0.5*(Ψ[ix+1]-Ψ[ix-1])/dx
    end
    if ix==1         Jn[ix] = Jn[ix+1] end
    if ix==length(H) Jn[ix] = Jn[ix-1] end
    return
end

@views function main()
    # physics
    lx       = 20.0
    npow0    = 3.0
    # numerics
    nx       = 256
    threads  = 128
    blocks   = cld(nx,threads)
    niter    = 100nx
    nchk     = 5nx
    εtol     = 1e-8
    dmp      = 1/2
    dmp_adj  = 3/2
    dmp_adj2 = 1dmp_adj
    # preprocessing
    dx       = lx/nx
    xc       = LinRange(dx/2,lx-dx/2,nx)
    dt       = dx/3
    # init
    H        = CuArray(collect(1.0 .- 0.5.*xc./lx))
    H_obs    = copy(H)
    npow_s   = CUDA.fill(npow0  ,nx)
    npow     = CUDA.fill(npow0-2.0,nx)
    R        = CUDA.zeros(Float64,nx)
    R_an     = CUDA.zeros(Float64,nx)
    R_obs    = CUDA.zeros(Float64,nx)
    dR       = CUDA.zeros(Float64,nx)
    dR_an    = CUDA.zeros(Float64,nx)
    dR_obs   = CUDA.zeros(Float64,nx)
    R_obs    = CUDA.zeros(Float64,nx)
    ∂J_∂H    = CUDA.zeros(Float64,nx)
    Jn       = CUDA.zeros(Float64,nx)
    Jn_an    = CUDA.zeros(Float64,nx)
    Ψ        = CUDA.zeros(Float64,nx) # adjoint state (discretize-then-optimise)
    Ψ_an     = CUDA.zeros(Float64,nx) # adjoint state (optimise-then-discretize)
    JVP      = CUDA.zeros(Float64,nx) # Jacobian-vector product storage
    dΨdt     = CUDA.zeros(Float64,nx)
    # action
    # forward solve
    println("forward solve...")
    for iter = 1:niter
        @cuda blocks=blocks threads=threads residual!(dR_obs,H_obs,npow_s,dx); synchronize()
        @cuda blocks=blocks threads=threads residual!(dR    ,H    ,npow  ,dx); synchronize()
        @. R_obs  = R_obs*(1.0-dmp/nx) + dt*dR_obs
        @. R      = R    *(1.0-dmp/nx) + dt*dR
        @. H_obs += dt*R_obs
        @. H     += dt*R
        if iter % nchk == 0
            merr1 = maximum(abs.(dR_obs))
            merr2 = maximum(abs.(dR))
            @printf("  #iter/nx = %.1f, err = %.1e, %.1e\n",iter/nx,merr1,merr2)
            if max(merr1,merr2) < εtol break end
        end
    end
    println("done")
    # adjoint solve
    println("adjoint solve...")
    @. ∂J_∂H = H - H_obs
    for iter = 1:niter
        # discretize-then-optimise
        JVP .= 0.0; Jn .= Ψ
        @cuda blocks=blocks threads=threads residual_grad!(R,Jn,H,JVP,npow,dx); synchronize()
        @. dΨdt = dΨdt*(1.0-dmp_adj2/nx) + dt*(JVP + ∂J_∂H)
        @. Ψ[2:end-1] += dt*dΨdt[2:end-1]
        # optimise-then-discretize
        @cuda blocks=blocks threads=threads adjoint_residual!(dR_an,Ψ_an,H,H_obs,npow,dx); synchronize()
        @. R_an  = R_an*(1.0-dmp_adj/nx) + dt*dR_an
        @. Ψ_an += dt*R_an
        # check convergence
        if iter % nchk == 0
            merr1 = maximum(abs.(dΨdt[2:end-1]))
            merr2 = maximum(abs.(dR_an))
            @printf("  #iter/nx = %.1f, err = %.1e, %.1e\n",iter/nx,merr1,merr2)
            if max(merr1,merr2) < εtol break end
        end
    end
    println("done")
    # dJdn
    JVP .= Ψ; Jn .= 0.0
    @cuda blocks=blocks threads=threads cost_grad!(R,JVP,H,npow,Jn,dx); synchronize()
    @cuda blocks=blocks threads=threads cost_gradient!(Jn_an,Ψ,H_obs,npow_s,dx); synchronize()
    p1 = plot(xc,Array([H,H_obs]) ; title="H" , label=["H" "H_obs"])
    p2 = plot(xc,Array([Jn,Jn_an]); title="Jn", label=["dto" "otd"])
    p3 = plot(xc,Array([Ψ,Ψ_an])  ; title="Ψ" , label=["dto" "otd"])
    display(plot(p1,p2,p3))
    return
end

main()
