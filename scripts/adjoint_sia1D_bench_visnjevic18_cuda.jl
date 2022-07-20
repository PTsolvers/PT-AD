using Enzyme,Plots,Printf
using CUDA

function mypow(a,n::Int)
    if n==3  a*a*a
    elseif n==5  tmp=a*a; tmp*tmp*a
    else NaN
    end
end

@inbounds function residual!(R,H,B,Ela,β,dτ,c,npow::Int,a1,a2,cfl,ε,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(H)-1
        # D_l    = (a1 * (0.5*(H[ix-1]+H[ix]))^(npow+2) + a2 * (0.5*(H[ix-1]+H[ix]))^npow) * ((B[ix]-B[ix-1])/dx + (H[ix]-H[ix-1])/dx)^2
        # D_r    = (a1 * (0.5*(H[ix]+H[ix+1]))^(npow+2) + a2 * (0.5*(H[ix]+H[ix+1]))^npow) * ((B[ix+1]-B[ix])/dx + (H[ix+1]-H[ix])/dx)^2
        D_l    = (a1 * mypow(0.5*(H[ix-1]+H[ix]),npow+2) + a2 * mypow(0.5*(H[ix-1]+H[ix]),npow)) * ((B[ix]-B[ix-1])/dx + (H[ix]-H[ix-1])/dx)^2
        D_r    = (a1 * mypow(0.5*(H[ix]+H[ix+1]),npow+2) + a2 * mypow(0.5*(H[ix]+H[ix+1]),npow)) * ((B[ix+1]-B[ix])/dx + (H[ix+1]-H[ix])/dx)^2
        qHx_l  = -D_l*((B[ix  ]-B[ix-1])/dx + (H[ix  ]-H[ix-1])/dx)
        qHx_r  = -D_r*((B[ix+1]-B[ix  ])/dx + (H[ix+1]-H[ix  ])/dx)
        R[ix]  = -(qHx_r - qHx_l)/dx + min(β[ix] * (B[ix] + H[ix] - Ela[ix]), c)
        dτ[ix] = 0.5*min(1.0, cfl/(ε + max(D_l, D_r)))
    end
    return
end

function cost(H,H_obs)
    J = 0.0
    @inbounds @simd for ix ∈ eachindex(H)
        J += (H[ix]-H_obs[ix])^2
    end
    return 0.5*J
end

mutable struct ForwardProblem{T<:Real,A<:AbstractArray{T}}
    H::A; R::A; dR::A; dτ::A; B::A; Ela::A; β::A;
    niter::Int; ncheck::Int; ϵtol::T; threads::Int; blocks::Int;
    dx::T; dmp::T; npow::Int; a1::T; a2::T; c::T; ε::T;
end

function ForwardProblem(H,B,Ela,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp,npow,a1,a2,c,ε)
    R  = similar(H)
    dR = similar(H)
    dτ = similar(H)
    return ForwardProblem(H,R,dR,dτ,B,Ela,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp,npow,a1,a2,c,ε)
end

function solve!(problem::ForwardProblem)
    (;H,R,dR,dτ,B,Ela,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp,npow,a1,a2,c,ε) = problem
    nx  = length(H)
    cfl = dx^2/2.1
    R .= 0; dR .= 0; dτ .= 0
    Err = CUDA.zeros(Float64,nx)
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        @cuda blocks=blocks threads=threads residual!(dR,H,B,Ela,β,dτ,c,npow,a1,a2,cfl,ε,dx); synchronize()
        @. Err = H
        @. R   = R*dmp + dR
        @. H   = max(0.0, H + dτ*R)
        if iter % ncheck == 0
            @. Err -= H
            @show merr = maximum(abs.(Err))
            isfinite(merr) || error("forward solve failed") 
        end
        iter += 1
    end
    if iter == niter && merr >= ϵtol
        error("forward solve not converged")
    end
    @printf("    forward solve converged: #iter/nx = %.1f, err = %.1e\n",iter/nx,merr)
    return
end

mutable struct AdjointProblem{T<:Real,A<:AbstractArray{T}}
    Ψ::A; R::A; dR; dτ::A; tmp1::A; tmp2::A; ∂J_∂H::A; H::A; H_obs::A; B::A; Ela::A; β::A;
    niter::Int; ncheck::Int; ϵtol::T; threads::Int; blocks::Int;
    dx::T; dmp::T; npow::Int; a1::T; a2::T; c::T; ε::T;
end

function AdjointProblem(H,H_obs,B,Ela,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp,npow,a1,a2,c,ε)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    dτ    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    return AdjointProblem(Ψ,R,dR,dτ,tmp1,tmp2,∂J_∂H,H,H_obs,B,Ela,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp,npow,a1,a2,c,ε)
end

function residual_grad!(tmp1,tmp2,H,dR,B,Ela,β,dτ,c,npow,a1,a2,cfl,ε,dx)
    Enzyme.autodiff_deferred(residual!,Const,Duplicated(tmp1,tmp2),Duplicated(H,dR),Const(B),Const(Ela),Const(β),Const(dτ),Const(c),Const(npow),Const(a1),Const(a2),Const(cfl),Const(ε),Const(dx))
    return
end

function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,dτ,tmp1,tmp2,∂J_∂H,H,H_obs,B,Ela,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp,npow,a1,a2,c,ε) = problem
    nx  = length(Ψ)
    cfl = dx^2/2.1
    Ψ .= 0; R .= 0; dR .= 0; dτ .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        @cuda blocks=blocks threads=threads residual_grad!(tmp1,tmp2,H,dR,B,Ela,β,dτ,c,npow,a1,a2,cfl,ε,dx); synchronize()
        @. R  = R*dmp + dR
        @. Ψ += 0.1*R
        Ψ[1:1] .= 0; Ψ[end:end] .= 0
        if iter % ncheck == 0
            @show merr = maximum(abs.(dR[2:end-1]))
            isfinite(merr) || error("adjoint solve failed") 
        end
        iter += 1
    end
    if iter == niter && merr >= ϵtol
        error("adjoint solve not converged")
    end
    @printf("    adjoint solve converged: #iter/nx = %.1f, err = %.1e\n",iter/nx,merr)
    return
end

function cost_grad!(tmp1,tmp2,H,B,Ela,Jn,β,dτ,c,npow,a1,a2,cfl,ε,dx)
    Enzyme.autodiff_deferred(residual!,Const,Duplicated(tmp1,tmp2),Const(H),Const(B),Duplicated(Ela,Jn),Const(β),Const(dτ),Const(c),Const(npow),Const(a1),Const(a2),Const(cfl),Const(ε),Const(dx))
    return
end

function cost_gradient!(Jn,problem::AdjointProblem)
    (;Ψ,dτ,tmp1,tmp2,H,B,Ela,β,threads,blocks,dx,npow,a1,a2,c,ε) = problem
    tmp1 .= .-Ψ; Jn .= 0.0
    cfl = dx^2/2.1
    @cuda blocks=blocks threads=threads cost_grad!(tmp1,tmp2,H,B,Ela,Jn,β,dτ,c,npow,a1,a2,cfl,ε,dx); synchronize()
    Jn[1:1] .= Jn[2:2]; Jn[end:end] .= Jn[end-1:end-1]
    return
end

@views function main()
    # physics
    ρg           = 910.0*9.81
    s2yr         = 31557600.0
    npow         = 3
    lx           = 250000
    B0           = 3500
    β0           = 0.01
    c            = 2.0
    # numerics
    ε            = 1e-2
    nx           = 128
    threads      = 128
    blocks       = cld(nx,threads)
    niter        = 300nx
    ncheck       = 5nx
    ϵtol         = 1e-8
    gd_ϵtol      = 1e-5
    dmp          = 0.72
    dmp_adj      = 0.5
    gd_niter     = 500
    bt_niter     = 10
    γ0           = 1e2
    # preprocessing
    dx           = lx/nx
    xc           = LinRange(-lx/2+dx/2, lx/2-dx/2, nx)
    a1           = 1.9e-24*ρg^npow*s2yr
    a2           = 5.7e-20*ρg^npow*s2yr
    # init
    H            = CuArray( 100.0*ones(nx) )
    H[[1,end]]  .= 0.0
    H_obs        = copy(H)
    H_ini        = copy(H)
    B            = CuArray( @. B0*exp(-xc*xc/1e10) + B0*exp.(-xc*xc/1e9) )
    Ela_synt     = CuArray( @. 3000 + 400 * atan(xc/lx) )
    Ela_ini      = CuArray( 3000 .+ 400.0 .* rand(nx) )
    Ela          = copy(Ela_ini)
    # β_synt       = @. β0 + 0.015 * atan(xc/lx)
    # β_ini        = 0.0153 .+ 0.007.*rand(nx)
    β            = CUDA.fill(β0,nx)
    Jn           = CUDA.zeros(Float64,nx) # cost function gradient
    synt_problem = ForwardProblem(H_obs,B,Ela_synt,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp    ,npow,a1,a2,c,ε)
    fwd_problem  = ForwardProblem(H    ,B,Ela     ,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp    ,npow,a1,a2,c,ε)
    adj_problem  = AdjointProblem(H,H_obs,B,Ela   ,β,niter,ncheck,ϵtol,threads,blocks,dx,dmp_adj,npow,a1,a2,c,ε)
    # action
    println("  generating synthetic data...")
    solve!(synt_problem)
    solve!(fwd_problem)
    println("  gradient descent")

    # solve!(adj_problem)

    # S  = B .+ H_obs
    # S[H_obs.==0] .= NaN
    # p1 = plot(xc,[B,S] , label=["Bed" "Surface"], linewidth=3)
    # p2 = plot(xc, H_obs, label="Ice thick", linewidth=3)
    # # p3 = plot(xc,β , label="β", linewidth=3)
    # p3 = plot(xc,Ela_synt, label="ELA", linewidth=3)
    # display(plot(p1,p2,p3, layout=(3, 1)))

    γ = γ0
    J_old = cost(H,H_obs)
    J_evo = Float64[]; iter_evo = Int[]
    # for gd_iter = 1:gd_niter
        Ela_ini .= Ela
        # adjoint solve
        solve!(adj_problem)
        # compute cost function gradient
        cost_gradient!(Jn,adj_problem)
    #     # line search
    #     for bt_iter = 1:bt_niter
    #         @. npow -= γ*Jn
    #         fwd_problem.H .= H_ini
    #         solve!(fwd_problem)
    #         J_new = cost(H,H_obs)
    #         if J_new < J_old
    #             γ *= 1.2
    #             J_old = J_new
    #             break
    #         else
    #             npow .= npow_init
    #             γ *= 0.5
    #         end
    #     end
    #     push!(iter_evo,gd_iter); push!(J_evo,J_old)
    #     if J_old < gd_ϵtol
    #         @printf("  gradient descent converged, misfit = %.1e\n", J_old)
    #         break
    #     else
    #         @printf("  #iter = %d, misfit = %.1e\n", gd_iter, J_old)
    #     end
        # visu
        # p1 = plot(xc,[H,H_obs]       ; title="H"     , label=["H" "H_obs"])
        # p2 = plot(iter_evo,J_evo     ; title="misfit", label="", yaxis=:log10)
        # p3 = plot(xc,[npow,npow_synt]; title="n"      , label=["current" "synthetic"])
        # display(plot(p1,p2,p3;layout=(1,3)))
    # end
    return
end

main()
