using Enzyme,Plots,Printf,LoopVectorization
using CUDA

function mypow(a,n::Int)
    tmp=a*a
    if n==1 a
    elseif n==2 tmp
    elseif n==3 tmp*a
    elseif n==4 tmp*tmp
    elseif n==5 tmp*tmp*a
    elseif n==6 tmp*tmp*tmp
    else NaN
    end
end

@inbounds function residual!(R,H,npow,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(R)-1
        R[ix] = (mypow(H[ix-1],(npow[ix-1]+1)) - 2.0*mypow(H[ix],(npow[ix]+1)) + mypow(H[ix+1],(npow[ix+1]+1)))/dx/dx/(npow[ix]+1)
    end
    return
end

function cost(H,H_obs)
    J = 0.0
    @turbo for ix ∈ eachindex(H)
        J += (H[ix]-H_obs[ix])^2
    end
    return 0.5*J
end

mutable struct ForwardProblem{T<:Real,A<:AbstractArray{T},B<:AbstractArray{<:Integer}}
    H::A; R::A; dR::A; npow::B
    niter::Int; ncheck::Int; ϵtol::T; threads::Int; blocks::Int
    dx::T; dmp::T
end

function ForwardProblem(H,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp)
    R  = similar(H)
    dR = similar(H)
    return ForwardProblem(H,R,dR,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp)
end

function solve!(problem::ForwardProblem)
    (;H,R,dR,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp) = problem
    nx = length(H)
    dt = dx/6
    R .= 0; dR .= 0
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        @cuda blocks=blocks threads=threads residual!(dR,H,npow,dx); synchronize()
        @. R  = R*(1.0-dmp/nx) + dt*dR
        @. H += dt*R
        if iter % ncheck == 0
            merr = maximum(abs.(dR))
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

mutable struct AdjointProblem{T<:Real,A<:AbstractArray{T},B<:AbstractArray{<:Integer}}
    Ψ::A; R::A; dR; tmp1::A; tmp2::A; ∂J_∂H::A; H::A; H_obs::A; npow::B
    niter::Int; ncheck::Int; ϵtol::T; threads::Int; blocks::Int
    dx::T; dmp::T
end

function AdjointProblem(H,H_obs,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    return AdjointProblem(Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp)
end

function residual_grad!(tmp1,tmp2,H,dR,npow,dx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp1,tmp2),Duplicated(H,dR),Const(npow),Const(dx))
    return
end

function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp) = problem
    nx = length(Ψ)
    dt = dx/3
    Ψ .= 0; R .= 0; dR .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        @cuda blocks=blocks threads=threads residual_grad!(tmp1,tmp2,H,dR,npow,dx); synchronize()
        @. R  = R*(1.0-dmp/nx) + dt*dR
        @. Ψ += dt*R
        Ψ[1:1] .= 0; Ψ[end:end] .= 0
        if iter % ncheck == 0
            merr = maximum(abs.(dR[2:end-1]))
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

function cost_grad!(tmp1,tmp2,H,Jn,npow,dx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp2,tmp1),Const(H),Duplicated(npow,Jn),Const(dx))
    return
end

function cost_gradient!(Jn,problem::AdjointProblem)
    (;Ψ,tmp1,tmp2,H,npow,threads,blocks,dx) = problem
    tmp1 .= .-Ψ; Jn .= 0.0
    @cuda blocks=blocks threads=threads cost_grad!(tmp1,tmp2,H,Jn,npow,dx); synchronize()
    # Jn[1:1] .= Jn[2:2]; Jn[end:end] .= Jn[end-1:end-1]
    return
end

@views function main()
    # physics
    lx           = 20.0
    npows0       = 3
    npowi0       = 1
    # numerics
    nx           = 128
    threads      = 128
    blocks       = cld(nx,threads)
    niter        = 100nx
    ncheck       = 5nx
    ϵtol         = 1e-8
    gd_ϵtol      = 1e-5
    dmp          = 1/2
    dmp_adj      = 3/2
    gd_niter     = 500
    bt_niter     = 10
    γ0           = 1e2
    # preprocessing
    dx           = lx/nx
    xc           = LinRange(dx/2,lx-dx/2,nx)
    # init
    H            = CuArray( collect(1.0 .- 0.5.*xc./lx) )
    H_obs        = copy(H)
    H_ini        = copy(H)
    npow_synt    = CUDA.fill(npows0,nx)
    npow_init    = CUDA.fill(npowi0,nx)
    npow         = copy(npow_init)
    Jn           = CUDA.zeros(Float64,nx) # cost function gradient
    synt_problem = ForwardProblem(H_obs,      npow_synt,niter,ncheck,ϵtol,threads,blocks,dx,dmp    )
    fwd_problem  = ForwardProblem(H    ,      npow     ,niter,ncheck,ϵtol,threads,blocks,dx,dmp    )
    adj_problem  = AdjointProblem(H    ,H_obs,npow     ,niter,ncheck,ϵtol,threads,blocks,dx,dmp_adj)
    # action
    println("  generating synthetic data...")
    solve!(synt_problem)
    println("  done.")
    solve!(fwd_problem)
    println("  gradient descent")
    γ = γ0
    J_old = cost(Array(H),Array(H_obs)) # CPU function for now
    J_evo = Float64[]; iter_evo = Int[]
    for gd_iter = 1:gd_niter
        npow_init .= npow
        # adjoint solve
        solve!(adj_problem)
        # compute cost function gradient
        cost_gradient!(Jn,adj_problem)
        error("stop")
        # line search
        for bt_iter = 1:bt_niter
            @. npow -= γ*Jn
            fwd_problem.H .= H_ini
            solve!(fwd_problem)
            J_new = cost(Array(H),Array(H_obs)) # CPU function for now
            if J_new < J_old
                γ *= 1.2
                J_old = J_new
                break
            else
                npow .= npow_init
                γ *= 0.5
            end
        end
        push!(iter_evo,gd_iter); push!(J_evo,J_old)
        if J_old < gd_ϵtol
            @printf("  gradient descent converged, misfit = %.1e\n", J_old)
            break
        else
            @printf("  #iter = %d, misfit = %.1e\n", gd_iter, J_old)
        end
        # visu
        p1 = plot(xc,[H,H_obs]       ; title="H"     , label=["H" "H_obs"])
        p2 = plot(iter_evo,J_evo     ; title="misfit", label="", yaxis=:log10)
        p3 = plot(xc,[npow,npow_synt]; title="n"      , label=["current" "synthetic"])
        display(plot(p1,p2,p3;layout=(1,3)))
    end
    return
end

main()
