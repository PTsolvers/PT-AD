using Enzyme,Plots,Printf,LoopVectorization

function residual!(R,H,npow,dx)
    @inbounds @simd for ix = 2:length(R)-1
        D_l  = 0.5*(H[ix  ]^npow[ix  ]+H[ix-1]^npow[ix-1])
        D_r  = 0.5*(H[ix+1]^npow[ix+1]+H[ix  ]^npow[ix  ])
        ∇H_l = (H[ix  ]-H[ix-1])/dx
        ∇H_r = (H[ix+1]-H[ix  ])/dx
        R[ix] = (D_r*∇H_r - D_l*∇H_l)/dx;
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
    H::A; R::A; dR::A; npow::A
    niter::Int; ncheck::Int; ϵtol::T
    dx::T; dmp::T
end

function ForwardProblem(H,npow,niter,ncheck,ϵtol,dx,dmp)
    R  = similar(H)
    dR = similar(H)
    return ForwardProblem(H,R,dR,npow,niter,ncheck,ϵtol,dx,dmp)
end

function solve!(problem::ForwardProblem)
    (;H,R,dR,npow,niter,ncheck,ϵtol,dx,dmp) = problem
    nx = length(H)
    dt = dx/6
    R .= 0; dR .= 0
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        residual!(dR,H,npow,dx)
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

mutable struct AdjointProblem{T<:Real,A<:AbstractArray{T}}
    Ψ::A; R::A; dR; tmp1::A; tmp2::A; ∂J_∂H::A; H::A; H_obs::A; npow::A
    niter::Int; ncheck::Int; ϵtol::T
    dx::T; dmp::T
end

function AdjointProblem(H,H_obs,npow,dx,dmp,niter,ncheck,ϵtol)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    return AdjointProblem(Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,npow,niter,ncheck,ϵtol,dx,dmp)
end

function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,npow,niter,ncheck,ϵtol,dx,dmp) = problem
    nx = length(Ψ)
    dt = dx/3
    Ψ .= 0; R .= 0; dR .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        Enzyme.autodiff(residual!,Duplicated(tmp1,tmp2),Duplicated(H,dR),Const(npow),Const(dx))
        @. R  = R*(1.0-dmp/nx) + dt*dR
        @. Ψ += dt*R
        Ψ[1] = 0; Ψ[end] = 0
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

function cost_gradient!(Jn,problem::AdjointProblem)
    (;Ψ,tmp1,tmp2,H,npow,dx) = problem
    tmp1 .= .-Ψ; Jn .= 0.0
    Enzyme.autodiff(residual!,Duplicated(tmp2,tmp1),Const(H),Duplicated(npow,Jn),Const(dx))
    Jn[1] = Jn[2]; Jn[end] = Jn[end-1]
    return
end

@views function main()
    # physics
    lx           = 20.0
    npows0       = 3.0
    npowi0       = 1.0
    # numerics
    nx           = 101
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
    H            = collect(1.0 .- 0.5.*xc./lx)
    H_obs        = copy(H)
    H_ini        = copy(H)
    npow_synt    = fill(npows0,nx)
    npow_init    = fill(npowi0,nx)
    npow         = copy(npow_init)
    Jn           = zeros(nx) # cost function gradient
    fwd_problem  = ForwardProblem(H,npow,niter,ncheck,ϵtol,dx,dmp)
    adj_problem  = AdjointProblem(H,H_obs,npow,dx,dmp_adj,niter,ncheck,ϵtol)
    synt_problem = ForwardProblem(H_obs,npow_synt,niter,ncheck,ϵtol,dx,dmp)
    # action
    println("  generating synthetic data...")
    solve!(synt_problem)
    solve!(fwd_problem)
    println("  gradient descent")
    γ = γ0
    J_old = cost(H,H_obs)
    J_evo = Float64[]; iter_evo = Int[]
    for gd_iter = 1:gd_niter
        npow_init .= npow
        # adjoint solve
        solve!(adj_problem)
        # compute cost function gradient
        cost_gradient!(Jn,adj_problem)
        # line search
        for bt_iter = 1:bt_niter
            @. npow -= γ*Jn
            fwd_problem.H .= H_ini
            solve!(fwd_problem)
            J_new = cost(H,H_obs)
            if J_new < J_old
                γ *= 1.5
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
