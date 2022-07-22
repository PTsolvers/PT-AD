using Enzyme,Plots,Printf,LoopVectorization

function residual!(R,H,B,ELA,β,npow,dx)
    @inbounds @simd for ix = 2:length(R)-1
        ∇S_l  = (B[ix  ]-B[ix-1])/dx + (H[ix  ]-H[ix-1])/dx
        ∇S_r  = (B[ix+1]-B[ix  ])/dx + (H[ix+1]-H[ix  ])/dx
        D_l   = ((∇S_l>0.0)*(H[ix  ]^(npow+2) + H[ix  ]^npow) + (∇S_l<0.0)*(H[ix-1]^(npow+2) + H[ix-1]^npow))*abs(∇S_l)^2
        D_r   = ((∇S_r>0.0)*(H[ix+1]^(npow+2) + H[ix+1]^npow) + (∇S_r<0.0)*(H[ix  ]^(npow+2) + H[ix  ]^npow))*abs(∇S_r)^2
        R[ix] = (D_r*∇S_r - D_l*∇S_l)/dx + min(β[ix]*(B[ix] + H[ix] - ELA[ix]),0.01)
    end
    return
end

function timestep!(dτ,H,B,npow,dx)
    @inbounds @simd for ix = 2:length(H)-1
        ∇S_l  = (B[ix  ]-B[ix-1])/dx + (H[ix  ]-H[ix-1])/dx
        ∇S_r  = (B[ix+1]-B[ix  ])/dx + (H[ix+1]-H[ix  ])/dx
        D_l   = ((∇S_l>0.0)*(H[ix  ]^(npow+2) + H[ix  ]^npow) + (∇S_l<0.0)*(H[ix-1]^(npow+2) + H[ix-1]^npow))*abs(∇S_l)^2
        D_r   = ((∇S_r>0.0)*(H[ix+1]^(npow+2) + H[ix+1]^npow) + (∇S_r<0.0)*(H[ix  ]^(npow+2) + H[ix  ]^npow))*abs(∇S_r)^2
        dτ[ix] = min(0.05, 0.05/(1e-5 + sqrt(0.5*(D_l + D_r))))
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

mutable struct ForwardProblem{T<:Real,A<:AbstractArray{T}}
    H::A; B::A; ELA::A; R::A; dR::A; β::A; npow::Int
    niter::Int; ncheck::Int; ϵtol::T
    dx::T; dmp::T
end

function ForwardProblem(H,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dmp)
    R  = similar(H)
    dR = similar(H)
    return ForwardProblem(H,B,ELA,R,dR,β,npow,niter,ncheck,ϵtol,dx,dmp)
end

@views function solve!(problem::ForwardProblem)
    (;H,B,ELA,R,dR,β,npow,niter,ncheck,ϵtol,dx,dmp) = problem
    nx  = length(H)
    dτ  = zeros(nx)
    R  .= 0; dR .= 0
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        residual!(dR,H,B,ELA,β,npow,dx)
        timestep!(dτ,H,B,npow,dx)
        @. R = R*(1.0-dmp/nx) + dτ*dR
        @. R[H == 0.0 && dR < 0.0] = 0.0
        @. H = max(0.0, H + dτ*R)
        if iter % ncheck == 0
            merr = maximum(abs.(R))
            # display(plot(H))
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
    Ψ::A; R::A; dR; tmp1::A; tmp2::A; ∂J_∂H::A; H::A; H_obs::A; B::A; ELA::A; β::A; npow::Int
    niter::Int; ncheck::Int; ϵtol::T
    dx::T; dmp::T
end

function AdjointProblem(H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dmp)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    return AdjointProblem(Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dmp)
end

@views function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dmp) = problem
    nx = length(Ψ)
    dt = dx/5
    Ψ .= 0; R .= 0; dR .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        Enzyme.autodiff(residual!,Duplicated(tmp1,tmp2),Duplicated(H,dR),Const(B),Const(ELA),Const(β),Const(npow),Const(dx))
        @. R  = R*(1.0-dmp/nx) + dt*dR
        @. Ψ += dt*R
        R[H .== 0.0] .= 0.0
        Ψ[H .== 0.0] .= 0.0
        Ψ[1] = 0; Ψ[end] = 0
        if iter % ncheck == 0
            merr = maximum(abs.(R[2:end-1]))
            # display(plot(Ψ))
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
    (;Ψ,tmp1,tmp2,H,B,ELA,β,npow,dx) = problem
    tmp1 .= .-Ψ; Jn .= 0.0
    Enzyme.autodiff(residual!,Duplicated(tmp2,tmp1),Const(H),Const(B),Const(ELA),Duplicated(β,Jn),Const(npow),Const(dx))
    Jn[1] = Jn[2]; Jn[end] = Jn[end-1]
    return
end

@views function main()
    # physics
    lx           = 100.0
    npow         = 3
    β0           = 0.25
    # numerics
    nx           = 128
    niter        = 1000nx
    ncheck       = 1nx
    ϵtol         = 1e-5
    gd_ϵtol      = 1e-3
    dmp          = 1.0
    dmp_adj      = 1.5
    gd_niter     = 100
    bt_niter     = 10
    γ0           = 1.0e-2
    # preprocessing
    dx           = lx/nx
    xc           = LinRange(dx/2,lx-dx/2,nx)
    # init
    H            = zeros(nx)
    S            = zeros(nx)
    H[[1,end]]  .= 0.0
    H_obs        = copy(H)
    H_ini        = copy(H)
    B            = 1.0 .* exp.(.-(xc./lx .- 0.5).^2 ./ 0.25) .+ 2.5 .* exp.(.-(xc./lx .-0.5).^2 ./ 0.025)
    ELA          = collect(2.0 .+ 0.5.*(xc./lx .- 0.5))
    β_synt       = collect(β0 .- 0.015 .* atan.(xc./lx))
    β_ini        = 0.4 .* β_synt
    β            = copy(β_ini)
    Jn           = zeros(nx) # cost function gradient
    synt_problem = ForwardProblem(H_obs,  B,ELA,β_synt,npow,niter,ncheck,ϵtol,dx,dmp)
    fwd_problem  = ForwardProblem(H,      B,ELA,β     ,npow,niter,ncheck,ϵtol,dx,dmp)
    adj_problem  = AdjointProblem(H,H_obs,B,ELA,β     ,npow,niter,ncheck,ϵtol,dx,dmp_adj)
    # action
    println("  generating synthetic data...")
    solve!(synt_problem)
    println("  done.")
    solve!(fwd_problem)
    println("  gradient descent")
    S_obs = B .+ H_obs; S_obs[H_obs .== 0] .= NaN
    γ = γ0
    J_old = cost(H,H_obs)
    J_evo = Float64[]; iter_evo = Int[]
    for gd_iter = 1:gd_niter
        β_ini .= β
        # adjoint solve
        solve!(adj_problem)
        # compute cost function gradient
        cost_gradient!(Jn,adj_problem)
        # line search
        for bt_iter = 1:bt_niter
            @. β -= γ*Jn
            for ism = 1:5
                β[2:end-1] .+= 1/5 .* (β[1:end-2] - 2.0.*β[2:end-1] .+ β[3:end])
            end
            fwd_problem.H .= H_ini
            solve!(fwd_problem)
            J_new = cost(H,H_obs)
            if J_new < J_old
                γ *= 1.1
                J_old = J_new
                break
            else
                β .= β_ini
                γ *= 0.5
            end
        end
        # visu
        push!(iter_evo,gd_iter); push!(J_evo,J_old)
        @. S = B + H; S[H .== 0] .= NaN
        p1 = plot(xc,[B,S,S_obs]; title="S"     , label=["B" "S" "S_obs"]      , ylim=(0,Inf), aspect_ratio=2, line = (2, [:solid :solid :dash]))
        p2 = plot(xc,[β,β_synt] ; title="β"     , label=["current" "synthetic"], linewidth=2)
        p3 = plot(xc,ELA        ; title="ELA"   , label=""                     , linewidth=2)
        p4 = plot(iter_evo,J_evo; title="misfit", label=""                     , yaxis=:log10  ,linewidth=2)
        display(plot(p1,p2,p3,p4;layout=(2,2),size=(980,980)))
        # check convergence
        if J_old < gd_ϵtol
            @printf("  gradient descent converged, misfit = %.1e\n", J_old)
            break
        else
            @printf("  #iter = %d, misfit = %.1e\n", gd_iter, J_old)
        end
    end
    return
end

main()
