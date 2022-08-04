using Enzyme,Plots,Printf,LoopVectorization
using CUDA

@inline function mypow(a,n::Int)
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

@inbounds function residual!(R,H,B,ELA,β,npow::Int,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(R)-1
        ∇S_l  = (B[ix  ]-B[ix-1])/dx + (H[ix  ]-H[ix-1])/dx
        ∇S_r  = (B[ix+1]-B[ix  ])/dx + (H[ix+1]-H[ix  ])/dx
        D_l   = ((∇S_l>0.0) ? (mypow(H[ix  ],npow+2)*1e-4 + mypow(H[ix  ],npow)) : (mypow(H[ix-1],npow+2) + mypow(H[ix-1],npow)))*mypow(∇S_l,2)
        D_r   = ((∇S_r>0.0) ? (mypow(H[ix+1],npow+2)*1e-4 + mypow(H[ix+1],npow)) : (mypow(H[ix  ],npow+2) + mypow(H[ix  ],npow)))*mypow(∇S_r,2)
        R[ix] = (D_r*∇S_r - D_l*∇S_l)/dx + min(β[ix]*(B[ix] + H[ix] - ELA[ix]),0.01)
    end
    return
end

@inbounds function timestep!(dτ,H,B,npow::Int,dτfac,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(H)-1
        ∇S_l   = (B[ix  ]-B[ix-1])/dx + (H[ix  ]-H[ix-1])/dx
        ∇S_r   = (B[ix+1]-B[ix  ])/dx + (H[ix+1]-H[ix  ])/dx
        D_l    = ((∇S_l>0.0) ? (H[ix  ]^(npow+2)*1e-4 + H[ix  ]^npow) : (∇S_l<0.0)*(H[ix-1]^(npow+2) + H[ix-1]^npow))*∇S_l^2
        D_r    = ((∇S_r>0.0) ? (H[ix+1]^(npow+2)*1e-4 + H[ix+1]^npow) : (∇S_r<0.0)*(H[ix  ]^(npow+2) + H[ix  ]^npow))*∇S_r^2
        dτ[ix] = dτfac*min(0.05, 0.05*dx/(1e-5 + sqrt(0.5*(D_l + D_r))))
    end
    return
end

mutable struct ForwardProblem{T<:Real,A<:AbstractArray{T}}
    H::A; B::A; ELA::A; R::A; dR::A; β::A; npow::Int
    Err::A; dτ::A; niter::Int; ncheck::Int; ϵtol::T; threads::Int; blocks::Int
    dx::T; dmp::T; dτfac::T
end

function ForwardProblem(H,B,ELA,β,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac)
    R   = similar(H)
    dR  = similar(H)
    Err = similar(H)
    dτ  = similar(H)
    return ForwardProblem(H,B,ELA,R,dR,β,npow,Err,dτ,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac)
end

@views function solve!(problem::ForwardProblem)
    (;H,B,ELA,R,dR,β,npow,Err,dτ,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac) = problem
    nx  = length(H)
    R  .= 0; dR .= 0; Err .= 0; dτ .= 0
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        Err .= H
        @cuda blocks=blocks threads=threads residual!(dR,H,B,ELA,β,npow,dx); synchronize()
        @cuda blocks=blocks threads=threads timestep!(dτ,H,B,npow,dτfac,dx); synchronize()
        @. R = R*(1.0-dmp/nx) + dτ*dR
        @. H = max(0.0, H + dτ*R)
        if iter % ncheck == 0
            @. Err -= H
            merr = maximum(abs.(Err))
            (isfinite(merr) && merr>0) || error("forward solve failed") 
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
    Err::A; niter::Int; ncheck::Int; ϵtol::T; threads::Int; blocks::Int
    dx::T; dmp::T; dτfac::T
end

function AdjointProblem(H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    Err   = similar(H)
    return AdjointProblem(Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,Err,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac)
end

function residual_grad!(tmp1,tmp2,H,dR,B,ELA,β,npow,dx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp1,tmp2),Duplicated(H,dR),Const(B),Const(ELA),Const(β),Const(npow),Const(dx))
    return
end

@views function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,Err,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac) = problem
    nx = length(Ψ)
    dt = dx/5*dτfac
    Ψ .= 0; R .= 0; dR .= 0; Err .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        Err .= Ψ
        @cuda blocks=blocks threads=threads residual_grad!(tmp1,tmp2,H,dR,B,ELA,β,npow,dx); synchronize()
        @. R  = R*(1.0-dmp/nx) + dt*dR
        @. Ψ += dt*R
        Ψ[H .== 0.0] .= 0.0
        Ψ[1:1] .= 0; Ψ[end:end] .= 0
        if iter % ncheck == 0
            @. Err -= Ψ
            merr = maximum(abs.(Err))
            (isfinite(merr) && merr>0) || error("adjoint solve failed") 
        end
        iter += 1
    end
    if iter == niter && merr >= ϵtol
        error("adjoint solve not converged")
    end
    @printf("    adjoint solve converged: #iter/nx = %.1f, err = %.1e\n",iter/nx,merr)
    return
end

function cost_grad!(tmp1,tmp2,H,B,ELA,β,Jn,npow,dx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp2,tmp1),Const(H),Const(B),Const(ELA),Duplicated(β,Jn),Const(npow),Const(dx))
    return
end

function cost_gradient!(Jn,problem::AdjointProblem)
    (;Ψ,tmp1,tmp2,H,B,ELA,β,npow,threads,blocks,dx) = problem
    tmp1 .= .-Ψ; Jn .= 0.0
    @cuda blocks=blocks threads=threads cost_grad!(tmp1,tmp2,H,B,ELA,β,Jn,npow,dx); synchronize()
    Jn[1:1] .= Jn[2:2]; Jn[end:end] .= Jn[end-1:end-1]
    return
end

@inbounds function cost!(J2,H,H_obs)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix<=length(J2)
        J2[ix] = (H[ix]-H_obs[ix])*(H[ix]-H_obs[ix])
    end
    return
end

function cost(J2,H,H_obs,threads,blocks)
    @cuda blocks=blocks threads=threads cost!(J2,H,H_obs); synchronize()
    return sum(J2)
end

@inbounds function smooth!(A)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(A)-1
        A[ix] += 1/5 * (A[ix-1] - 2.0*A[ix] + A[ix+1])
    end
    return
end

function smooth(A,nsm,threads,blocks)
    for ism = 1:nsm
        @cuda blocks=blocks threads=threads smooth!(A); synchronize()
    end
    return
end

@views function main()
    # physics
    lx           = 100.0
    npow         = 3
    β0           = 0.5
    # numerics
    nx           = 64*4#*16*2
    threads      = 512
    blocks       = cld(nx,threads)
    niter        = 1000nx
    ncheck       = 1nx
    ϵtol         = 1e-5
    gd_ϵtol      = 1e-3
    dmp          = 1.0
    dmp_adj      = 1.5
    dτfac        = 0.9 # 1024->0.7, 2048->0.4
    gd_niter     = 100
    bt_niter     = 10
    γ0           = 1e1
    # preprocessing
    dx           = lx/nx
    xc           = LinRange(dx/2,lx-dx/2,nx)
    # init
    # H            = 4.0 .* CUDA.ones(Float64,nx)
    H            = CuArray( 3.0 .* exp.(.-(xc./lx .- 0.5).^2 ./ 0.04) )
    S            = CUDA.zeros(Float64,nx) # visu
    βv           = CUDA.zeros(Float64,nx) # visu
    H[[1,end]]  .= 0.0
    H_obs        = copy(H)
    H_ini        = copy(H)
    B            = CuArray( 1.0 .* exp.(.-(xc./lx .- 0.5).^2 ./ 0.25) .+ 2.5 .* exp.(.-(xc./lx .-0.5).^2 ./ 0.025) )
    ELA          = CuArray( collect(2.0 .+ 0.5.*(xc./lx .- 0.5)) )
    # β_synt       = CuArray( collect(β0 .- 0.04 .* atan.(xc./lx .- 0.5)) )
    # β_ini        = CuArray( collect(β0 .+ 0.003 .* atan.(xc./lx .- 0.5)) )
    β_synt       = CuArray( clamp.(β0 .+ 0.04 .* (1.0 .- 2.0 .* 10.0 .* abs.((2.0 .* xc/lx .- 1.0)./10.0)), 0.495, 0.51) )
    β_ini        = CuArray( clamp.(β0 .+ 0.005 .* (1.0 .- 2.0 .* 10.0 .* abs.((2.0 .* xc/lx .- 1.0)./10.0)), 0.495, 0.51) )
    # display(plot(xc,[β_synt,β_ini])); error("stop")
    β            = copy(β_ini)
    Jn           = CUDA.zeros(Float64,nx) # cost function gradient
    J2           = CUDA.zeros(Float64,nx) # tmp storage
    synt_problem = ForwardProblem(H_obs,  B,ELA,β_synt,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac)
    fwd_problem  = ForwardProblem(H,      B,ELA,β     ,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp,dτfac)
    adj_problem  = AdjointProblem(H,H_obs,B,ELA,β     ,npow,niter,ncheck,ϵtol,threads,blocks,dx,dmp_adj,dτfac)
    # action
    println("  generating synthetic data (nx=$nx)...")
    solve!(synt_problem)
    println("  done.")
    solve!(fwd_problem)
    println("  gradient descent")
    S_obs = B .+ H_obs; S_obs[H_obs .== 0] .= NaN
    γ = γ0
    # J_old = cost(Array(H),Array(H_obs))
    J_old = cost(J2,H,H_obs,threads,blocks)
    J_ini = J_old
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
            smooth(β,5,threads,blocks)
            fwd_problem.H .= H_ini
            solve!(fwd_problem)
            J_new = cost(J2,H,H_obs,threads,blocks)
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
        push!(iter_evo,gd_iter); push!(J_evo,J_old/J_ini)
        @. S = B + H; S[H .== 0] .= NaN
        @. βv = β; #βv[H .== 0] .= NaN
        p1 = plot(xc,[Array(B),Array(S),Array(S_obs)]; title="S"     , label=["B" "S" "S_obs"]      , ylim=(0,Inf), aspect_ratio=2, line = (2, [:solid :solid :dash]))
        p2 = plot(xc,[Array(βv),Array(β_synt)]       ; title="β"     , label=["current" "synthetic"], linewidth=2)
        p3 = plot(xc,Array(ELA)                      ; title="ELA"   , label=""                     , linewidth=2)
        p4 = plot(iter_evo,J_evo                     ; title="misfit", label=""                     , yaxis=:log10 ,linewidth=2)
        display(plot(p1,p2,p3,p4;layout=(2,2),size=(980,980)))
        # check convergence
        if J_old/J_ini < gd_ϵtol
            @printf("  gradient descent converged, misfit = %.1e\n", J_old/J_ini)
            break
        else
            @printf("  #iter = %d, misfit = %.1e\n", gd_iter, J_old/J_ini)
        end
    end
    return
end

main()
