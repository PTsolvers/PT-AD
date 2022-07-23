using Enzyme,Plots,Printf,LoopVectorization

@inline ∇Sx(H,B,dx,ix,iy)         = (B[ix,iy]-B[ix-1,iy])/dx + (H[ix,iy]-H[ix-1,iy])/dx
@inline ∇Sy(H,B,dy,ix,iy)         = (B[ix,iy]-B[ix,iy-1])/dy + (H[ix,iy]-H[ix,iy-1])/dy
@inline Dcoef(H,npow,ix,iy)       = H[ix,iy]^(npow+2) + H[ix,iy]^npow
@inline D_upw_x(∇Sx,H,npow,ix,iy) = ∇Sx > 0.0 ? Dcoef(H,npow,ix,iy) : Dcoef(H,npow,ix-1,iy)
@inline D_upw_y(∇Sy,H,npow,ix,iy) = ∇Sy > 0.0 ? Dcoef(H,npow,ix,iy) : Dcoef(H,npow,ix,iy-1)

function residual!(R,H,B,ELA,β,npow,dx,dy)
    @inbounds for iy=2:size(H,2)-1, ix = 2:size(H,1)-1
        ∇Sx_l = ∇Sx(H,B,dx,ix  ,iy  )
        ∇Sx_r = ∇Sx(H,B,dx,ix+1,iy  )
        ∇Sx_d = 0.25*(∇Sx_l + ∇Sx_r + ∇Sx(H,B,dx,ix,iy-1)+∇Sx(H,B,dx,ix+1,iy-1))
        ∇Sx_u = 0.25*(∇Sx_l + ∇Sx_r + ∇Sx(H,B,dx,ix,iy+1)+∇Sx(H,B,dx,ix+1,iy+1))

        ∇Sy_d = ∇Sy(H,B,dy,ix  ,iy  )
        ∇Sy_u = ∇Sy(H,B,dy,ix  ,iy+1)
        ∇Sy_l = 0.25*(∇Sy_d + ∇Sy_u + ∇Sy(H,B,dy,ix-1,iy) + ∇Sy(H,B,dy,ix-1,iy+1))
        ∇Sy_r = 0.25*(∇Sy_d + ∇Sy_u + ∇Sy(H,B,dy,ix+1,iy) + ∇Sy(H,B,dy,ix+1,iy+1))

        Dx_l = D_upw_x(∇Sx_l,H,npow,ix  ,iy  )*(∇Sx_l^2+∇Sy_l^2)
        Dx_r = D_upw_x(∇Sx_r,H,npow,ix+1,iy  )*(∇Sx_r^2+∇Sy_r^2)
        Dy_d = D_upw_y(∇Sy_d,H,npow,ix  ,iy  )*(∇Sx_d^2+∇Sy_d^2)
        Dy_u = D_upw_y(∇Sy_u,H,npow,ix  ,iy+1)*(∇Sx_u^2+∇Sy_u^2)

        R[ix,iy] = (Dx_r*∇Sx_r - Dx_l*∇Sx_l)/dx + (Dy_u*∇Sy_u - Dy_d*∇Sy_d)/dy + min(β[ix,iy]*(B[ix,iy] + H[ix,iy] - ELA[ix,iy]),0.01)
    end
    return
end

function timestep!(dτ,H,B,npow,dx,dy)
    @inbounds for iy = 2:size(H,2)-1, ix = 2:size(H,1)-1
        ∇Sx_l = ∇Sx(H,B,dx,ix  ,iy  )
        ∇Sx_r = ∇Sx(H,B,dx,ix+1,iy  )
        ∇Sx_d = 0.25*(∇Sx_l + ∇Sx_r + ∇Sx(H,B,dx,ix,iy-1)+∇Sx(H,B,dx,ix+1,iy-1))
        ∇Sx_u = 0.25*(∇Sx_l + ∇Sx_r + ∇Sx(H,B,dx,ix,iy+1)+∇Sx(H,B,dx,ix+1,iy+1))

        ∇Sy_d = ∇Sy(H,B,dy,ix  ,iy  )
        ∇Sy_u = ∇Sy(H,B,dy,ix  ,iy+1)
        ∇Sy_l = 0.25*(∇Sy_d + ∇Sy_u + ∇Sy(H,B,dy,ix-1,iy) + ∇Sy(H,B,dy,ix-1,iy+1))
        ∇Sy_r = 0.25*(∇Sy_d + ∇Sy_u + ∇Sy(H,B,dy,ix+1,iy) + ∇Sy(H,B,dy,ix+1,iy+1))

        Dx_l = D_upw_x(∇Sx_l,H,npow,ix  ,iy  )*(∇Sx_l^2+∇Sy_l^2)
        Dx_r = D_upw_x(∇Sx_r,H,npow,ix+1,iy  )*(∇Sx_r^2+∇Sy_r^2)
        Dy_d = D_upw_y(∇Sy_d,H,npow,ix  ,iy  )*(∇Sx_d^2+∇Sy_d^2)
        Dy_u = D_upw_y(∇Sy_u,H,npow,ix  ,iy+1)*(∇Sx_u^2+∇Sy_u^2)

        dτ[ix,iy] = min(0.25, 0.2*min(dx,dy)/(1e-5 + sqrt(0.25*(Dx_l + Dx_r + Dy_d + Dy_u))))
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
    dx::T; dy::T; dmp::T
end

function ForwardProblem(H,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
    R  = similar(H)
    dR = similar(H)
    return ForwardProblem(H,B,ELA,R,dR,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
end

@views function solve!(problem::ForwardProblem)
    (;H,B,ELA,R,dR,β,npow,niter,ncheck,ϵtol,dx,dy,dmp) = problem
    nx,ny = size(H)
    dτ    = zeros(nx,ny)
    R  .= 0; dR .= 0
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        residual!(dR,H,B,ELA,β,npow,dx,dy)
        timestep!(dτ,H,B,npow,dx,dy)
        @. R = R*(1.0-dmp/nx) + dτ*dR
        @. R[H == 0.0 && dR < 0.0] = 0.0
        @. H = max(0.0, H + dτ*R)
        if iter % ncheck == 0
            merr = maximum(abs.(R))
            # display(heatmap(β))
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
    dx::T; dy::T; dmp::T
end

function AdjointProblem(H,H_obs,B,ELA,β,npow,dx,dy,dmp,niter,ncheck,ϵtol)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    return AdjointProblem(Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
end

@views function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp) = problem
    nx,ny = size(Ψ)
    dt = min(dx,dy)/5
    Ψ .= 0; R .= 0; dR .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        Enzyme.autodiff(residual!,Duplicated(tmp1,tmp2),Duplicated(H,dR),Const(B),Const(ELA),Const(β),Const(npow),Const(dx),Const(dy))
        @. R  = R*(1.0-dmp/min(nx,ny)) + dt*dR
        @. Ψ += dt*R
        R[H .== 0.0] .= 0.0
        Ψ[H .== 0.0] .= 0.0
        Ψ[[1,end],:] .= 0; Ψ[:,[1,end]] .= 0
        if iter % ncheck == 0
            merr = maximum(abs.(R[2:end-1,2:end-1]))
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
    (;Ψ,tmp1,tmp2,H,B,ELA,β,npow,dx,dy) = problem
    tmp1 .= .-Ψ; Jn .= 0.0
    Enzyme.autodiff(residual!,Duplicated(tmp2,tmp1),Const(H),Const(B),Const(ELA),Duplicated(β,Jn),Const(npow),Const(dx),Const(dy))
    Jn[[1,end],:] .= Jn[[2,end-1],:]; Jn[:,[1,end]] .= Jn[:,[2,end-1]]
    return
end

@views function main()
    # physics
    lx,ly        = 100.0,100.0
    npow         = 3
    β0           = 0.25
    # numerics
    nx           = 64
    ny           = ceil(Int,nx*ly/lx)
    niter        = 1000max(nx,ny)
    ncheck       = 1min(nx,ny)
    ϵtol         = 1e-5
    gd_ϵtol      = 1e-3
    dmp          = 1.0
    dmp_adj      = 1.5
    gd_niter     = 100
    bt_niter     = 10
    γ0           = 1.0e-2
    # preprocessing
    dx,dy        = lx/nx,ly/ny
    xc           = LinRange(dx/2,lx-dx/2,nx)
    yc           = LinRange(dy/2,ly-dy/2,ny)
    # init
    H            = zeros(nx,ny)# .+ 0.001; H[[1,end],:] .= 0; H[:,[1,end]] .= 0
    S            = zeros(nx,ny)
    H_obs        = copy(H)
    H_ini        = copy(H)
    hatfun1      = @. 1.0*exp((-(xc/lx - 0.5)^2 - (yc'/lx - 0.5*ly/lx)^2) / 0.25)
    hatfun2      = @. 2.5*exp((-(xc/lx - 0.5)^2 - (yc'/lx - 0.5*ly/lx)^2) / 0.025)
    B            = hatfun1 .+ hatfun2
    ELA          = collect(@. 2.0 + 1.0*(xc/lx - 0.5) + 0.0*yc')
    β_synt       = collect(@. β0 - 0.015 * atan(xc/lx) + 0.0*yc')
    β_ini        = 0.4 .* β_synt
    β            = copy(β_ini)
    Jn           = zeros(nx,ny) # cost function gradient
    fwd_problem  = ForwardProblem(H,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
    adj_problem  = AdjointProblem(H,H_obs,B,ELA,β,npow,dx,dy,dmp_adj,niter,ncheck,ϵtol)
    synt_problem = ForwardProblem(H_obs,B,ELA,β_synt,npow,niter,ncheck,ϵtol,dx,dy,dmp)
    # action
    println("  generating synthetic data...")
    solve!(synt_problem)
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
                @. β[2:end-1,2:end-1] += 1/5 * (β[1:end-2,2:end-1] + β[3:end,2:end-1] + β[2:end-1,1:end-2] .+ β[2:end-1,3:end] - 4.0.*β[2:end-1,2:end-1])
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
        p1 = heatmap(xc,yc,S'    ; title="S"    , aspect_ratio=1)
        p2 = heatmap(xc,yc,β'    ; title="β"    , aspect_ratio=1)
        p3 = heatmap(xc,yc,S_obs'; title="S_obs", aspect_ratio=1)
        p4 = plot(iter_evo,J_evo; title="misfit", label="", yaxis=:log10,linewidth=2)
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
