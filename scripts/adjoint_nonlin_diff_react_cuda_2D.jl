using Enzyme,Plots,Printf,CUDA

@inline function mypow(a,n::Integer)
    tmp = a*a
    if n == 3
        return tmp*a
    elseif n == 4
        return tmp*tmp
    elseif n == 5
        return tmp*tmp*a
    end
    return NaN
end

@inline ∇Sx(H,B,dx,ix,iy)         = (B[ix,iy]-B[ix-1,iy])/dx + (H[ix,iy]-H[ix-1,iy])/dx
@inline ∇Sy(H,B,dy,ix,iy)         = (B[ix,iy]-B[ix,iy-1])/dy + (H[ix,iy]-H[ix,iy-1])/dy
@inline Dcoef(H,npow,ix,iy)       = 1e-2*mypow(H[ix,iy],(npow+2)) + mypow(H[ix,iy],npow)
@inline D_upw_x(∇Sx,H,npow,ix,iy) = ∇Sx > 0.0 ? Dcoef(H,npow,ix,iy) : Dcoef(H,npow,ix-1,iy)
@inline D_upw_y(∇Sy,H,npow,ix,iy) = ∇Sy > 0.0 ? Dcoef(H,npow,ix,iy) : Dcoef(H,npow,ix,iy-1)


function residual!(R,dτ,H,B,ELA,β,npow,min_dxy,dx,dy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix >= 2 && ix <= size(H,1)-1 && iy >= 2 && iy <= size(H,2)-1
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

        divQ = (Dx_r*∇Sx_r-Dx_l*∇Sx_l)/dx + (Dy_u*∇Sy_u-Dy_d*∇Sy_d)/dy
        mb   = min(β[ix,iy]*(B[ix,iy] + H[ix,iy] - ELA[ix,iy]),0.01)

        R[ix,iy]  = divQ + mb
        dτ[ix,iy] = min(0.02*min_dxy, 0.02*min_dxy/(1e-5 + sqrt(0.25*(Dx_l + Dx_r + Dy_d + Dy_u))))
    end
    return
end

function mask_by_field!(A,Eq,Less)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix >= 1 && ix <= size(A,1) && iy >= 1 && iy <= size(A,2)
        if Eq[ix,iy] == 0.0 && Less[ix,iy] < 0.0
            A[ix,iy] = 0.0
        end
    end
    return
end

function grad_residual_H!(dR,dτ,H,B,ELA,β,npow,tmp1,tmp2,min_dxy,dx,dy)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp1,tmp2),Const(dτ),Duplicated(H,dR),Const(B),Const(ELA),Const(β),Const(npow),Const(min_dxy),Const(dx),Const(dy))
    return
end

function grad_residual_β!(Jn,dτ,H,B,ELA,β,npow,tmp1,tmp2,min_dxy,dx,dy)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp2,tmp1),Const(dτ),Const(H),Const(B),Const(ELA),Duplicated(β,Jn),Const(npow),Const(min_dxy),Const(dx),Const(dy))
    return
end

function timestep!(H,R,dR,dτ,dmp_nxy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix >= 2 && ix <= size(H,1)-1 && iy >= 2 && iy <= size(H,2)-1
        if H[ix,iy] == 0.0 && dR[ix,iy] < 0.0
            R[ix,iy] = 0.0
        else
            R[ix,iy] = R[ix,iy]*(1.0-dmp_nxy) + dτ[ix,iy]*dR[ix,iy]
            H[ix,iy] = max(0.0, H[ix,iy] + dτ[ix,iy]*R[ix,iy])
        end
    end
    return
end

cost(H,H_obs) = 0.5*sum((H.-H_obs).^2)

mutable struct ForwardProblem{T<:Real,A<:AbstractArray{T}}
    H::A; B::A; ELA::A; R::A; dR::A; dτ::A; β::A; npow::Int
    niter::Int; ncheck::Int; ϵtol::T
    dx::T; dy::T; dmp::T
end

function ForwardProblem(H,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
    R  = similar(H)
    dR = similar(H)
    dτ = similar(H)
    return ForwardProblem(H,B,ELA,R,dR,dτ,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
end

@views function solve!(problem::ForwardProblem)
    (;H,B,ELA,R,dR,dτ,β,npow,niter,ncheck,ϵtol,dx,dy,dmp) = problem
    nx,ny    = size(H)
    dmp_nxy  = dmp/min(nx,ny)
    min_dxy  = min(dx,dy)
    nthreads = (32,16)
    nblocks  = cld.((nx,ny),nthreads)
    R  .= 0; dR .= 0; dτ .= 0
    merr = 2ϵtol; iter = 1
    while merr >= ϵtol && iter < niter
        CUDA.@sync @cuda threads=nthreads blocks=nblocks residual!(dR,dτ,H,B,ELA,β,npow,min_dxy,dx,dy)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks timestep!(H,R,dR,dτ,dmp_nxy)
        if iter % ncheck == 0
            CUDA.@sync @cuda threads=nthreads blocks=nblocks mask_by_field!(dR,H,dR)
            merr = maximum(abs.(dR))
            # display(heatmap(Array(H')))
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
    Ψ::A; R::A; dR::A; dτ::A; tmp1::A; tmp2::A; ∂J_∂H::A; H::A; H_obs::A; B::A; ELA::A; β::A; npow::Int
    niter::Int; ncheck::Int; ϵtol::T
    dx::T; dy::T; dmp::T
end

function AdjointProblem(H,H_obs,B,ELA,β,npow,dx,dy,dmp,niter,ncheck,ϵtol)
    Ψ     = similar(H)
    R     = similar(H)
    dR    = similar(H)
    dτ    = similar(H)
    tmp1  = similar(H)
    tmp2  = similar(H)
    ∂J_∂H = similar(H)
    return AdjointProblem(Ψ,R,dR,dτ,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp)
end

function update_Ψ!(Ψ,R,dR,H,dt,dmp_nxy)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix >= 2 && ix <= size(H,1)-1 && iy >= 2 && iy <= size(H,2)-1
        if H[ix,iy] == 0.0
            R[ix,iy] = 0.0
            Ψ[ix,iy] = 0.0
        else
            R[ix,iy]  = R[ix,iy]*(1.0-dmp_nxy) + dt*dR[ix,iy]
            Ψ[ix,iy] += dt*R[ix,iy]
        end
    end
    if ix == 1 || ix == size(H,1) || iy == 1|| iy == size(H,2)
        Ψ[ix,iy] == 0.0
    end
    return
end

@views function solve!(problem::AdjointProblem)
    (;Ψ,R,dR,dτ,tmp1,tmp2,∂J_∂H,H,H_obs,B,ELA,β,npow,niter,ncheck,ϵtol,dx,dy,dmp) = problem
    nx,ny = size(Ψ)
    dt = min(dx,dy)/6
    dmp_nxy = dmp/min(nx,ny)
    min_dxy = min(dx,dy)
    Ψ .= 0; R .= 0; dR .= 0
    @. ∂J_∂H = H - H_obs
    merr = 2ϵtol; iter = 1
    nthreads = (32,16)
    nblocks  = cld.((nx,ny),nthreads)
    while merr >= ϵtol && iter < niter
        dR .= .-∂J_∂H; tmp2 .= Ψ
        CUDA.@sync @cuda threads=nthreads blocks=nblocks grad_residual_H!(dR,dτ,H,B,ELA,β,npow,tmp1,tmp2,min_dxy,dx,dy)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_Ψ!(Ψ,R,dR,H,dt,dmp_nxy)
        if iter % ncheck == 0
            merr = maximum(abs.(R[2:end-1,2:end-1]))
            # display(heatmap(Array(Ψ')))
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
    (;Ψ,tmp1,tmp2,H,B,ELA,β,dτ,npow,dx,dy) = problem
    nx,ny   = size(H)
    min_dxy = min(dx,dy)
    tmp1 .= .-Ψ; Jn .= 0.0
    nthreads = (32,16)
    nblocks  = ceil.(Int,(nx,ny)./nthreads)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks grad_residual_β!(Jn,dτ,H,B,ELA,β,npow,tmp1,tmp2,min_dxy,dx,dy)
    Jn[[1,end],:] .= Jn[[2,end-1],:]; Jn[:,[1,end]] .= Jn[:,[2,end-1]]
    return
end

function laplacian!(A2,A)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if ix >= 2 && ix <= size(A,1)-1 && iy >= 2 && iy <= size(A,2)-1
        A2[ix,iy] = A[ix,iy] + 0.125*(A[ix-1,iy] + A[ix+1,iy] + A[ix,iy-1] + A[ix,iy+1] - 4.0*A[ix,iy])
    end
    return
end

function smooth!(A,A2,nsm)
    nthreads = (32,32)
    nblocks  = cld.(size(A),nthreads)
    for _ = 1:nsm
        CUDA.@sync @cuda threads=nthreads blocks=nblocks laplacian!(A2,A)
        A,A2 = A2,A
    end
    return
end

@views function main()
    # physics
    lx,ly        = 100.0,100.0
    npow         = 3
    β0           = 0.25
    # numerics
    nx           = 256
    ny           = ceil(Int,nx*ly/lx)
    niter        = 500max(nx,ny)
    ncheck       = 1min(nx,ny)
    ϵtol         = 1e-4
    gd_ϵtol      = 1e-3
    dmp          = 0.6
    dmp_adj      = 1.6
    gd_niter     = 100
    bt_niter     = 3
    γ0           = 1.0e0
    # preprocessing
    dx,dy        = lx/nx,ly/ny
    xc           = LinRange(dx/2,lx-dx/2,nx)
    yc           = LinRange(dy/2,ly-dy/2,ny)
    # init
    H            = CUDA.zeros(Float64,nx,ny) .+ 0.0; H[[1,end],:] .= 0; H[:,[1,end]] .= 0
    S            = CUDA.zeros(Float64,nx,ny)
    H_obs        = copy(H)
    H_ini        = copy(H)
    hatfun1      = CuArray(@. 1.0*exp((-(xc/lx - 0.5)^2 - (yc'/lx - 0.5*ly/lx)^2) / 0.25))
    hatfun2      = CuArray(@. 2.5*exp((-(xc/lx - 0.5)^2 - (yc'/lx - 0.5*ly/lx)^2) / 0.025))
    B            = hatfun1 .+ hatfun2
    ELA          = CuArray(@. 2.0 + 1.0*(xc/lx - 0.5) + 0.0*yc')
    β_synt       = CuArray(@. β0 - 0.0 * atan(xc/lx) - 0.015 * atan(yc'/lx))
    β_ini        = 0.4 .* β_synt
    β            = copy(β_ini)
    β2           = similar(β)
    Jn           = CUDA.zeros(Float64,nx,ny) # cost function gradient
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
    J_old = sqrt(cost(H,H_obs)*dx*dy)
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
            smooth!(β,β2,5)
            β[[1,end],:] .= β[[2,end-1],:]; β[:,[1,end]] .= β[:,[2,end-1]]
            fwd_problem.H .= H_ini
            solve!(fwd_problem)
            J_new = sqrt(cost(H,H_obs)*dx*dy)
            if J_new < J_old
                γ *= 1.1
                J_old = J_new
                break
            else
                β .= β_ini .+ 0.025.*(CUDA.rand(nx,ny) .- 0.5)
                γ = max(γ*0.5, 0.1*γ0)
            end
        end
        # visu
        push!(iter_evo,gd_iter); push!(J_evo,J_old/J_ini)
        @. S = B + H; S[H .== 0] .= NaN
        p1 = heatmap(xc,yc,Array(S')    ; title="S"    , aspect_ratio=1)
        p2 = heatmap(xc,yc,Array(β')    ; title="β"    , aspect_ratio=1)
        p3 = heatmap(xc,yc,Array(S_obs'); title="S_obs", aspect_ratio=1)
        p4 = plot(iter_evo,J_evo; title="misfit", label="", yaxis=:log10,linewidth=2)
        display(plot(p1,p2,p3,p4;layout=(2,2),size=(980,980)))
        # check convergence
        if J_old/J_ini < gd_ϵtol
            @printf("  gradient descent converged, misfit = %.1e\n", J_old)
            break
        else
            @printf("  #iter = %d, misfit = %.1e\n", gd_iter, J_old)
        end
    end
    return
end

main()
