using Enzyme,Plots,Printf

function residual!(R,H,npow,dx)
    @inbounds @simd for ix = 2:length(R)-1
        R[ix] = (H[ix-1]^(npow[ix-1]+1.0) - 2.0*H[ix]^(npow[ix]+1.0) + H[ix+1]^(npow[ix+1]+1.0))/dx/dx/(npow[ix]+1.0)
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

function adjoint_residual!(R,Ψ,H,H_obs,npow,dx)
    @inbounds @simd for ix = 2:length(R)-1
        R[ix] = H[ix]^npow[ix]*(Ψ[ix-1] - 2.0*Ψ[ix] + Ψ[ix+1])/dx/dx - (H[ix] - H_obs[ix])
    end
    return
end

function cost_gradient!(Jn,Ψ,H,npow,dx)
    @inbounds @simd for ix = 2:length(Jn)-1
        Jn[ix] = H[ix]^npow[ix]*log(H[ix])*0.5*(H[ix+1]-H[ix-1])/dx*0.5*(Ψ[ix+1]-Ψ[ix-1])/dx
    end
    Jn[1] = Jn[2]; Jn[end] = Jn[end-1]
    return
end

@views function main()
    # physics
    lx    = 20.0
    npow0 = 3.0
    # numerics
    nx       = 101
    niter    = 100nx
    nchk     = 5nx
    εtol     = 1e-8
    dmp      = 1/2
    dmp_adj  = 3/2
    dmp_adj2 = 1dmp_adj
    # preprocessing
    dx     = lx/nx
    xc     = LinRange(dx/2,lx-dx/2,nx)
    dt     = dx/6
    # init
    H      = collect(1.0 .- 0.5.*xc./lx)
    H_obs  = copy(H)
    npow_s = fill(npow0  ,nx)
    npow_i = fill(npow0-2,nx)
    R      = zeros(nx)
    R_an   = zeros(nx)
    R_obs  = zeros(nx)
    dR     = zeros(nx)
    dR_an  = zeros(nx)
    dR_obs = zeros(nx)
    R_obs  = zeros(nx)
    ∂J_∂H  = zeros(nx)
    Jn     = zeros(nx)
    Jn_an  = zeros(nx)
    Ψ      = zeros(nx) # adjoint state (discretize-then-optimise)
    Ψ_an   = zeros(nx) # adjoint state (optimise-then-discretize)
    JVP    = zeros(nx) # Jacobian-vector product storage
    dΨdt   = zeros(nx)
    # action
    # forward solve
    println("forward solve...")
    for iter = 1:niter
        residual!(dR_obs,H_obs,npow_s,dx)
        residual!(dR,H        ,npow_i ,dx)
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
    npow = copy(npow_i)
    γ    = 1e1
    for gd_iter = 1:1000
        npow_i .= npow
        # adjoint solve
        println("adjoint solve...")
        @. ∂J_∂H = H - H_obs
        for iter = 1:niter
            # discretize-then-optimise
            JVP .= 0.0; Jn .= Ψ
            Enzyme.autodiff(residual!,Duplicated(R,Jn),Duplicated(H,JVP),Const(npow),Const(dx))
            @. dΨdt = dΨdt*(1.0-dmp_adj2/nx) + dt*(JVP - ∂J_∂H)
            @. Ψ[2:end-1] += dt*dΨdt[2:end-1]
            # optimise-then-discretize
            adjoint_residual!(dR_an,Ψ_an,H,H_obs,npow,dx)
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
        # compute cost function gradient
        JVP .= .-Ψ; Jn .= 0.0;
        Enzyme.autodiff(residual!,Duplicated(R,JVP),Const(H),Duplicated(npow,Jn),Const(dx))
        Jn[1] = Jn[2]; Jn[end] = Jn[end-1]
        cost_gradient!(Jn_an,Ψ_an,H_obs,npow,dx)
        # line search
        H_i  = copy(H)
        for bt_iter = 1:100
            J_old = cost(H,H_obs)
            @. npow -= γ*Jn
            H .= H_i
            println("checking $bt_iter...")
            for iter = 1:niter
                residual!(dR,H,npow,dx)
                @. R  = R*(1.0-dmp/nx) + dt*dR
                @. H += dt*R
                if iter % nchk == 0
                    merr = maximum(abs.(dR))
                    @printf("  #iter/nx = %.1f, err = %.1e\n",iter/nx,merr)
                    if merr < εtol break end
                end
            end
            J = cost(H,H_obs)
            if J < J_old
                γ *= 1.1
                break
            else
                npow .= npow_i
                # break
                γ *= 0.5
            end
            println("done")
        end
        p1 = plot(xc,[H ,H_obs]; title="H" , label=["H" "H_obs"])
        p2 = plot(xc,[Jn,Jn_an]; title="Jn", label=["dto" "otd"])
        p3 = plot(xc,Ψ .-Ψ_an; title="Ψ_dto - Ψ_otd",label="")
        p4 = plot(xc,npow; title="$gd_iter $γ",label="")
        display(plot(p1,p2,p3,p4))
    end

    return
end

main()
