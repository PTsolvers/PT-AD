const USE_GPU  = false  # Use GPU? If this is set false, then the CUDA packages do not need to be installed! :)
const GPU_ID   = 0
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
    CUDA.device!(GPU_ID) # select GPU
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, Statistics, LinearAlgebra

@parallel function err_1!(Err, H)
    @all(Err) = @all(H)
    return
end 

@parallel function err_2!(Err, H)
    @all(Err) = @all(Err) - @all(H)
    return
end 

@parallel function compute_dS!(H_avg, dSdx, dSdy, H, B, dx, dy)
    @all(H_avg) = @av(H)
    @all(dSdx)  = @d_xa(B)/dx + @d_xa(H)/dx
    @all(dSdy)  = @d_ya(B)/dy + @d_ya(H)/dy
    return
end

@parallel function compute_2!(grdS2, D, H_avg, dSdx, dSdy, a1, a2, npow)
    @all(grdS2) = @av_ya(dSdx)*@av_ya(dSdx) + @av_xa(dSdy)*@av_xa(dSdy)  # DEBUG: |∇S|^2 from Visnjevic et al. 2018 eq. (A5) should be |∇S|^(n-1). In the code |∇S| = sqrt(dS/dx^2 + dS/dy^2), and |∇S|^(n-1) = (sqrt(dS/dx^2 + dS/dy^2))^(npow-1)
    @all(D)     = (a1*@all(H_avg)^(npow+2) + a2*@all(H_avg)^npow)*@all(grdS2)
    return
end

@parallel function compute_3!(qHx, qHy, D, B, H, dx, dy)
    @all(qHx)  = -@av_ya(D)*(@d_xi(B)/dx + @d_xi(H)/dx)
    @all(qHy)  = -@av_xa(D)*(@d_yi(B)/dy + @d_yi(H)/dy)
    return
end

@parallel_indices (ix,iy) function residual!(RH2,H,B,Ela,β,a1,a2,npow,c,dx,dy)
    nx, ny = size(H)
    ixi,iyi = ix+1,iy+1
    if ix<=nx-2 && iy<=ny-2
        ∇xH_l = (H[ixi  ,iyi]-H[ixi-1,iyi])/dx
        ∇xH_r = (H[ixi+1,iyi]-H[ixi  ,iyi])/dx

        ∇xH_lu = (H[ixi  ,iyi+1]-H[ixi-1,iyi+1])/dx
        ∇xH_ru = (H[ixi+1,iyi+1]-H[ixi  ,iyi+1])/dx

        ∇xH_ld = (H[ixi  ,iyi-1]-H[ixi-1,iyi-1])/dx
        ∇xH_rd = (H[ixi+1,iyi-1]-H[ixi  ,iyi-1])/dx

        ∇yH_l = (H[ixi,iyi  ]-H[ixi,iyi-1])/dy
        ∇yH_r = (H[ixi,iyi+1]-H[ixi,iyi  ])/dy

        ∇yH_lu = (H[ixi+1,iyi  ]-H[ixi+1,iyi-1])/dy
        ∇yH_ru = (H[ixi+1,iyi+1]-H[ixi+1,iyi  ])/dy

        ∇yH_ld = (H[ixi-1,iyi  ]-H[ixi-1,iyi-1])/dy
        ∇yH_rd = (H[ixi-1,iyi+1]-H[ixi-1,iyi  ])/dy

        ∇xS_l = ∇xH_l + (B[ixi  ,iyi]-B[ixi-1,iyi])/dx
        ∇xS_r = ∇xH_r + (B[ixi+1,iyi]-B[ixi  ,iyi])/dx

        ∇xS_lu = ∇xH_lu + (B[ixi  ,iyi+1]-B[ixi-1,iyi+1])/dx
        ∇xS_ru = ∇xH_ru + (B[ixi+1,iyi+1]-B[ixi  ,iyi+1])/dx
        
        ∇xS_ld = ∇xH_ld + (B[ixi  ,iyi-1]-B[ixi-1,iyi-1])/dx
        ∇xS_rd = ∇xH_rd + (B[ixi+1,iyi-1]-B[ixi  ,iyi-1])/dx

        ∇xS_yr =  0.25*(∇xS_lu + ∇xS_ru + ∇xS_l + ∇xS_r)
        ∇xS_yl =  0.25*(∇xS_ld + ∇xS_rd + ∇xS_l + ∇xS_r)

        ∇yS_l = ∇yH_l + (B[ixi,iyi  ]-B[ixi,iyi-1])/dy
        ∇yS_r = ∇yH_r + (B[ixi,iyi+1]-B[ixi,iyi  ])/dy

        ∇yS_lu = ∇yH_lu + (B[ixi+1,iyi  ]-B[ixi+1,iyi-1])/dy
        ∇yS_ru = ∇yH_ru + (B[ixi+1,iyi+1]-B[ixi+1,iyi  ])/dy

        ∇yS_ld = ∇yH_ld + (B[ixi-1,iyi  ]-B[ixi-1,iyi-1])/dy
        ∇yS_rd = ∇yH_rd + (B[ixi-1,iyi+1]-B[ixi-1,iyi  ])/dy

        ∇yS_xr =  0.25*(∇yS_lu + ∇yS_ru + ∇yS_l + ∇yS_r)
        ∇yS_xl =  0.25*(∇yS_ld + ∇yS_rd + ∇yS_l + ∇yS_r)

        H_ax_l = 0.5*(H[ixi  ,iyi]+H[ixi-1,iyi])
        H_ax_r = 0.5*(H[ixi+1,iyi]+H[ixi  ,iyi])

        H_ay_l = 0.5*(H[ixi,iyi  ]+H[ixi,iyi-1])
        H_ay_r = 0.5*(H[ixi,iyi+1]+H[ixi,iyi  ])

        Dx_l = (a1*H_ax_l^(npow+2) + a2*H_ax_l^npow)*(∇xS_l^2 + ∇yS_xl^2)
        Dx_r = (a1*H_ax_r^(npow+2) + a2*H_ax_r^npow)*(∇xS_r^2 + ∇yS_xr^2)
        
        Dy_l = (a1*H_ay_l^(npow+2) + a2*H_ay_l^npow)*(∇yS_l^2 + ∇xS_yl^2)
        Dy_r = (a1*H_ay_r^(npow+2) + a2*H_ay_r^npow)*(∇yS_r^2 + ∇xS_yr^2)

        qHx_l = Dx_l*∇xS_l #qHx[ix  ,iy]
        qHx_r = Dx_r*∇xS_r #qHx[ix+1,iy]
        
        qHy_l = Dy_l*∇yS_l #qHy[ix  ,iy  ]
        qHy_r = Dy_r*∇yS_r #qHy[ix  ,iy+1]

        RH2[ix,iy] = -(qHx_r-qHx_l)/dx -(qHy_r-qHy_l)/dy + min(β[ix+1,iy+1]*((B[ix+1,iy+1]+H[ix+1,iy+1])-Ela[ix+1,iy+1]), c)
    end
    return
end

@parallel function compute_4!(dHdτ, RH, dτ, qHx, qHy, A, D, _dx, _dy, damp, cfl, ε)
    @all(RH)   = -_dx*@d_xa(qHx) -_dy*@d_ya(qHy) + @inn(A)
    @all(dHdτ) = @all(dHdτ)*damp + @all(RH)
    @all(dτ)   = 0.5*min(1.0, cfl/(ε+@av(D)))
    return
end

@parallel function compute_5!(H, dHdτ, dτ)
    @inn(H) = max(0.0, @inn(H) + @all(dτ)*@all(dHdτ))
    return
end

@parallel_indices (ix,iy) function set_BC!(H)
    if (ix==1         && iy<=size(H,2)) H[ix,iy] = 0.0 end
    if (ix==size(H,1) && iy<=size(H,2)) H[ix,iy] = 0.0 end
    if (ix<=size(H,1) && iy==1        ) H[ix,iy] = 0.0 end
    if (ix<=size(H,1) && iy==size(H,2)) H[ix,iy] = 0.0 end
    return
end 

# @parallel function compute_S!(S, A, B, H, Ela, β, c)
#     @all(S) = @all(B) + @all(H)
#     @all(A) = min(@all(β)*(@all(S)-@all(Ela)), c)
#     return
# end 

##################################################
@views function sia2D()
    # physics
    ρg        = 910.0*9.81
    s2yr      = 31557600.0
    npow      = 3
    Lx        = 250000
    Ly        = 200000
    B0        = 3500
    β0        = 0.01
    c         = 2.0
    # numerics
    nx        = 64
    ny        = 64
    nout      = 1000
    ε         = 1e-2
    ε_nl      = 1e-8
    damp      = 0.6
    itMax     = 1e5
    # preprocess
    a1        = 1.9e-24*ρg^npow*s2yr
    a2        = 5.7e-20*ρg^npow*s2yr
    dx, dy    = Lx/nx, Ly/ny
    xc        = LinRange(-Lx/2+dx/2, Lx/2-dx/2, nx)
    yc        = LinRange(-Ly/2+dy/2, Ly/2-dy/2, ny)
    (Xc,Yc)   = ([x for x=xc,y=yc], [y for x=xc,y=yc])
    _dx, _dy  = 1.0/dx, 1.0/dy
    cfl       = 1.0/8.1*max(dx*dx, dy*dy)
    # initial
    S         = @zeros(nx  ,ny  )
    H_avg     = @zeros(nx-1,ny-1)
    dSdx      = @zeros(nx-1,ny  )
    dSdy      = @zeros(nx  ,ny-1)
    grdS2     = @zeros(nx-1,ny-1)
    D         = @zeros(nx-1,ny-1)
    qHx       = @zeros(nx-1,ny-2)
    qHy       = @zeros(nx-2,ny-1)
    RH        = @zeros(nx-2,ny-2)
    RH2       = @zeros(nx-2,ny-2)
    dHdτ      = @zeros(nx-2,ny-2)
    dτ        = @zeros(nx-2,ny-2)
    Err       = @zeros(nx  ,ny  )
    H         = @zeros(nx  ,ny  )
    B         =  zeros(nx  ,ny  )
    A         =  zeros(nx  ,ny  )
    Ela       =  zeros(nx  ,ny  )
    β         =  zeros(nx  ,ny  )
    
    B        .= B0.*exp.(-Xc.*Xc./1e10 .- Yc.*Yc./1e9) .+ B0.*exp.(-Xc.*Xc./1e9 .- (Yc.-Ly./8).*(Yc.-Ly./8)./1e10)
    S        .= B
    Ela      .= 2150 .+ 900   .*atan.(Yc./Ly)
    # β        .= β0
    β        .= β0 .+ 0.015 .*atan.(Xc./Lx)
    A        .= min.(β.*(S.-Ela), c)
    B         = Data.Array(B)
    A         = Data.Array(A)
    Ela       = Data.Array(Ela)
    β         = Data.Array(β)

    @parallel compute_S!(S, A, B, H, Ela, β, c)

    # p1 = heatmap(xc, yc, B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="B")
    # p2 = heatmap(xc, yc, A', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="A")
    # display(plot(p1, p2))
    # visu
    H_v       = fill(NaN, nx, ny)
    Ela_v     = fill(NaN, nx, ny)
    β_v       = fill(NaN, nx, ny)

    it = 1; err = 2*ε_nl

    while (err>ε_nl && it<itMax)

        @parallel err_1!(Err, H)
        @parallel compute_dS!(H_avg, dSdx, dSdy, H, B, dx, dy)
        @parallel compute_2!(grdS2, D, H_avg, dSdx, dSdy, a1, a2, npow)
        @parallel compute_3!(qHx, qHy, D, B, H, dx, dy)
        @parallel residual!(RH2,H,B,Ela,β,a1,a2,npow,c,dx,dy)
        p1 = heatmap(xc[2:end-1], yc[2:end-1], RH', aspect_ratio=1, xlims=(xc[2], xc[end-1]), ylims=(yc[2], yc[end-1]), c=:turbo, title="$it")
        p2 = heatmap(xc[2:end-1], yc[2:end-1], RH2', aspect_ratio=1, xlims=(xc[2], xc[end-1]), ylims=(yc[2], yc[end-1]), c=:turbo, title="$it")
        p3 = heatmap(xc[2:end-1], yc[2:end-1], RH'-RH2', aspect_ratio=1, xlims=(xc[2], xc[end-1]), ylims=(yc[2], yc[end-1]), c=:turbo, title="$it")
        display(plot(p1,p2,p3))
        # sleep(1.0)
        @parallel compute_4!(dHdτ, RH, dτ, qHx, qHy, A, D, _dx, _dy, damp, cfl, ε)
        @parallel compute_5!(H, dHdτ, dτ)
        @parallel set_BC!(H)
        # @parallel compute_S!(S, A, B, H, Ela, β, c)

        it = it+1

        if mod(it, nout) == 0
            @parallel err_2!(Err, H)
            # err = sum(abs.(Err[:]))./nx./ny
            err = maximum(abs.(Err))
            @printf("iter = %d, max error = %1.3e \n", it, err)
            if (err < ε_nl) break; end
        end
    end

    Ela_v .= Ela; Ela_v[H.<=0.01] .= NaN
    H_v   .= H;     H_v[H.<=0.01] .= NaN
    β_v   .= β;     β_v[H.<=0.01] .= NaN
    p1 = heatmap(xc, yc, Ela_v', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title="ELA")
    # p2 = heatmap(xc, yc, H_v'  , aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title="H")
    p2 = heatmap(xc, yc, β_v'  , aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:turbo, title="β")
    display(plot( p1, p2 ))
    return
end

@time sia2D()
