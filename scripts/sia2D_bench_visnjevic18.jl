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

################################ Macros from cuda_scientific
import ParallelStencil: INDICES
ix,  iy  = INDICES[1], INDICES[2]
ixi, iyi = :($ix+1), :($iy+1)

macro     sw(A) esc(:( $A[ $ix   ,  $iy   ] )) end
macro     se(A) esc(:( $A[ $ix   , ($iy+1)] )) end
macro     nw(A) esc(:( $A[($ix+1),  $iy   ] )) end
macro     ne(A) esc(:( $A[($ix+1), ($iy+1)] )) end
macro  e_inn(A) esc(:( $A[($ix+1),  $iyi  ] )) end
macro  w_inn(A) esc(:( $A[ $ix   ,  $iyi  ] )) end
macro  n_inn(A) esc(:( $A[ $ixi  , ($iy+1)] )) end
macro  s_inn(A) esc(:( $A[ $ixi  ,  $iy   ] )) end
################################

@parallel function err_1!(Err, H)
    @all(Err) = @all(H)
    return
end 

@parallel function err_2!(Err, H)
    @all(Err) = @all(Err) - @all(H)
    return
end 

# @parallel function compute_1!(B_avg, Bx, By, B)
#     @all(B_avg) = max(max(@sw(B), @se(B)), max(@nw(B), @ne(B)))
#     @all(Bx)    = max(@e_inn(B), @w_inn(B))
#     @all(By)    = max(@n_inn(B), @s_inn(B))
#     return
# end

@parallel function compute_2!(H_avg, dSdx, dSdy, grdS2, D, H, S, B_avg, _dx, _dy, a1, a2, npow)
    # @all(H_avg) = 0.25*(max(0.0, @sw(S)-@all(B_avg)) + max(0.0, @se(S)-@all(B_avg)) + max(0.0, @nw(S)-@all(B_avg)) + max(0.0, @ne(S)-@all(B_avg)))
    # @all(dSdx)  = 0.5*_dx*(max(@all(B_avg), @se(S)) - max(@all(B_avg), @sw(S)) + max(@all(B_avg), @ne(S)) - max(@all(B_avg), @nw(S)))
    # @all(dSdy)  = 0.5*_dy*(max(@all(B_avg), @nw(S)) - max(@all(B_avg), @sw(S)) + max(@all(B_avg), @ne(S)) - max(@all(B_avg), @se(S)))
    @all(H_avg) = @av(H)
    @all(dSdx)  = _dx*@d_xa(S)
    @all(dSdy)  = _dy*@d_ya(S)
    # @all(grdS2) = @all(dSdx)*@all(dSdx) + @all(dSdy)*@all(dSdy) # DEBUG: |∇S|^2 from Visnjevic et al. 2018 eq. (A5) should be |∇S|^(n-1). In the code |∇S| = sqrt(dS/dx^2 + dS/dy^2), and |∇S|^(n-1) = (sqrt(dS/dx^2 + dS/dy^2))^(npow-1)
    @all(grdS2) = @av_ya(dSdx)*@av_ya(dSdx) + @av_xa(dSdy)*@av_xa(dSdy)  # DEBUG: |∇S|^2 from Visnjevic et al. 2018 eq. (A5) should be |∇S|^(n-1). In the code |∇S| = sqrt(dS/dx^2 + dS/dy^2), and |∇S|^(n-1) = (sqrt(dS/dx^2 + dS/dy^2))^(npow-1)
    @all(D)     = (a1*@all(H_avg)^(npow+2) + a2*@all(H_avg)^npow)*@all(grdS2)
    return
end

@parallel function compute_3!(qHx, qHy, D, Bx, By, S, _dx, _dy)
    # @all(qHx)  = -@av_ya(D)*( max(@all(Bx), @e_inn(S)) - max(@all(Bx), @w_inn(S)) )*_dx
    # @all(qHy)  = -@av_xa(D)*( max(@all(By), @n_inn(S)) - max(@all(By), @s_inn(S)) )*_dy
    @all(qHx)  = -@av_ya(D)*@d_xi(S)*_dx
    @all(qHy)  = -@av_xa(D)*@d_yi(S)*_dy
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

@parallel function compute_S!(S, A, B, H, Ela, β, c)
    @all(S) = @all(B) + @all(H)
    @all(A) = min(@all(β)*(@all(S)-@all(Ela)), c)
    return
end 

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
    nx        = 128
    ny        = 128
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
    B_avg     = @zeros(nx-1,ny-1)
    H_avg     = @zeros(nx-1,ny-1)
    # dSdx      = @zeros(nx-1,ny-1)
    # dSdy      = @zeros(nx-1,ny-1)
    dSdx      = @zeros(nx-1,ny  )
    dSdy      = @zeros(nx  ,ny-1)
    grdS2     = @zeros(nx-1,ny-1)
    D         = @zeros(nx-1,ny-1)
    Bx        = @zeros(nx-1,ny-2)
    By        = @zeros(nx-2,ny-1)
    qHx       = @zeros(nx-1,ny-2)
    qHy       = @zeros(nx-2,ny-1)
    RH        = @zeros(nx-2,ny-2)
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

    p1 = heatmap(xc, yc, B', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="B")
    p2 = heatmap(xc, yc, A', aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), c=:viridis, title="A")
    display(plot(p1, p2))
    # visu
    H_v       = fill(NaN, nx, ny)
    Ela_v     = fill(NaN, nx, ny)
    β_v       = fill(NaN, nx, ny)

    it = 1; err = 2*ε_nl

    while (err>ε_nl && it<itMax)

        @parallel err_1!(Err, H)
        # @parallel compute_1!(B_avg, Bx, By, B)
        @parallel compute_2!(H_avg, dSdx, dSdy, grdS2, D, H, S, B_avg, _dx, _dy, a1, a2, npow)
        @parallel compute_3!(qHx, qHy, D, Bx, By, S, _dx, _dy)
        @parallel compute_4!(dHdτ, RH, dτ, qHx, qHy, A, D, _dx, _dy, damp, cfl, ε)
        @parallel compute_5!(H, dHdτ, dτ)
        @parallel set_BC!(H)
        @parallel compute_S!(S, A, B, H, Ela, β, c)

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
