using Plots, Printf

function residual!(R,H,B,Ela,β,dτ,c,npow,a1,a2,cfl,ε,dx)
    @inbounds @simd for ix = 2:length(H)-1
        D_l    = (a1 * (0.5*(H[ix-1]+H[ix]))^(npow+2) + a2 * (0.5*(H[ix-1]+H[ix]))^npow) * ((B[ix]-B[ix-1])/dx + (H[ix]-H[ix-1])/dx)^2
        D_r    = (a1 * (0.5*(H[ix]+H[ix+1]))^(npow+2) + a2 * (0.5*(H[ix]+H[ix+1]))^npow) * ((B[ix+1]-B[ix])/dx + (H[ix+1]-H[ix])/dx)^2
        qHx_l  = -D_l*((B[ix  ]-B[ix-1])/dx + (H[ix  ]-H[ix-1])/dx)
        qHx_r  = -D_r*((B[ix+1]-B[ix  ])/dx + (H[ix+1]-H[ix  ])/dx)
        R[ix]  = -(qHx_r - qHx_l)/dx + min(β[ix] * (B[ix] + H[ix] - Ela[ix]), c)
        dτ[ix] = 0.5*min(1.0, cfl/(ε + max(D_l, D_r)))
    end
    return
end

@views function sia1D()
    # physics
    ρg     = 910.0*9.81
    s2yr   = 31557600.0
    npow   = 3
    Lx     = 250000
    B0     = 3500
    β0     = 0.01
    c      = 2.0
    # numerics
    nx     = 128*4
    nout   = 100
    ε      = 1e-2
    ε_nl   = 1e-8
    dmp    = 0.72
    itMax  = 1e5
    # preprocess
    a1     = 1.9e-24*ρg^npow*s2yr
    a2     = 5.7e-20*ρg^npow*s2yr
    dx     = Lx/nx
    xc     = LinRange(-Lx/2+dx/2, Lx/2-dx/2, nx)
    cfl    = 1.0/2.1*dx*dx
    # initial
    B      = zeros(nx)
    R      = zeros(nx)
    dHdτ   = zeros(nx)
    dτ     = zeros(nx)
    Err    = zeros(nx)
    Ela    = zeros(nx)
    β      = zeros(nx)
    H      = 100.0*ones(nx)
    H[[1,end]] .= 0.0 
    @. B   = B0*exp(-xc*xc/1e10) + B0*exp.(-xc*xc/1e9)
    # @. Ela = 2150 + 900
    @. Ela = 3000 + 400 * atan(xc/Lx)
    # @. β   = β0
    @. β   = β0 + 0.015 * atan(xc/Lx)
    # PT solve
    it = 1; err = 2*ε_nl
    while (err>ε_nl && it<itMax)
        residual!(R,H,B,Ela,β,dτ,c,npow,a1,a2,cfl,ε,dx)
        @. Err  = H
        @. dHdτ = dHdτ*dmp + R
        @. H    = max(0.0, H + dτ*dHdτ)
        it += 1
        if it % nout == 0
            @. Err -= H
            err = maximum(abs.(Err))
            @printf("iter = %d, max error = %1.3e \n", it, err)
            if (err < ε_nl) break; end
        end
    end
    S  = B .+ H; S[H.==0] .= NaN
    p1 = plot(xc,[B,S], label=["Bed" "Surface"], linewidth=3)
    p2 = plot(xc, H   , label="Ice thick", linewidth=3)
    # p3 = plot(xc,β  , label="β", linewidth=3)
    p3 = plot(xc,Ela, label="ELA", linewidth=3)
    display(plot(p1,p2,p3, layout=(3, 1)))
    return
end

@time sia1D()
