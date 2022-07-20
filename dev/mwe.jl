using Enzyme
using CUDA

@inbounds function my_fun!(R,H,N)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix<=length(R)  R[ix] = H[ix]^N[ix]  end # fails
    # if ix<=length(R)  R[ix] = H[ix]^2.0  end # fails
    # if ix<=length(R)  R[ix] = H[ix]^2  end # works
    return
end

function compute!(R,dR,H,dH,N)
    Enzyme.autodiff_deferred(my_fun!,Const,Duplicated(R,dR),Duplicated(H,dH),Const(N))
    return
end

function mwe()
    R  = CUDA.rand(10)
    dR = CUDA.zeros(10)
    H  = CUDA.rand(10)
    dH = CUDA.zeros(10)
    N  = CUDA.rand(10)
    threads = 10

    @cuda threads=threads compute!(R,dR,H,dH,N); synchronize()

    return
end

mwe()
