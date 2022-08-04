@inbounds function residual!(R,H,npow,dx)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    if ix>=2 && ix<=length(R)-1
        R[ix] = (mypow(H[ix-1],(npow[ix-1]+1)) - 2.0*mypow(H[ix],(npow[ix]+1)) + mypow(H[ix+1],(npow[ix+1]+1)))/dx/dx/(npow[ix]+1)
    end
    return
end

# compute forward solve
while merr >= ϵtol && iter < niter
    @cuda blocks=blocks threads=threads residual!(dR,H,npow,dx); synchronize()
    @. R  = R*(1.0-dmp/nx) + dt*dR
    @. H += dt*R
    if (iter % ncheck == 0)  merr = maximum(abs.(dR))  end
    iter += 1
end

function residual_grad!(tmp1,Ψ,H,∂J_∂H,npow,dx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp1,Ψ),Duplicated(H,∂J_∂H),Const(npow),Const(dx))
    return
end

# compute ajoint solve
while merr >= ϵtol && iter < niter
    @cuda blocks=blocks threads=threads residual_grad!(tmp1,Ψ,H,∂J_∂H,npow,dx); synchronize()
    @. R  = R*(1.0-dmp/nx) + dt*∂J_∂H
    @. Ψ += dt*R
    if (iter % ncheck == 0)  merr = maximum(abs.(∂J_∂H))  end
    iter += 1
end

function cost_grad!(Ψ,tmp2,H,Jn,npow,dx)
    Enzyme.autodiff_deferred(residual!,Duplicated(tmp2,Ψ),Const(H),Duplicated(npow,Jn),Const(dx))
    return
end

@cuda blocks=blocks threads=threads cost_grad!(.-Ψ,tmp2,H,Jn,npow,dx); synchronize()


### CPU

function residual!(R,H,npow,dx)
    @inbounds @simd for ix = 2:length(R)-1
        R[ix] = (H[ix-1]^(npow[ix-1]+1.0) - 2.0*H[ix]^(npow[ix]+1.0) + H[ix+1]^(npow[ix+1]+1.0))/dx/dx/(npow[ix]+1.0)
    end
    return
end

# compute forward solve
for iter = 1:niter
    residual!(dR,H,npow_i,dx)
    @. R      = R    *(1.0-dmp/nx) + dt*dR
    @. H_obs += dt*R_obs
    @. H     += dt*R
    if iter % nchk == 0 # check convergence
        if maximum(abs.(dR)) < εtol break end
    end
end

# compute adjoint solve
@. ∂J_∂H = H - H_obs
for iter = 1:niter
    JVP .= 0.0; Jn .= Ψ
    Enzyme.autodiff(residual!,Duplicated(R,Jn),Duplicated(H,JVP),Const(npow),Const(dx))
    @. dΨdt = dΨdt*(1.0-dmp_adj2/nx) + dt*(JVP - ∂J_∂H)
    @. Ψ[2:end-1] += dt*dΨdt[2:end-1]
    if iter % nchk == 0 # check convergence
        if maximum(abs.(dΨdt[2:end-1])) < εtol break end
    end
end

# compute cost function gradient
JVP .= .-Ψ; Jn .= 0.0;
Enzyme.autodiff(residual!,Duplicated(R,JVP),Const(H),Duplicated(npow,Jn),Const(dx))

