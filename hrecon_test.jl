using Plots
using LinearAlgebra
x = LinRange(-1,1,3)
b = @. 2/3*(1+x)
h = @. max(1e-10,1.0 - b)
Q = diagm(1=>ones(length(x)-1),-1=>-ones(length(x)-1))
Q[1] = -1; Q[end] = 1
Q = Q./2;
recon(hL,bL,bR) = max(0,hL + bL - max(bL,bR))
val = zeros(size(Q,1))
for i = 1:size(Q,1)
    for j = 1:size(Q,2)
        hij = recon(h[i],b[i],b[j])
        hji = recon(h[j],b[j],b[i])
        val[i] += Q[i,j]*(hji^2 - hij^2)
    end
end
@show norm(val)
