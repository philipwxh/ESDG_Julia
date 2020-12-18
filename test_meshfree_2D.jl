using Revise
using Plots
using ForwardDiff
using LinearAlgebra
using SparseArrays
using StaticArrays


push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using NodesAndModes
using NodesAndModes.Tri
# using SetupDG
using UnPack
using Basis1D
using Basis2DTri

push!(LOAD_PATH, "./StartUpDG/src") # user defined modules
using StartUpDG

# using TriangleMesh

N = 3
Nq = 2*N
rd = init_reference_tri(N,Nq=Nq)
# rd = init_reference_tri(N)
@unpack rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd

α = 2.1

# h = [0.27266912599079735, 0.3031075210365489, 0.23059195079479833, 0.17564948347284076, 0.17335679156397352, 0.17151846079077565, 0.1389489400379573, 0.13257957516241775, 0.12882178206543554, 0.10996189711173587]
# err = vec([0.2565579655139645 0.06760292524095085 0.06640308587559195 0.11235885234018354 0.07341686552932333 0.04582634500602509 0.018387741850793386 0.01869444862625115  0.018818558042348683 0.009318336328365326])

function build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
    # [-1,1,0], [-1,-1,sqrt(4/3)]
    equilateral_map(r,s) = (@. .5*(2*r+1*s+1), @. sqrt(3)*(1+s)/2 - 1/sqrt(3) )
    req,seq = equilateral_map(rq,sq)
    ref,sef = equilateral_map(rf,sf)
    barycentric_coords(r,s) = ((@. (1+r)/2), (@. (1+s)/2), (@. -(r+s)/2))
    λ1,λ2,λ3 = barycentric_coords(rq,sq)
    λ1f,λ2f,λ3f = barycentric_coords(rf,sf)

    Br = diagm(nrJ.*wf)
    Bs = diagm(nsJ.*wf)

    # build extrapolation matrix
    E = zeros(length(rf),length(rq))
    for i = 1:length(rf)
        # d = @. (λ1 - λ1f[i])^2 + (λ2 - λ2f[i])^2 + (λ3 - λ3f[i])^2
        d2 = @. (req-ref[i])^2 + (seq-sef[i])^2
        p = sortperm(d2)
        h2 = (wf[i]/sum(wf))*2/pi # set so that h = radius of circle with area w_i = face weight
        nnbrs = min(4,max(3,count(d2[p] .< h2))) # find 3 closest points
        p = p[1:nnbrs]
        Ei = vandermonde_2D(1,[rf[i]],[sf[i]])/vandermonde_2D(1,rq[p],sq[p])
        E[i,p] = Ei
    end
    E = Matrix(droptol!(sparse(E),1e-14))

    # build stencil
    A = spzeros(length(req),length(req))
    for i = 1:length(req)
        d2 = @. (req-req[i])^2 + (seq-seq[i])^2
        p = sortperm(d2)

        # h^2 = wq[i]/pi = radius of circle with area wq[i]
        # h2 =     (sqrt(3)/sum(wq))*wq[i]/pi
        h2 = α^2*(sqrt(3)/sum(wq))*wq[i]/pi

        nnbrs = count(d2[p] .< h2)
        nbrs = p[1:nnbrs]
        A[i,nbrs] .= one(eltype(A))
    end
    A = (A+A')
    A.nzval .= one(eltype(A)) # bool-ish

    # build graph Laplacian
    L1 = (A-diagm(diag(A))) # ignore
    L1 -= diagm(vec(sum(L1,dims=2)))

    b1r = -sum(.5*E'*Br*E,dims=2)
    b1s = -sum(.5*E'*Bs*E,dims=2)
    ψ1r = pinv(L1)*b1r
    ψ1s = pinv(L1)*b1s

    function fillQ(adj,ψ)
        Np = length(ψ)
        S = zeros(Np,Np)
        for i = 1:Np
            for j = 1:Np
                if adj[i,j] != 0
                        S[i,j] += (ψ[j]-ψ[i])
                end
            end
        end
        return S
    end

    S1r,S1s = fillQ.((A,A),(ψ1r,ψ1s))
    Qr = Matrix(droptol!(sparse(S1r + .5*E'*Br*E),1e-14))
    Qs = Matrix(droptol!(sparse(S1s + .5*E'*Bs*E),1e-14))

    return Qr,Qs,E,Br,Bs,A
end
# error("d")

# Nvec = 2:13
# h = zeros(size(Nvec))
# err = zeros(size(Nvec))
# for (i,N) in enumerate(Nvec)
#     rd = init_reference_tri(N,Nq=2*N)
#     @unpack rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
#     α = 2.75
#     Qr,Qs,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
#     h[i] = sqrt(maximum(wq)/pi)
#     err[i] = sqrt(sum(wq.*(((Qr*sq)./wq).^2 + ((Qs*rq)./wq).^2 + ((Qr*rq)./wq .- 1).^2 + ((Qs*sq)./wq .- 1).^2)))
# end
# error("d")

# avec = 1/3:.01:1.5
# err = zero.(avec)
# nzs = zero.(avec)
# for (i,α) in enumerate(avec)
#     Qr,Qs,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,α)
#     err0 = norm(sum(Qr,dims=2))^2 + norm(sum(Qs,dims=2))^2
#     err[i] = 100*err0 + norm(Qr*sq)^2+norm((Qs*rq))^2+norm((Qr*rq)-wq)^2+norm((Qs*sq)-wq)^2
#     nzs[i] = nnz(droptol!(sparse(Qr-Qr'),1e-12))+nnz(droptol!(sparse(Qs-Qs'),1e-12))
# end
# display(plot(avec,err,marker=:dot,ms=4))
# display(plot!(avec,nzs/(2*length(rq)^2),marker=:dot,ms=4,ylim=(0.,.25),legend=false))
# error("d")

Qr,Qs,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)

function build_K2D(rq,sq,wq,rf,sf,wf,nrJ,nsJ,E,adj)
    Np = length(rq)
    # build Laplacian matrix
    Qr = zeros(Np,Np)
    Qs = zeros(Np,Np)
    for i = 1:Np
        for j = 1:Np
            if adj[i,j] != 0
                Qr[i,j] = abs(rq[j]-rq[i])
                Qs[i,j] = abs(sq[j]-sq[i])
                # Lr[i,j] = sqrt((rq[i]-rq[j])^2)# + (sq[i]-sq[j])^2)
            end
        end
    end
    return Qr,Qs
end
Krd,Ksd = build_K2D(rq,sq,wq,rf,sf,wf,nrJ,nsJ,E,A)
Kr = Krd-diagm(vec(sum(Krd,dims=2)))
Ks = Ksd-diagm(vec(sum(Ksd,dims=2)))
@show norm(Kr*rq)/(norm(Kr)*norm(rq)), norm(Ks*sq)/(norm(Ks)*norm(sq))
# error("d")


equilateral_map(r,s) = (@. .5*(2*r+1*s+1), @. sqrt(3)*(1+s)/2 - 1/sqrt(3) )
re,se = equilateral_map(rd.r,rd.s)
req,seq = equilateral_map(rq,sq)
ref,sef = equilateral_map(rf,sf)

# build graph Laplacian
L1 = (A-Diagonal(diag(A))) # ignore
L1 -= diagm(vec(sum(L1,dims=2)))
b1r = -sum(.5*E'*Br*E,dims=2)
b1s = -sum(.5*E'*Bs*E,dims=2)
ψ1r = pinv(L1)*b1r
ψ1s = pinv(L1)*b1s
# error("d")


plot([-1,1,0,-1],[-1/sqrt(3),-1/sqrt(3),sqrt(4/3),-1/sqrt(3)],lw=2,linecolor=:black)
scatter!(req,seq)
scatter!(ref,sef,marker=:square)
# t = LinRange(0,2*pi,30)
# rc = @. cos(t)
# sc = @. sin(t)
# vol = 0
# for i = 1:length(rq)
#     global vol
#     # π*r^2 = wq, and wq sum to sqrt(4/3)
#     h2 = (sqrt(3)/sum(wq))*wq[i]/pi
#     h  = sqrt(h2)
#     vol += pi*h^2
#     plot!(@. h*rc+req[i],@. h*sc+seq[i])
# end
# @show vol

for i = 1:length(rq), j = 1:length(rq)
    if abs(A[i,j])>1e-13
        ids = [i,j]
        plot!(req[ids],seq[ids])
    end
end
for i = 1:length(rf), j = 1:length(rq)
    if abs(E[i,j])>1e-10
        plot!([ref[i], req[j]],[sef[i], seq[j]],linestyle=:dash)
    end
end
display(plot!(legend=false,ratio=1))

@assert(norm(sum(E,dims=2) .- 1) + norm(E*rq-rf) + norm(E*sq-sf) < 1e-12)
@assert(norm(sum(Qr,dims=2)) + norm(sum(Qs,dims=2)) < 1e-12)
@show norm(Qr*sq)^2+norm((Qs*rq))^2+norm((Qr*rq)-wq)^2+norm((Qs*sq)-wq)^2
