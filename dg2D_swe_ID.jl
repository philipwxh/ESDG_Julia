using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using ForwardDiff
using SparseArrays
using StaticArrays
push!(LOAD_PATH, "./src") # user defined modules
# using CommonUtils
# using Basis1D
using Basis2DTri
# using SetupDG
# using UniformTriMesh
using NodesAndModes
using NodesAndModes.Tri
using UnPack
using StartUpDG
using StartUpDG.ExplicitTimestepUtils

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
    E = Matrix(droptol!(sparse(E),1e-13))

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

function init_reference_tri_sbp_GQ(N)
    include("SBP_quad_data.jl")
    # initialize a new reference element data struct
    rd = RefElemData()

    fv = tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = Tri.nodes(N)
    VDM = Tri.vandermonde(N, r, s)
    Vr, Vs = Tri.grad_vandermonde(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM,Dr,Ds

    # low order interpolation nodes
    r1,s1 = Tri.nodes(1)
    V1 = Tri.vandermonde(1,r,s)/Tri.vandermonde(1,r1,s1)
    @pack! rd = V1

    #Nodes on faces, and face node coordinate
    r1D, w1D = gauss_quad(0,0,N)
    Nfp = length(r1D) # number of points per face
    e = ones(Nfp) # vector of all ones
    z = zeros(Nfp) # vector of all zeros
    rf = [r1D; -r1D; -e];
    sf = [-e; r1D; -r1D];
    wf = vec(repeat(w1D,3,1));
    nrJ = [z; e; -e]
    nsJ = [-e; e; z]
    @pack! rd = rf,sf,wf,nrJ,nsJ

    rq,sq,wq = GQ_SBP[N]
    Vq = Tri.vandermonde(N,rq,sq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = Tri.vandermonde(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(Vf'*diagm(wf)) # lift matrix used in rhs evaluation
    @pack! rd = Vf,LIFT

    # plotting nodes
    rp, sp = Tri.equi_nodes(10)
    Vp = Tri.vandermonde(N,rp,sp)/VDM
    @pack! rd = rp,sp,Vp

    return rd
end

avg(a,b) = .5*(a+b)
function fS2D(UL,UR,g)
    hL,huL,hvL = UL
    hR,huR,hvR = UR
    uL,vL = (x->x./hL).((huL,hvL))
    uR,vR = (x->x./hR).((huR,hvR))
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*hL*hR
    fxS3 = @. avg(huL,huR)*avg(vL,vR)

    fyS1 = @. avg(hvL,hvR)
    fyS2 = @. avg(hvL,hvR)*avg(uL,uR)
    fyS3 = @. avg(hvL,hvR)*avg(vL,vR) + .5*g*hL*hR
    return (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3)
end

"Approximation parameters"
N   = 3 # The order of approximation
K1D = 2
CFL = .125
T   = 2.0 # endtimeA
global g = 1.0

"Mesh related variables"
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

"Construct matrices on reference elements"
Nq = 2*N
rd = init_reference_tri_sbp_GQ(N)
# rd = init_reference_tri(N)
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

# E_sbp = zeros(length(rf),length(rq));
# for i = 1:length(rf)
#     for j = 1:length(rq)
#         if abs(rq[j]-rf[i])+abs(sq[j]-sf[i])<1e-10
#             E_sbp[i,j] = 1
#         end
#     end
# end

# V = vandermonde_2D(N, r, s)
# Vr, Vs = grad_vandermonde_2D(N, r, s)
# M = inv(V*V')
# Dr = Vr/V
# Ds = Vs/V

# "Nodes on faces, and face node coordinate"
# r1D, w1D = gauss_quad(0,0,N)
# Nfp = length(r1D)
# e = ones(Nfp,1)
# z = zeros(Nfp,1)
# rf = [r1D; -r1D; -e];
# sf = [-e; r1D; -r1D];
# wf = vec(repeat(w1D,3,1));
# nrJ = [z; e; -e]
# nsJ = [-e; e; z]
# Vf = vandermonde_2D(N,rf,sf)/V
# LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix

α = 2.1
Qr,Qs,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
Qr = .5*(Qr-transpose(Qr))
Qs = .5*(Qs-transpose(Qs))

"Construct global coordinates"
# r1,s1 = nodes_2D(1)
# V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Connectivity maps"
xf,yf = (x->Vf*x).((x,y))
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
mapM = reshape(mapM,length(wf),K)
mapP = reshape(mapP,length(wf),K)

"Make periodic"
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)
rxJ = Vq*rxJ
sxJ = Vq*sxJ
ryJ = Vq*ryJ
syJ = Vq*syJ
J = Vq*J
nx = nxJ./sJ;
ny = nyJ./sJ;

M_inv = diagm(@. 1/(wq*J[1][1]))
Pf = transpose(E)*diagm(wf)
"initial conditions"
xq = Vq*x
yq = Vq*y

# function ck45()
#     rk4a = [            0.0 ...
#     -567301805773.0/1357537059087.0 ...
#     -2404267990393.0/2016746695238.0 ...
#     -3550918686646.0/2091501179385.0  ...
#     -1275806237668.0/842570457699.0];
#
#     rk4b = [ 1432997174477.0/9575080441755.0 ...
#     5161836677717.0/13612068292357.0 ...
#     1720146321549.0/2090206949498.0  ...
#     3134564353537.0/4481467310338.0  ...
#     2277821191437.0/14882151754819.0]
#
#     rk4c = [ 0.0  ...
#     1432997174477.0/9575080441755.0 ...
#     2526269341429.0/6820363962896.0 ...
#     2006345519317.0/3224310063776.0 ...
#     2802321613138.0/2924317926251.0 ...
#     1.0];
#     return rk4a,rk4b,rk4c
# end
"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

h = @. exp(-25*( xq^2+yq^2))+2
hu = h*0
hv = h*0

"pack arguments into tuples"
ops = (Qr,Qs,E,Br,Bs, M_inv, Pf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ)
nodemaps = (mapP,mapB)
# U = (h, hu, hv)

function swe_2d_rhs(h, hu, hv,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Qr,Qs,E,Br,Bs,M_inv,Pf = ops
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ = fgeo
    (mapP,mapB) = nodemaps

    hf = E*h;
    huf = E*hu;
    hvf = E*hv;
    hP = hf[mapP];
    huP = huf[mapP];
    hvP = hvf[mapP];
    UL = (hf, huf, hvf)
    UR = (hP, huP, hvP)

    (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D(UL,UR,g)

    fs1 = @. fxS1*nxJ + fyS1*nyJ;
    fs2 = @. fxS2*nxJ + fyS2*nyJ;
    fs3 = @. fxS3*nxJ + fyS3*nyJ;
    @show findmin(hf)
    Lf  = @. abs(huf./hf*nx + hvf./hf*ny) + sqrt(g*hf);
    Lfc = @. max(Lf,Lf[mapP]);
    tau = 1
    f1 = @. fs1 - .5*tau*Lfc*(hP-hf)*sJ;
    f2 = @. fs2 - .5*tau*Lfc*(huP-huf)*sJ;
    f3 = @. fs3 - .5*tau*Lfc*(hvP-hvf)*sJ;

    rhs1 = zeros(size(h));
    rhs2 = zeros(size(h));
    rhs3 = zeros(size(h));

    # % loop over all elements
    for e = 1:K
        hx, hy  = meshgrid(h[:,e] );
        hux,huy = meshgrid(hu[:,e]);
        hvx,hvy = meshgrid(hv[:,e]);
        UL = (hx, hux, hvx)
        UR = (hy, huy, hvy)
        # % get the flux for each component and both directions
        (FxV1,FxV2,FxV3),(FyV1,FyV2,FyV3) = fS2D(UL,UR,g)

        # % build the QNx and QNy for each elements
        QNx = 1/2*( diagm(rxJ[:,e])*Qr + Qr*diagm(rxJ[:,e]) + diagm(sxJ[:,e])*Qs + Qs*diagm(sxJ[:,e]) );
        QNy = 1/2*( diagm(ryJ[:,e])*Qr + Qr*diagm(ryJ[:,e]) + diagm(syJ[:,e])*Qs + Qs*diagm(syJ[:,e]) );

        # % VTr Me*PN = [Vq' Vf']
        rhs1[:,e] = 2*(sum(QNx.*FxV1,dims = 2) + sum(QNy.*FyV1,dims = 2));
        rhs2[:,e] = 2*(sum(QNx.*FxV2,dims = 2) + sum(QNy.*FyV2,dims = 2));
        rhs3[:,e] = 2*(sum(QNx.*FxV3,dims = 2) + sum(QNy.*FyV3,dims = 2));

        # rhs2[:,e] = rhs2[:,e] + g*(h[:,e].*(QNx*b[:,e]));
        # rhs3[:,e] = rhs3[:,e] + g*(h[:,e].*(QNy*b[:,e]));
    end
    rhs1 = rhs1 + Pf*f1;
    rhs2 = rhs2 + Pf*f2;
    rhs3 = rhs3 + Pf*f3;

    # rhs1 = M_inv*Pf*f1
    # rhs2 = M_inv*Pf*f1
    # rhs3 = M_inv*Pf*f1
    return -M_inv*rhs1, -M_inv*rhs2, -M_inv*rhs3
end

res1 = zeros(size(xq))
res2 = zeros(size(xq))
res3 = zeros(size(xq))
for i = 1:Nsteps
    global h, hu, hv, res1, res2, res3 # for scoping - these variables are updated

    for INTRK = 1:5
        rhs1,rhs2,rhs3 = swe_2d_rhs(h, hu, hv,ops,vgeo,fgeo,nodemaps)
        # @show findmax( @. abs(rhsu) )
        @. res1 = rk4a[INTRK]*res1 + dt*rhs1
        @. res2 = rk4a[INTRK]*res2 + dt*rhs2
        @. res3 = rk4a[INTRK]*res3 + dt*rhs3

        @. h  += rk4b[INTRK]*res1
        @. hu += rk4b[INTRK]*res2
        @. hv += rk4b[INTRK]*res3
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

"plotting nodes"

gr(aspect_ratio=1,legend=false,
   markerstrokewidth=0,markersize=2)

rp, sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/vandermonde_2D(N,rq,sq)
vv = Vp*h
scatter(Vp*xq,Vp*yq,vv,zcolor=vv,camera=(0,90))
