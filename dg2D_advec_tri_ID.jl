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
beta=[1,0]
avg(a,b) = .5*(a+b)

"Approximation parameters"
N   = 3 # The order of approximation
K1D = 16
CFL = 0.125
T   = .1 # endtimeA

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

α = 2.5
Qr,Qs,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
# Qr = .5*(Qr-transpose(Qr))
# Qs = .5*(Qs-transpose(Qs))

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

x_i,x_j = meshgrid(rq)
y_i,y_j = meshgrid(sq)
x_ij = x_j-x_i
y_ij = y_j-y_i
x_ij = x_ij./(@. sqrt(x_ij^2+y_ij^2))
y_ij = y_ij./(@. sqrt(x_ij^2+y_ij^2))
replace!(x_ij, NaN=>0)
replace!(y_ij, NaN=>0)

wn  =  wq*0
wn[1:length(wf)] = transpose(wf.*sJ[:,1])
wn = diagm(wn)*x_ij.*Matrix(A)
wn[:,length(wf)+1:end] .= 0
QNx =  rxJ[1,1]*Qr  + sxJ[1,1]*Qs;
QNy =  ryJ[1,1]*Qr  + syJ[1,1]*Qs;
D = zeros(length(wq[:,1]), length(wq[:,1]))
for i = 1:length(wq[:,1])
    for j = i+1:length(wq[:,1])
        D[i,j] = QNx[i,j]*sqrt(QNx[i,j]^2+ QNy[i,j]^2)
        D[j,i] = QNx[j,i]*sqrt(QNx[j,i]^2+ QNy[j,i]^2)
    end
    D[i,i] = -sum(D[i,:])
end

M_inv = diagm(@. 1/(wq*J[1][1]))
Pf = transpose(E)*diagm(wf)
"initial conditions"
xq = Vq*x
yq = Vq*y

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Qr,Qs,E,Br,Bs, M_inv, Pf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ)
nodemaps = (mapP,mapB)

"initial conditions"
xq = Vq*x
yq = Vq*y
u = @. exp(-25*( xq^2+yq^2))

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Qr,Qs,E,Br,Bs, M_inv, Pf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ, wn)
nodemaps = (mapP,mapB)

function rhs(u,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Qr,Qs,E,Br,Bs,M_inv,Pf = ops
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ,wn = fgeo
    (mapP,mapB) = nodemaps

    uM = E*u # can replace with nodal extraction
    uP = uM[mapP]
    du = uP-uM
    tau = 1
    FxS = @. (avg(uM, uP)-uM)*nxJ - 0.5*tau*du*abs(nxJ)
    rhsu = u*0
    for e = 1:K
        ux, uy  = meshgrid(u[:,e] );
        FxV = avg(ux,uy);
        # % build the QNx and QNy for each elements
        QNx =  rxJ[1,e]*Qr  + sxJ[1,e]*Qs;
        QNy =  ryJ[1,e]*Qr  + syJ[1,e]*Qs;
        # D = zeros(length(u[:,1]), length(u[:,1]))
        # for i = 1:length(u[:,1])
        #     for j = i+1:length(u[:,1])
        #         D[i,j] = abs(Qr[i,j])
        #         D[j,i] = abs(Qr[i,j])
        #     end
        #     D[i,i] = -sum(D[i,:])
        # end
        # % VTr Me*PN = [Vq' Vf']
        rhsu[:,e] = 2*(sum(QNx.*FxV,dims = 2) )#+ sum(QNy.*FxV,dims = 2));
        # rhsu[:,e] -= transpose(sum(D*(uy-ux), dims = 1))
        rhsu[:,e] -= D*u[:,e]
        # rhsu[:,e] += 0.5*sum(wn.*FxV, dims = 2)
        # rhsu[:,e] -= 0.5*transpose( sum((@. abs(wn).*(uy-ux)), dims = 1) )
    end
    # ur = Qr*u
    # us = Qs*u
    # ux = @. rxJ*ur + sxJ*us
    # uy = @. ryJ*ur + syJ*us
    # tau = 1
    # rhsu = M_inv*((ux+uy) +  Pf*(@. .5*du*nxJ - tau*du*abs(nxJ)))
    rhsu = M_inv*(rhsu + Pf*FxS)
    return -rhsu
end

resu = zeros(size(xq))
for i = 1:Nsteps
    global u, resu # for scoping - these variables are updated

    for INTRK = 1:5
        rhsu = rhs(u,ops,vgeo,fgeo,nodemaps)
        # @show findmax( @. abs(rhsu) )
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u += rk4b[INTRK]*resu
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
vv = Vp*u
scatter(Vp*xq,Vp*yq,vv,zcolor=vv,camera=(0,90))
