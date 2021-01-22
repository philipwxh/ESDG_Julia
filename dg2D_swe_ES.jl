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
    r1D, w1D = Tri.gauss_quad(0,0,N)
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
N   = 2 # The order of approximation
K1D = 2
CFL = .25
T   = 1.0 # endtimeA
global g = 1.0

"Mesh related variables"
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

"Construct matrices on reference elements"
rd = init_reference_tri_sbp_GQ(N)
# rd = init_reference_tri(N)
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

E_sbp = zeros(length(rf),length(rq));
for i = 1:length(rf)
    for j = 1:length(rq)
        if abs(rq[j]-rf[i])+abs(sq[j]-sf[i])<1e-10
            E_sbp[i,j] = 1
        end
    end
end

V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
M = inv(V*V')
Dr = Vr/V
Ds = Vs/V

VN = [Vq;Vf]
VfTrWf = Vf'*diagm(wf)
VTr = [Vq;Vf]'
M = Vq'*diagm(wq)*Vq

Qr = M*Dr;
Qs = M*Ds;
Pq = M\(Vq'*diagm(wq));

# diff. matrices redefined in terms of quadrature points
Qr = Pq'*Qr*Pq;
Qs = Pq'*Qs*Pq;
E = Vf*Pq;


"Construct global coordinates"
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

Br = diagm(nrJ.*wf);
Bs = diagm(nsJ.*wf);

QNr = [Qr - .5*E'*Br*E .5*E'*Br;
    -.5*Br*E .5*Br];
QNs = [Qs - .5*E'*Bs*E .5*E'*Bs;
    -.5*Bs*E .5*Bs];

VN_sbp = [eye(length(wq));E_sbp];
QNr_sbp = VN_sbp'*QNr*VN_sbp;
@show norm(QNr_sbp+QNr_sbp' - E_sbp'*diagm(wf.*nrJ)*E_sbp)
QNs_sbp = VN_sbp'*QNs*VN_sbp;
@show norm(QNs_sbp+QNs_sbp' - E_sbp'*diagm(wf.*nsJ)*E_sbp)

#  QNr = .5*(QNr-QNr');
#  QNs = .5*(QNs-QNs');
QNr_sbp = .5*(QNr_sbp-QNr_sbp');
QNs_sbp = .5*(QNs_sbp-QNs_sbp');
PN = M\([Vq;Vf]');
Pf = E_sbp'*diagm(wf);
VNPq = [Vq;Vf]*Pq;

M_inv = diagm(@. 1/(wq*J[1][1]))
Pf = transpose(E_sbp)*diagm(wf)
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
# h = h*0 .+2
hu = h*0
hv = h*0

"pack arguments into tuples"
ops = (QNr_sbp,QNs_sbp,E_sbp,Br,Bs, M_inv, Pf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ)
nodemaps = (mapP,mapB)
# U = (h, hu, hv)

function swe_2d_rhs(h, hu, hv,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Qr,Qs,E_sbp,Br,Bs,M_inv,Pf = ops
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ = fgeo
    (mapP,mapB) = nodemaps

    hf = E_sbp*h;
    huf = E_sbp*hu;
    hvf = E_sbp*hv;
    hP = hf[mapP];
    huP = huf[mapP];
    hvP = hvf[mapP];
    UL = (hf, huf, hvf)
    UR = (hP, huP, hvP)

    (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D(UL,UR,g)

    fs1 = @. fxS1*nxJ + fyS1*nyJ;
    fs2 = @. fxS2*nxJ + fyS2*nyJ;
    fs3 = @. fxS3*nxJ + fyS3*nyJ;
    # @show findmin(hf)
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
