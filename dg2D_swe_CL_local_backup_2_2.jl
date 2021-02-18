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

const g = 1.0
"Approximation parameters"
N   = 3 # The order of approximation
K1D = 8
CFL = 1/4
T   = 1 # endtimeA
MAXIT = 1000000

ts_ft= 1/2

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

function fS2D_LF(UL,UR,g)
    hL,huL,hvL = UL
    hR,huR,hvR = UR
    uL,vL = (x->x./hL).((huL,hvL))
    uR,vR = (x->x./hR).((huR,hvR))
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL*uL,huR*uR) + .5*g*avg(hL*hL,hR*hR)
    fxS3 = @. avg(huL*vL,huR*vR)

    fyS1 = @. avg(hvL,hvR)
    fyS2 = @. avg(hvL*uL,hvR*uR)
    fyS3 = @. avg(hvL*uL,hvR*vR) + .5*g*avg(hL*hL,hR*hR)
    return (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3)
end

function convex_limiter(rhs_ES, rhs_ID, hh, htmp, tol, dt)
    rhs1_ES, rhs2_ES, rhs3_ES = rhs_ES
    rhs1_ID, rhs2_ID, rhs3_ID = rhs_ID
    rhs1_Diff = rhs1_ES - rhs1_ID; rhs2_Diff = rhs2_ES - rhs2_ID; rhs3_Diff = rhs3_ES - rhs3_ID;
    rhs1 = zeros(size(hh)); rhs2 = zeros(size(hh)); rhs3 = zeros(size(hh));
    L = ones(1,size(hh,2));
    for e = 1:size(hh,2)
        for k = 1:size(hh,1)
            l_k = 1
            if rhs1_Diff[k,e] < 0
                l_k = -(hh[k,e] + dt*rhs1_ID[k,e]-tol) / (dt*(rhs1_Diff[k,e]))
                l_k = min(l_k, 1)
            end
            l_k = max(l_k,0)
            L[e] = min(L[e], l_k);
            if hh[k,e] <= tol || htmp[k,e] <= tol || norm(rhs1_Diff)<=tol
                L[e] = 0
                # if e == 1
                #     L[e+1] = 0
                #     L[end] = 0
                # elseif  e == K1D
                #     L[e-1] = 0
                #     L[1] = 0
                # else
                #     L[e+1] = 0
                #     L[e-1] = 0
                # end
            end
        end
        rhs1[:,e] = rhs1_ID[:,e] + rhs1_Diff[:,e] * L[e]
        rhs2[:,e] = rhs2_ID[:,e] + rhs2_Diff[:,e] * L[e]
        rhs3[:,e] = rhs3_ID[:,e] + rhs3_Diff[:,e] * L[e]
    end
    return (rhs1, rhs2, rhs3), L
end

# Mesh related variables
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

# Construct matrices on reference elements
rd = init_reference_tri_sbp_GQ(N)
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
M = inv(V*V')
Dr_ES = Vr/V; Ds = Vs/V;
M = Vq'*diagm(wq)*Vq

Qr_ES = M*Dr;
Qs_ES = M*Ds;
Pq = M\(Vq'*diagm(wq));

# diff. matrices redefined in terms of quadrature points
Qr_ES = Pq'*Qr_ES*Pq;
Qs_ES = Pq'*Qs_ES*Pq;
E_ES = Vf*Pq;

# Need to choose α so that Qr, Qs have zero row sums (and maybe a minimum number of neighbors)
# α = 4 # for N=1
# α = 2.5 #for N=2
α = 3.5 # for N=3
Qr_ID,Qs_ID,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
if (norm(sum(Qr_ID,dims=2)) > 1e-10) | (norm(sum(Qs_ID,dims=2)) > 1e-10)
    error("Qr_ID or Qs_ID doesn't sum to zero for α = $α")
end
Qr_ID = Matrix(droptol!(sparse(Qr_ID),1e-15))
Qs_ID = Matrix(droptol!(sparse(Qs_ID),1e-15))
Qrskew_ID = .5*(Qr_ID-transpose(Qr_ID))
Qsskew_ID = .5*(Qs_ID-transpose(Qs_ID))

# "Construct global coordinates"
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

# "Connectivity maps"
xf,yf = (x->Vf*x).((x,y))
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
mapM = reshape(mapM,length(wf),K)
global mapP = reshape(mapP,length(wf),K)

# "Make periodic"
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

# "Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
rxJ = Matrix(droptol!(sparse(rxJ),1e-14)); sxJ = Matrix(droptol!(sparse(sxJ),1e-14))
ryJ = Matrix(droptol!(sparse(ryJ),1e-14)); syJ = Matrix(droptol!(sparse(syJ),1e-14))
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)
nx = nxJ./sJ; ny = nyJ./sJ;
rxJ,sxJ,ryJ,syJ,J = (x->Vq*x).((rxJ,sxJ,ryJ,syJ,J))

QNr = [Qr_ES - .5*E_ES'*Br*E_ES .5*E_ES'*Br;
    -.5*Br*E_ES .5*Br];
QNs = [Qs_ES - .5*E_ES'*Bs*E_ES .5*E_ES'*Bs;
    -.5*Bs*E_ES .5*Bs];

VN_sbp = [eye(length(wq));E];
QNr_sbp = VN_sbp'*QNr*VN_sbp;
@show norm(QNr_sbp+QNr_sbp' - E'*diagm(wf.*nrJ)*E)
QNs_sbp = VN_sbp'*QNs*VN_sbp;
@show norm(QNs_sbp+QNs_sbp' - E'*diagm(wf.*nsJ)*E)

#  QNr = .5*(QNr-QNr');
#  QNs = .5*(QNs-QNs');
Qrskew_ES = .5*(QNr_sbp-QNr_sbp');
Qsskew_ES = .5*(QNs_sbp-QNs_sbp');

M_inv = diagm(@. 1/(wq*J[1][1]))
Mf_inv = E*M_inv*E';
Pf = transpose(E)*diagm(wf)
Cf = abs.(rxJ[1,1]*diag(Qr_ID)[1:length(wf)] + ryJ[1,1]*diag(Qr_ID)[1:length(wf)] + syJ[1,1]*diag(Qs_ID)[1:length(wf)])

cij_x =  rxJ[1,1]*Qr_ID + sxJ[1,1]*Qs_ID
cij_y =  ryJ[1,1]*Qr_ID + syJ[1,1]*Qs_ID

C = sqrt.(cij_x.*cij_x+cij_y.*cij_y)
C_x = cij_x./C; C_y = cij_y./C
replace!(C_x, NaN=>0); replace!(C_y, NaN=>0);
"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dT = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dT))
dt = T/Nsteps


"initial conditions"
xq = Vq*x
yq = Vq*y
# u = @. sin(pi*xq)*sin(pi*yq)
# u = @. exp(-25*(xq^2+yq^2))
# u0 = u
# h = @. exp(-25*( xq^2+yq^2))+2

h = ones(size(xq))
# col_idx = zeros(convert(Int, size(h,2)/4) )
# for i = 0:convert(Int,K1D/2)-1
#     idx_start = convert(Int, K1D*K1D*2/4) + i*K1D*2 + convert(Int,K1D/2)
#     col_idx[i*K1D+1:(i+1)*K1D] = idx_start+1:idx_start+convert(Int,K1D)
# end
# col_idx = convert.(Int, col_idx)
# h[:,col_idx] .= 1e-10;# h[:,K1D+1:end] .= 1

h[:,1:K1D] .= 1e-10; h[:,K1D+1:end] .= 1


hu = h*0
hv = h*0
u = (h, hu, hv)

"pack arguments into tuples"
ops = (Qrskew_ID,Qsskew_ID, Qrskew_ES,Qsskew_ES,E, M_inv, Mf_inv, Pf)
dis_cst = (Cf, C, C_x, C_y)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ, nx, ny)
nodemaps = (mapP,mapB)

function swe_2d_esdg_surface(UL, UR, dU, Pf, c)
    (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D(UL,UR,g)
    dh, dhu, dhv = dU
    (hf, huf, hvf)=UL;
    (hP, huP, hvP)=UR;
    fs1 = @. fxS1*nxJ + fyS1*nyJ;
    fs2 = @. fxS2*nxJ + fyS2*nyJ;
    fs3 = @. fxS3*nxJ + fyS3*nyJ;
    # @show findmin(hf)
    tau = 1
    f1_ES =  Pf*(fs1 .- .5*tau*c.*(hP.-hf).*sJ);
    f2_ES =  Pf*(fs2 .- .5*tau*c.*(huP.-huf).*sJ);
    f3_ES =  Pf*(fs3 .- .5*tau*c.*(hvP.-hvf).*sJ);
    # f1_ES =  fs1 .- .5*tau*c.*(hP.-hf).*sJ;
    # f2_ES =  fs2 .- .5*tau*c.*(huP.-huf).*sJ;
    # f3_ES =  fs3 .- .5*tau*c.*(hvP.-hvf).*sJ;
    return f1_ES, f2_ES, f3_ES
end

function swe_2d_esdg_vol(UL_E, UR_E, Qr_ES, Qs_ES, vgeo_e, i, j)
    (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    (FxV1,FxV2,FxV3),(FyV1,FyV2,FyV3) = fS2D(UL_E,UR_E,g)
    # QNx_ij = Qr_ES[i,j]*(rxJ[i,e]+ rxJ[j,e]) + Qs_ES[i,j]*(sxJ[i,e]+ sxJ[j,e]);
    # QNy_ij = Qr_ES[i,j]*(ryJ[i,e]+ ryJ[j,e]) + Qs_ES[i,j]*(syJ[i,e]+ syJ[j,e]);
    QNx_ij = Qr_ES[i,j]*(rxJ_i*2) + Qs_ES[i,j]*(sxJ_i*2);
    QNy_ij = Qr_ES[i,j]*(ryJ_i*2) + Qs_ES[i,j]*(syJ_i*2);
    fv1_ES = (QNx_ij*FxV1 + QNy_ij*FyV1);
    fv2_ES = (QNx_ij*FxV2 + QNy_ij*FyV2);
    fv3_ES = (QNx_ij*FxV3 + QNy_ij*FyV3);
    return fv1_ES, fv2_ES, fv3_ES
end

function swe_2d_ID_surface(UL, UR, dU, Pf, c)
    (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D_LF(UL,UR,g)
    dh, dhu, dhv = dU
    fs1 = @. fxS1*nxJ + fyS1*nyJ;
    fs2 = @. fxS2*nxJ + fyS2*nyJ;
    fs3 = @. fxS3*nxJ + fyS3*nyJ;

    f1_ID = Pf * (0.5*fs1) - transpose(E)*(Cf.*c.*dh)
    f2_ID = Pf * (0.5*fs2) - transpose(E)*(Cf.*c.*dhu)
    f3_ID = Pf * (0.5*fs3) - transpose(E)*(Cf.*c.*dhv)
    return f1_ID, f2_ID, f3_ID
end

function swe_2d_ID_vol(UL_E, Qr_ID, Qs_ID, vgeo_e, i, j)
    (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    Qr_ID_ij = Qr_ID[i,j]; Qs_ID_ij = Qs_ID[i,j];
    dhdx  = rxJ_i*(Qr_ID_ij*fxV1) + sxJ_i*(Qs_ID_ij*fxV1)
    dhudx = rxJ_i*(Qr_ID_ij*fxV2) + sxJ_i*(Qs_ID_ij*fxV2)
    dhvdx = rxJ_i*(Qr_ID_ij*fxV3) + sxJ_i*(Qs_ID_ij*fxV3)

    dhdy  = ryJ_i*(Qr_ID_ij*fyV1) + syJ_i*(Qs_ID_ij*fyV1)
    dhudy = ryJ_i*(Qr_ID_ij*fyV2) + syJ_i*(Qs_ID_ij*fyV2)
    dhvdy = ryJ_i*(Qr_ID_ij*fyV3) + syJ_i*(Qs_ID_ij*fyV3)
    fv1_ID = dhdx  + dhdy
    fv2_ID = dhudx + dhudy
    fv3_ID = dhvdx + dhvdy
    return fv1_ID, fv2_ID, fv3_ID
end

function swe_2d_ID_h(UL_E, Qr_ID, Qs_ID, vgeo_e, i, j)
    (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    Qr_ID_ij = Qr_ID[i,j]; Qs_ID_ij = Qs_ID[i,j];
    dhdx  = rxJ_i*(Qr_ID_ij*fxV1) + sxJ_i*(Qs_ID_ij*fxV1)
    dhdy  = ryJ_i*(Qr_ID_ij*fyV1) + syJ_i*(Qs_ID_ij*fyV1)
    fv1_ID = dhdx  + dhdy
    return fv1_ID
end

function swe_2d_rhs(U,ops,dis_cst,vgeo,fgeo,nodemaps, dt, tol)
    # unpack args
    h, hu, hv = U
    Qr_ID,Qs_ID,Qr_ES,Qs_ES,E,M_inv,EfTW= ops
    Cf, C, C_x, C_y = dis_cst
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ,nx,ny = fgeo
    (mapP,mapB) = nodemaps
    u = hu./h; v = hv./h
    uf = E*u; vf = E*v
    hf  = E*h; huf = E*hu; hvf = E*hv
    hP  = hf[mapP]; huP = huf[mapP] ;hvP = hvf[mapP]
    dh  = hP-hf; dhu = huP-huf; dhv = hvP-hvf
    dU = (dh, dhu, dhv);
    UL = (hf, huf, hvf); UR = (hP, huP, hvP)
    lambdaf = abs.(uf.*nx+vf.*ny) + sqrt.(g*hf)
    lambdaP = lambdaf[mapP]
    c = max.(abs.(lambdaf), abs.(lambdaP))

    ##surface part
    #ES part
    f1_ES, f2_ES, f3_ES = swe_2d_esdg_surface(UL, UR, dU, Pf, c);
    # @show norm(f1_ES_f - f1_ES), norm(f2_ES_f - f2_ES), norm(f3_ES_f - f3_ES)
    #ID part
    f1_ID, f2_ID, f3_ID = swe_2d_ID_surface(UL, UR, dU, Pf, c);

    # rhs1_CL = zeros(size(h)); rhs2_CL = zeros(size(h)); rhs3_CL = zeros(size(h));
    # rhs1_ES = zeros(size(h)); rhs2_ES = zeros(size(h)); rhs3_ES = zeros(size(h));
    rhs1_ID = zeros(size(h)); #rhs2_ID = zeros(size(h)); rhs3_ID = zeros(size(h));
    Nq = size(h,1); Nfq = size(E,1);
    #build low order solution first
    for e = 1:size(h,2)
        rxJ_i = rxJ[1,e]; sxJ_i = sxJ[1,e];
        ryJ_i = ryJ[1,e]; syJ_i = syJ[1,e];
        vgeo_e = (rxJ_i, sxJ_i, ryJ_i, syJ_i);
        for i=1:Nq
            UL_E = (h[i,e], hu[i,e], hv[i,e]);
            for j=i:Nq
                UR_E = (h[j,e], hu[j,e], hv[j,e])
                cij = C[i,j]
                if cij!=0
                    fv1_i_ID = swe_2d_ID_h(UR_E, Qr_ID, Qs_ID, vgeo_e, i, j);
                    fv1_j_ID = swe_2d_ID_h(UL_E, Qr_ID, Qs_ID, vgeo_e, j, i);

                    rhs1_ID[i,e] += fv1_i_ID; rhs1_ID[j,e] += fv1_j_ID;
                    lambda_i = abs.(u[i,e].*C_x[i,j]+v[i,e].*C_y[i,j])+sqrt.(g.*h[i,e])
                    lambda_j = abs.(u[j,e].*C_x[j,i]+v[j,e].*C_y[j,i])+sqrt.(g.*h[j,e])
                    lambda = max(lambda_i, lambda_j)
                    d1 = cij * lambda * (h[j,e]  - h[i,e]);
                    rhs1_ID[i,e] -= d1; rhs1_ID[j,e] += d1;
                end
            end
        end
    end
    rhs1_ID = rhs1_ID + f1_ID;
    h_L_next = h -M_inv * rhs1_ID *dt;
    h_L_next_f = E*h_L_next;
    h_L_next_P  = h_L_next_f[mapP]
    f1_IDf = E*f1_ID; f1_ESf = E*f1_ES;
    lf = (h_L_next_f .-tol)./(Nq*M_inv[1:Nfq,1:Nfq]*(f1_ESf-f1_IDf)*dt);
    lf = min.(lf, lf[mapP]);
    lf = min.(1, lf);
    lf = max.(lf,0);
    for e = 1:size(h,2)
        for i = 1:Nfq
            if h_L_next_f[i,e]< tol || h_L_next_P[i,e]< tol
                lf[i,e] = 0;
            end
        end
    end
    lf = E'*lf;
    # lf = zeros(size(h));

    rhs1_CL = f1_ID + lf.*(f1_ES - f1_ID);
    rhs2_CL = f2_ID + lf.*(f2_ES - f2_ID);
    rhs3_CL = f3_ID + lf.*(f3_ES - f3_ID);

    ##volume part
    # loop over all elements
    for e = 1:size(h,2)
        rxJ_i = rxJ[1,e]; sxJ_i = sxJ[1,e];
        ryJ_i = ryJ[1,e]; syJ_i = syJ[1,e];
        vgeo_e = (rxJ_i, sxJ_i, ryJ_i, syJ_i);
        for i=1:Nq
            UL_E = (h[i,e], hu[i,e], hv[i,e]);
            for j=i:Nq
                UR_E = (h[j,e], hu[j,e], hv[j,e])
                fv1_ES, fv2_ES, fv3_ES = swe_2d_esdg_vol(UL_E, UR_E, Qr_ES, Qs_ES, vgeo_e, i, j)

                cij = C[i,j]
                fv1_i_ID = 0; fv2_i_ID = 0; fv3_i_ID = 0;
                fv1_j_ID = 0; fv2_j_ID = 0; fv3_j_ID = 0;
                if C[i,j]!=0
                    fv1_i_ID, fv2_i_ID, fv3_i_ID = swe_2d_ID_vol(UR_E, Qr_ID, Qs_ID, vgeo_e, i, j);
                    fv1_j_ID, fv2_j_ID, fv3_j_ID = swe_2d_ID_vol(UL_E, Qr_ID, Qs_ID, vgeo_e, j, i);

                    lambda_i = abs.(u[i,e].*C_x[i,j]+v[i,e].*C_y[i,j])+sqrt.(g.*h[i,e])
                    lambda_j = abs.(u[j,e].*C_x[j,i]+v[j,e].*C_y[j,i])+sqrt.(g.*h[j,e])
                    lambda = max(lambda_i, lambda_j)
                    d1 = cij * lambda * (h[j,e]  - h[i,e]);
                    d2 = cij * lambda * (hu[j,e] - hu[i,e]);
                    d3 = cij * lambda * (hv[j,e] - hv[i,e]);
                    fv1_i_ID -= d1
                    fv2_i_ID -= d2
                    fv3_i_ID -= d3
                    fv1_j_ID += d1
                    fv2_j_ID += d2
                    fv3_j_ID += d3

                end
                l_ij = (h_L_next[i,e] -tol)/(Nq*M_inv[i,i]*(fv1_ES-fv1_i_ID)*dt);
                l_ji = (h_L_next[j,e] -tol)/(Nq*M_inv[j,j]*(-fv1_ES-fv1_j_ID)*dt);
                l = min(l_ij, l_ij);
                l = min(1, l);
                l = max(l,0);
                if h_L_next[i,e]< tol || h_L_next[j,e]< tol || fv1_ES-fv1_i_ID < tol
                    l = 0;
                end
                # l = 0;
                rhs1_CL[i,e] += fv1_i_ID + l * (fv1_ES-fv1_i_ID);
                rhs2_CL[i,e] += fv2_i_ID + l * (fv2_ES-fv2_i_ID);
                rhs3_CL[i,e] += fv3_i_ID + l * (fv3_ES-fv3_i_ID);
                rhs1_CL[j,e] += fv1_j_ID + l * (-fv1_ES-fv1_j_ID);
                rhs2_CL[j,e] += fv2_j_ID + l * (-fv2_ES-fv2_j_ID);
                rhs3_CL[j,e] += fv3_j_ID + l * (-fv3_ES-fv3_j_ID);
            end
        end
    end
    return -M_inv*rhs1_CL, -M_inv*rhs2_CL, -M_inv*rhs3_CL
end


tol = 1e-8
DT = zeros(MAXIT)
t = 0
t_plot = dT*10
global i;
@time begin
for i = 1:MAXIT
    if i%1000 == 0
        @show i, t
    end
    global h, hu, hv, u, t, t_plot

    # Heun's method - this is an example of a 2nd order SSP RK method
    # local rhs_ES1, rhs_ID1 = swe_2d_rhs(u,ops,dis_cst,vgeo,fgeo,nodemaps, dt1)
    lambda = maximum(sqrt.((hu./h).^2+(hv./h).^2)+sqrt.(g.*h))
    dt1 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    local rhs_1 = swe_2d_rhs(u,ops,dis_cst,vgeo,fgeo,nodemaps, dt1, tol)

    # rhs_1, L1 = convex_limiter(rhs_ES1, rhs_ID1, h, h, tol, dt1)
    # if dt1<dT
    #     rhs_1 = rhs_ID1;
    # end
    htmp  = h  + dt1*rhs_1[1]
    hutmp = hu + dt1*rhs_1[2]
    hvtmp = hv + dt1*rhs_1[3]

    # utmp = (htmp, hutmp)
    hutmp[findall(x->x<2*tol, htmp)] .= 0
    hvtmp[findall(x->x<2*tol, htmp)] .= 0
    # @show L1, htmp dt1

    h_min, pos = findmin(htmp)
    if h_min <= 0
       # @show L1
       error("htmp_min<0 ", h_min, pos, "iteration ", i )
    end
    lambda = maximum(sqrt.((hutmp./htmp).^2+(hvtmp./htmp).^2)+sqrt.(g.*htmp))
    dt2 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    while dt2<dt1
        dt1 = dt2
        htmp  = h  + dt1*rhs_1[1]
        hutmp = hu + dt1*rhs_1[2]
        hvtmp = hv + dt1*rhs_1[3]
        lambda = maximum(sqrt.((hutmp./htmp).^2+(hvtmp./htmp).^2)+sqrt.(g.*htmp))
        dt2 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
        # @show dt2
    end
    utmp = (htmp, hutmp, hvtmp)
    rhs_2 = swe_2d_rhs(utmp,ops,dis_cst,vgeo,fgeo,nodemaps, dt2, tol)
    dt = min(dt1, dt2)
    # rhs_2, L2 = convex_limiter(rhs_ES2, rhs_ID2, htmp, h, tol, dt)
    # rhsh_ES = rhsh_ES2; rhshu_ES = rhshu_ES2; rhsh_ID = rhsh_ID2; rhshu_ID = rhshu_ID2;
    # dt = min(T-t, minimum(w)/(2*lambda), dT);
    # if dt1<dT
    #     rhs_2 = rhs_ID2;
    # end
    # s1 = sum(h)+sum(hu)+sum(hv)
    h  .+= .5*dt*(rhs_1[1] + rhs_2[1])
    hu .+= .5*dt*(rhs_1[2] + rhs_2[2])
    hv .+= .5*dt*(rhs_1[3] + rhs_2[3])
    hu[findall(x->x<2*tol, h)] .= 0
    hv[findall(x->x<2*tol, h)] .= 0
    # @show L2, h, dt
    h_min, pos = findmin(h)
    # s2 = sum(h)+sum(hu)+sum(hv)
    # @show norm(s2-s1)
    # @show sum(rhs_1[1] + rhs_2[1])
    # @show sum(rhs_1[2] + rhs_2[2])
    # @show sum(rhs_1[3] + rhs_2[3])
    # @show norm(rhs_1[1] + rhs_2[1])
    # @show norm(rhs_1[2] + rhs_2[2])
    # @show norm(rhs_1[3] + rhs_2[3])
    if h_min <=0
        # @show L2
        @show maximum(hu./h)
        error("h_min<0 ", h_min, pos, "iteration ", i )
    end
    u = (h,hu,hv)
    t +=dt
    if t>=T
            break
    end
    # @show L1, L2
    DT[i] = dt
    i +=1
    if t>= t_plot #|| i==Nsteps
        t_plot += dT*10;
        # println("Number of time steps $i out of $Nsteps")
    end
end
end
DT = DT[1:findmin(DT)[2]-1];

gr(aspect_ratio=1,legend=false,
   markerstrokewidth=0,markersize=2)

# plotting nodes"
rp, sp = equi_nodes(10)
Vp = vandermonde(N,rp,sp)/vandermonde(N,r,s)
vv = Vp*Pq*h
Plots.scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
