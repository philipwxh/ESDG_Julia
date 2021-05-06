using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using ForwardDiff
using SparseArrays
using StaticArrays
push!(LOAD_PATH, "./src") # user defined modules
using Basis2DTri
using NodesAndModes
using NodesAndModes.Tri
using UnPack
using StartUpDG
using StartUpDG.ExplicitTimestepUtils
using DelimitedFiles
include("dg2d_swe_flux.jl")
include("dg2d_swe_mesh_opt.jl")

g = 1.0
"Approximation parameters"
N   = 3 # The order of approximation
K1D = 16
CFL = 1/4
T   = 0.5 # endtimeA
MAXIT = 1000000

ts_ft= 1/2
tol = 1e-32
t_plot = [T];#[1/3,2/3,1.0];
qnode_choice = "GQ" #"GQ" "GL" "tri_diage"

# Mesh related variables
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

# Construct matrices on reference elements
rd = init_reference_tri_sbp_GQ(N, qnode_choice)
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
btm = cos.(pi*xq) .+1;
# btm = @. 5*exp(-25*(xq^2+yq^2))
# btm = btm*0;
h = xq*0 .+ 1.5  - btm;
# h = exp.(-25*((xq.+0.5).^2)) .+2 - btm;
h[findall(x->x<tol, h)] .= tol;
# h = xq*0 .+ 5 - btm;


# u = @. sin(pi*xq)*sin(pi*yq)
# u = @. exp(-25*(xq^2+yq^2))
# u0 = u
# h = @. exp(-25*( xq^2+yq^2))+2
h0 = copy(h)
# h = ones(size(xq))
# col_idx = zeros(convert(Int, size(h,2)/4) )
# for i = 0:convert(Int,K1D/2)-1
#     idx_start = convert(Int, K1D*K1D*2/4) + i*K1D*2 + convert(Int,K1D/2)
#     col_idx[i*K1D+1:(i+1)*K1D] = idx_start+1:idx_start+convert(Int,K1D)
# end
# col_idx = convert.(Int, col_idx)
# h[:,col_idx] .= 1e-10;# h[:,K1D+1:end] .= 1

# h[:,1:K1D] .= 1e-10; h[:,K1D+1:end] .= 1

######
# a = 2; B = 2; h0 = 8;
#
# btm = h0.*(xq./a).^2
# omega = sqrt(2*g*h0)/a
#
# h = h0 .- B^2/(4*g)*cos(2*omega*0) .- B^2/(4*g) .- (B*xq)/(2*a)*sqrt(8*h0/g)*cos(omega*0);
# h = h - btm;
# h[findall(x->x<tol, h)] .= tol;
#######

hu = h*0;
# hu[findall(x->x>2.001, h)] .= 1;
hv = h*0;
u = (h, hu, hv)

"pack arguments into tuples"
ops = ( Qrskew_ID,Qsskew_ID, Qr_ID, Qs_ID,
        Qrskew_ES,Qsskew_ES, QNr_sbp, QNs_sbp,
        E, M_inv, Pf);
dis_cst = (Cf, C, C_x, C_y)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ, nx, ny)
nodemaps = (mapP,mapB)
(gQNxb, gQNyb) =  ESDG_bottom(QNr_sbp, QNs_sbp, btm, vgeo, g);
u = (h, hu, hv, btm, gQNxb, gQNyb)


function swe_2d_rhs(U,ops,dis_cst,vgeo,fgeo,nodemaps, dt, tol, g)
    # unpack args
    h, hu, hv, btm, gQNxb, gQNyb  = U
    Qr_ID,Qs_ID,Qrb_ID,Qsb_ID,Qr_ES,Qs_ES,QNr_sbp, QNs_sbp, E,M_inv,Pf= ops
    Cf, C, C_x, C_y = dis_cst
    rxJ,sxJ,ryJ,syJ,J = vgeo
    # nxJ,nyJ,sJ,nx,ny = fgeo
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
    f1_ES, f2_ES, f3_ES = swe_2d_esdg_surface(UL, UR, dU, Pf, fgeo, c, g);
    # @show norm(f1_ES_f - f1_ES), norm(f2_ES_f - f2_ES), norm(f3_ES_f - f3_ES)
    #ID part
    f1_ID, f2_ID, f3_ID = swe_2d_ID_surface(UL, UR, dU, Pf, fgeo, c, g);
    # f1_ID, f2_ID, f3_ID = swe_2d_esdg_surface(UL, UR, dU, Pf, c);

    rhs1_ID = zeros(Float64, size(h));
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
                    fv1_i_ID = swe_2d_ID_h(UR_E, Qr_ID, Qs_ID, vgeo_e, i, j, g);
                    fv1_j_ID = swe_2d_ID_h(UL_E, Qr_ID, Qs_ID, vgeo_e, j, i, g);

                    rhs1_ID[i,e] += fv1_i_ID; rhs1_ID[j,e] += fv1_j_ID;
                    lambda_i = abs(u[i,e]*C_x[i,j]+v[i,e]*C_y[i,j])+sqrt(g*h[i,e])
                    lambda_j = abs(u[j,e]*C_x[j,i]+v[j,e]*C_y[j,i])+sqrt(g*h[j,e])
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
    f1_IDf = E*f1_ID; f1_ESf = E*f1_ES;
    lf = (h_L_next_f .-tol)./(Nq*M_inv[1:Nfq,1:Nfq]*(f1_ESf-f1_IDf)*dt);
    lf[findall(x->x<tol, f1_ESf-f1_IDf)] .= 1;
    for e = 1:size(h,2)
        for i = 1:Nfq
            if h_L_next_f[i,e]< tol || hf[i,e]< tol
                lf[i,e] = 0.0;
                lf[mapP[i,e]]=0.0;
            end
        end
    end
    lf = min.(lf, lf[mapP]); lf = min.(1.0, lf); lf = max.(lf,0.0);
    lf = E'*lf;
    # lf = zeros(size(h));
    # lf = ones(size(lf));
    rhs1_CL = f1_ID + lf.*(f1_ES - f1_ID);
    rhs2_CL = f2_ID + lf.*(f2_ES - f2_ID);
    rhs3_CL = f3_ID + lf.*(f3_ES - f3_ID);

    ##volume part
    # loop over all elements
    for e = 1:size(h,2)
        rxJ_i = rxJ[1,e]; sxJ_i = sxJ[1,e];
        ryJ_i = ryJ[1,e]; syJ_i = syJ[1,e];
        vgeo_e = (rxJ_i, sxJ_i, ryJ_i, syJ_i);
        b_e = btm[:,e];
        for i=1:Nq
            UL_E = (h[i,e], hu[i,e], hv[i,e]);
            for j=i:Nq
                UR_E = (h[j,e], hu[j,e], hv[j,e])
                fv1_i_ES, fv2_i_ES, fv3_i_ES, fv1_j_ES, fv2_j_ES, fv3_j_ES = swe_2d_esdg_vol(UL_E, UR_E, ops, vgeo_e, i, j, b_e, g)

                cij = C[i,j]
                fv1_i_ID = 0.0; fv2_i_ID = 0.0; fv3_i_ID = 0.0;
                fv1_j_ID = 0.0; fv2_j_ID = 0.0; fv3_j_ID = 0.0;
                if C[i,j]!=0 || i == j
                    fv1_i_ID, fv2_i_ID, fv3_i_ID = swe_2d_ID_vol(UR_E, ops, vgeo_e, i, j, b_e, g);
                    fv1_j_ID, fv2_j_ID, fv3_j_ID = swe_2d_ID_vol(UL_E, ops, vgeo_e, j, i, b_e, g);

                    # fv1_i_ID, fv2_i_ID, fv3_i_ID = swe_2d_ID_vol(UR_E, Qr_ID, Qs_ID, vgeo_e, i, j);
                    # fv1_j_ID, fv2_j_ID, fv3_j_ID = swe_2d_ID_vol(UL_E, Qr_ID, Qs_ID, vgeo_e, j, i);
                    lambda_i = abs(u[i,e]*C_x[i,j]+v[i,e]*C_y[i,j])+sqrt(g*h[i,e])
                    lambda_j = abs(u[j,e]*C_x[j,i]+v[j,e]*C_y[j,i])+sqrt(g*h[j,e])
                    lambda = max(lambda_i, lambda_j)
                    # d1 = 0; d2 = 0; d3 = 0
                    # if h[i,e]>tol &&  h[i,e]>tol
                    # d1 = cij * lambda * (h[j,e] + b_e[j]  - h[i,e] - b_e[i]);
                    d1 = cij * lambda * (h[j,e] - h[i,e]);
                    # if h[i,e]<=tol ||  h[j,e]<=tol
                    #     d1 = 0;
                    # end
                    # d1 = 0;
                    d2 = cij * lambda * (hu[j,e] - hu[i,e]);
                    d3 = cij * lambda * (hv[j,e] - hv[i,e]);
                    # end
                    fv1_i_ID -= d1
                    fv2_i_ID -= d2
                    fv3_i_ID -= d3
                    fv1_j_ID += d1
                    fv2_j_ID += d2
                    fv3_j_ID += d3
                end
                l_ij = (h_L_next[i,e] -tol)/(Nq*M_inv[i,i]*(fv1_i_ES-fv1_i_ID)*dt);
                if fv1_i_ES-fv1_i_ID<tol
                    l_ij = 1.0
                end
                l_ji = (h_L_next[j,e] -tol)/(Nq*M_inv[j,j]*(fv1_j_ES-fv1_j_ID)*dt);
                if fv1_j_ES-fv1_j_ID<tol
                    l_ji = 1.0
                end
                l = min(l_ij, l_ji);
                l = min(1.0,l);
                l = max(l,0.0);

                if h[i,e] < tol || h[j,e] < tol || h_L_next[i,e]< tol || h_L_next[j,e]< tol #|| fv1_ES-fv1_i_ID < tol
                    l = 0.0;
                end

                # l = 0.0;
                rhs1_CL[i,e] += fv1_i_ID + l * (fv1_i_ES-fv1_i_ID);
                rhs2_CL[i,e] += fv2_i_ID + l * (fv2_i_ES-fv2_i_ID);
                rhs3_CL[i,e] += fv3_i_ID + l * (fv3_i_ES-fv3_i_ID);
                if i!= j
                    rhs1_CL[j,e] += fv1_j_ID + l * (fv1_j_ES-fv1_j_ID);
                    rhs2_CL[j,e] += fv2_j_ID + l * (fv2_j_ES-fv2_j_ID);
                    rhs3_CL[j,e] += fv3_j_ID + l * (fv3_j_ES-fv3_j_ID);
                end
            end
        end
    end
    return -M_inv*rhs1_CL, -M_inv*rhs2_CL, -M_inv*rhs3_CL
end

DT = zeros(MAXIT)
t = 0
pl_idx = 1;
global i;
# filename = string("h",string(N),"_",string(K1D),"_","0_wave_bump.csv");
# writedlm( filename,  h, ',');
@time begin
for i = 1:MAXIT
    if i%1000 == 0
        @show i, t
    end
    global h, hu, hv, u, t, t_plot, pl_idx

    # Heun's method - this is an example of a 2nd order SSP RK method
    # local rhs_ES1, rhs_ID1 = swe_2d_rhs(u,ops,dis_cst,vgeo,fgeo,nodemaps, dt1)
    lambda = maximum(sqrt.((hu./h).^2+(hv./h).^2)+sqrt.(g.*h))
    # dt1 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    dt1 = min(min(T,t_plot[pl_idx])-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    rhs_1 = swe_2d_rhs(u,ops,dis_cst,vgeo,fgeo,nodemaps, dt1, tol, g)

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
    # dt2 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    dt2 = min(min(T,t_plot[pl_idx])-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    while dt2<dt1
        dt1 = dt2
        htmp  = h  + dt1*rhs_1[1]
        hutmp = hu + dt1*rhs_1[2]
        hvtmp = hv + dt1*rhs_1[3]
        lambda = maximum(sqrt.((hutmp./htmp).^2+(hvtmp./htmp).^2)+sqrt.(g.*htmp))
        # dt2 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
        dt2 = min(min(T,t_plot[pl_idx])-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    end
    utmp = (htmp, hutmp, hvtmp, btm, gQNxb, gQNyb)
    rhs_2 = swe_2d_rhs(utmp,ops,dis_cst,vgeo,fgeo,nodemaps, dt2, tol, g)
    dt = min(dt1, dt2)

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
    u = (h,hu,hv, btm, gQNxb, gQNyb)
    t +=dt
    # @show L1, L2
    DT[i] = dt
    i +=1
    # if t>= t_plot[pl_idx] #|| i==Nsteps
    #     # @show t
    #     filename = string("h",string(N),"_",string(K1D),"_",pl_idx,"_wave_bump.csv");
    #     writedlm( filename,  h, ',');
    #     pl_idx+=1;
    #     # t_plot += dT*10;
    #     # println("Number of time steps $i out of $Nsteps")
    # end
    if t>=T
            break
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
# Plots.scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
Plots.scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(45,45))
