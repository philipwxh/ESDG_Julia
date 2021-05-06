using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using ForwardDiff
using SparseArrays
using StaticArrays

# push!(LOAD_PATH, "./src") # user defined modules
# using CommonUtils
using NodesAndModes
# using NodesAndModes.Line
push!(LOAD_PATH, "./src") # user defined modules
using Basis1D
# using SetupDG
using UnPack

# push!(LOAD_PATH, "./StartUpDG/src") # user defined modules
using StartUpDG
using StartUpDG.ExplicitTimestepUtils
include("dg1d_swe_flux.jl")

global g = 1
"Approximation parameters"
N   = 3 # The order of approximation
K1D = 32
CFL = 1/4
T   = 1 # endtime
MAXIT = 100#0000
tol = 1e-16
t_plot = [T];
global ts_ft = 2;

"Mesh related variables"
VX = LinRange(-5,5,K1D+1)
EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]),2,K1D))

"Construct matrices on reference elements"
r,w = NodesAndModes.gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/V
M = inv(V*V')

Q_ID,E,B,ψ = make_meshfree_ops(r, w)
E = zeros(size(E)); E[1,1] = 1; E[end,end]=1;
Q_IDskew = .5*(Q_ID-transpose(Q_ID))
Q_ES = M*Dr
Q_ESskew = .5*(Q_ES-transpose(Q_ES))

"Nodes on faces, and face node coordinate"
wf = [1;1]
Vf = vandermonde_1D(N,[-1;1])/V
LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix

"Construct global coordinates"
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
x = V1*VX[transpose(EToV)]

"Connectivity maps"
xf = Vf*x
mapM = reshape(1:2*K1D,2,K1D)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

"Make periodic"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K1D)
rxJ = 1

M_inv = diagm(1 ./(w.*J[1]))
Mf_inv = E*M_inv*E'
# M_inv = spdiagm(0 => 1 ./vec(diagm(w)*J))

"initial conditions"
a = 2; B = 2; h0 = 8;

btm = h0.*(x./a).^2
omega = sqrt(2*g*h0)/a

h = h0 .- B^2/(4*g)*cos(2*omega*0) .- B^2/(4*g) .- (B*x)/(2*a)*sqrt(8*h0/g)*cos(omega*0);
h = h - btm;
h[findall(x->x<tol, h)] .= tol;

hu = h*0;
h_0 = copy(h);

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dT = T/Nsteps

"pack arguments into tuples - will "
u    = (h, hu, btm)
ops  = (Q_IDskew, Q_ID, Q_ESskew, Q_ES, E, M_inv, Mf_inv)
vgeo = (rxJ,J)
fgeo = (nxJ)
nodemaps = (mapM, mapP)

function swe_1d_rhs(h, hu, btm, ops, vgeo, fgeo, mapP, dt, g, tol)
    # h, hu = U
    Q_ID, Qb_ID, Q_ES, Qb_ES, E, M_inv, Mf_inv = ops
    rxJ,J = vgeo
    nxJ = fgeo

    u = hu./h;
    hf = E*h; hP = hf[mapP]; dh = hP-hf;
    huf = E*hu; huP = huf[mapP]; dhu = huP - huf;
    uf = E*u; uP = uf[mapP]; du = uP - uf;
    UL = (hf, huf); UR = (hP, huP); dU = (dh, dhu);

    lambdaC = abs.(u)+sqrt.(g*h)
    lambdaM = E*lambdaC
    lambdaP = lambdaM[mapP]
    c = max.(lambdaM, lambdaP)
    tau = 1

    ##surface part
    #ES part
    f1_ES, f2_ES = swe_1d_esdg_surface(UL, UR, dU, E, nxJ, c, g);
    # @show norm(f1_ES_f - f1_ES), norm(f2_ES_f - f2_ES), norm(f3_ES_f - f3_ES)
    #ID part
    f1_ID, f2_ID = swe_1d_ID_surface(UL, UR, dU, E, nxJ, c, g);
    # f1_ID, f2_ID, f3_ID = swe_2d_esdg_surface(UL, UR, dU, Pf, c);


    rhs1_ID = zeros(Float64, size(h));
    Nq = size(h,1); Nfq = size(E,1);
    cij = 1/2
    #build low order solution first
    for e = 1:size(h,2)
        for i=1:Nq
            UL_E = (h[i,e], hu[i,e]);
            for j=i:Nq
                UR_E = (h[j,e], hu[j,e])
                if Q_ID[i,j]!=0
                    fv1_i_ID = swe_1d_ID_h(UR_E, Q_ID, i, j, g);
                    fv1_j_ID = swe_1d_ID_h(UL_E, Q_ID, j, i, g);

                    rhs1_ID[i,e] += fv1_i_ID; rhs1_ID[j,e] += fv1_j_ID;
                    lambda_i = abs(u[i,e]*cij)+sqrt(g*h[i,e])
                    lambda_j = abs(u[j,e]*cij)+sqrt(g*h[j,e])
                    lambda_ij = max(lambda_i, lambda_j)
                    d1 = cij * 2 *lambda_ij * (h[j,e]  - h[i,e]);
                    rhs1_ID[i,e] -= d1; rhs1_ID[j,e] += d1;
                end
            end
        end
    end

    rhs1_ID = rhs1_ID + f1_ID;
    h_L_next = h - M_inv * rhs1_ID *dt;
    h_L_next_f = E*h_L_next;
    f1_IDf = E*f1_ID; f1_ESf = E*f1_ES;
    lf = (h_L_next_f .- tol)./(Nq*Mf_inv*(f1_ESf-f1_IDf)*dt);
    lf[findall(x->x<=0, f1_ESf-f1_IDf)] .= 1;
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

    ##volume part
    # loop over all elements
    for e = 1:size(h,2)
        b_e = btm[:,e];
        for i=1:Nq
            UL_E = (h[i,e], hu[i,e]);
            for j=i:Nq
                UR_E = (h[j,e], hu[j,e])
                fv1_i_ES, fv2_i_ES, fv1_j_ES, fv2_j_ES = swe_1d_esdg_vol(UL_E, UR_E, ops, vgeo, i, j, b_e, g)

                fv1_i_ID = 0.0; fv2_i_ID = 0.0; fv3_i_ID = 0.0;
                fv1_j_ID = 0.0; fv2_j_ID = 0.0; fv3_j_ID = 0.0;
                if Q_ID[i,j]!=0 || i == j
                    fv1_i_ID, fv2_i_ID = swe_1d_ID_vol(UR_E, ops, vgeo, i, j, b_e, g);
                    fv1_j_ID, fv2_j_ID = swe_1d_ID_vol(UL_E, ops, vgeo, j, i, b_e, g);

                    lambda_i = abs(u[i,e]*cij)+sqrt(g*h[i,e])
                    lambda_j = abs(u[j,e]*cij)+sqrt(g*h[j,e])
                    lambda = max(lambda_i, lambda_j)
                    # d1 = 0; d2 = 0; d3 = 0
                    # if h[i,e]>tol &&  h[i,e]>tol
                    # d1 = cij * lambda * (h[j,e] + b_e[j]  - h[i,e] - b_e[i]);
                    d1 = cij * 2 *lambda * (h[j,e] - h[i,e]);
                    # if h[i,e]<=tol ||  h[j,e]<=tol
                    #     d1 = 0;
                    # end
                    # d1 = 0;
                    d2 = cij * 2 *lambda * (hu[j,e] - hu[i,e]);
                    # end
                    fv1_i_ID -= d1
                    fv2_i_ID -= d2
                    fv1_j_ID += d1
                    fv2_j_ID += d2
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
                if i!= j
                    rhs1_CL[j,e] += fv1_j_ID + l * (fv1_j_ES-fv1_j_ID);
                    rhs2_CL[j,e] += fv2_j_ID + l * (fv2_j_ES-fv2_j_ID);
                end
            end
        end
    end
    return -M_inv*rhs1_CL, -M_inv*rhs2_CL
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*h)
resu = zeros(size(x))

DT = zeros(MAXIT)
t = 0
pl_idx = 1;
global i;
@time begin
@gif for i = 1:MAXIT
    if i%100 == 0
        @show i, t
    end
    global h, hu, u, t, t_plot, pl_idx
    # for INTRK = 1:5
    #     rhsh = rhs(u,ops,vgeo,fgeo,mapP)
    #     # rhsh = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsh
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    lambda = maximum(abs.(hu./h)+sqrt.(g.*h))
    # dt1 = min(T-t, minimum(wq)*J[1]/(ts_ft*lambda), dT);
    dt1 = min(min(T,t_plot[pl_idx])-t, minimum(w)*J[1]/(ts_ft*lambda), dT);
    rhs_1 = swe_1d_rhs(h, hu,btm, ops,vgeo,fgeo,mapP,dt1, g, tol)

    htmp  = h  + dt1*rhs_1[1];
    hutmp = hu + dt1*rhs_1[2];
    # utmp = (htmp, hutmp)
    # htmp[findall(x->x<tol, htmp)] .= tol;
    hutmp[findall(x->x<tol, htmp)] .= 0;
    # @show L1, htmp dt1

    h_min, pos = findmin(htmp)
    if h_min <= 0
       # @show L1
       error("htmp_min<0 ", h_min, pos, "iteration ", i )
    end
    lambda = maximum(abs.(hutmp./htmp)+sqrt.(g.*htmp))
    dt2 = min(min(T,t_plot[pl_idx])-t, minimum(w)*J[1]/(ts_ft*lambda), dT);

    while dt2<dt1
        dt1 = dt2
        # rhsh1, rhshu1, L1 = convex_limiter(rhsh_ES1, rhshu_ES1, rhsh_ID1, rhshu_ID1, h, h, tol, dt1)
        htmp  = h  + dt1*rhs_1[1];
        hutmp = hu + dt1*rhs_1[2];
        # htmp[findall(x->x<tol, htmp)] .= tol;
        hutmp[findall(x->x<tol, htmp)] .= 0;
        lambda = maximum(abs.(hutmp./htmp)+sqrt.(g.*htmp))
        dt2 = min(min(T,t_plot[pl_idx])-t, minimum(w)*J[1]/(ts_ft*lambda), dT);
    end

    utmp = (htmp, hutmp, btm)
    rhs_2 = swe_1d_rhs(h, hu,btm, ops,vgeo,fgeo,mapP,dt1, g, tol)
    dt = min(dt1, dt2)

    h  .+= .5*dt*(rhs_1[1] + rhs_2[1])
    hu .+= .5*dt*(rhs_1[2] + rhs_2[2])
    # h[findall(x->x<tol, h)] .= tol
    hu[findall(x->x<tol, h)] .= 0
    # @show L2, h, dt
    h_min, pos = findmin(h)
    if h_min <=0
        # @show L2
        @show maximum(hu./h)
        error("h_min<0 ", h_min, pos, "iteration ", i )
    end
    if t>=T
            break
    end
    u = (h,hu, btm)
    t += dt
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
end #every 10
end
DT = DT[1:findmin(DT)[2]-1];
plot(Vp*x,Vp*(h+btm),ylims=(-.1,50))

h0 = 8
h_final = h0 .- B^2/(4*g)*cos(2*omega*t) .- B^2/(4*g) .- (B*x)/(2*a)*sqrt(8*h0/g)*cos(omega*t);
h_final = h_final - btm;
h_final[findall(x->x<tol, h_final)] .= tol;
plot!(Vp*x,Vp*(h_final+btm),ylims=(-.1,50), linestyle = :dot)
