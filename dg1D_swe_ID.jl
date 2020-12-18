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

global g = 1
"Approximation parameters"
N   = 3 # The order of approximation
K1D = 8
CFL = 1/4
T   = 0.5 # endtime

"Mesh related variables"
VX = LinRange(-1,1,K1D+1)
EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]),2,K1D))

"Construct matrices on reference elements"
r,w = NodesAndModes.gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/V
M = inv(V*V')

avg(a,b) = .5*(a+b)
function fS1D_LF(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL
    uR = huR./hR
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL*uL,huR*uR) + .5*avg(hL*hL,hR*hR)
    return fxS1,fxS2
end

function make_meshfree_ops(r,w)
        # p = 1
        EL = [vandermonde_1D(1,[-1])/vandermonde_1D(1,r[1:2]) zeros(1,N-1)]
        ER = [zeros(1,N-1) vandermonde_1D(1,[1])/vandermonde_1D(1,r[end-1:end])]
        E  = [EL;ER]

        # # using p=2 extrapolation
        # EL = [vandermonde_1D(2,[-1])/vandermonde_1D(2,r[1:3]) zeros(1,N-2)]
        # ER = [zeros(1,N-2) vandermonde_1D(2,[1])/vandermonde_1D(2,r[end-2:end])]
        # E  = [EL;ER]

        B = diagm([-1,1])

        S = diagm(1=>ones(N),-1=>ones(N))
        # S[1,3] = 1
        # S[end,end-2] = 1
        # S = one.(S)
        adj = sparse(triu(S)-triu(S)')
        # @show S
        # @show adj
        function build_weighted_graph_laplacian(adj,r,p)
                Np = length(r)
                L  = zeros(Np,Np)
                for i = 1:Np
                        for j = 1:Np
                                if adj[i,j] != 0
                                        L[i,j] += @. (.5*(r[i]+r[j]))^p
                                end
                        end
                        L[i,i] = -sum(L[i,:])
                end
                return L
        end

        # constant exactness
        L = build_weighted_graph_laplacian(adj,r,0)
        @show L
        b1 = zeros(N+1) - .5*sum(E'*B*E,dims=2)
        ψ1 = pinv(L)*b1

        ψx = pinv(L)*(w - .5*E'*B*E*r)

        function fillQ(adj,ψ,r,p)
                Np = length(ψ)
                Q = zeros(Np,Np)
                for i = 1:Np
                        for j = 1:Np
                                if adj[i,j] != 0
                                        Q[i,j] += (ψ[j]-ψ[i])*r[j]^p #(ψ[j]-ψ[i])*(.5*(r[i]+r[j]))^p
                                end
                        end
                end
                return Q
        end

        S1 = fillQ(adj,ψ1,r,0)
        Q = S1 + .5*E'*B*E # first order accuracy
        # S1 = fillQ(adj,ψx,r,0)
        # Q = S1 + .5*E'*B*E # first order accuracy
        return Q,E,B,ψ1
end

function build_rhs_matrix(applyRHS,Np,K,vargs...)
    u = zeros(Np,K)
    A = spzeros(Np*K,Np*K)
    for i in eachindex(u)
        u[i] = one(eltype(u))
        r_i = applyRHS(u,vargs...)
        A[:,i] = droptol!(sparse(r_i[:]),1e-12)
        u[i] = zero(eltype(u))
    end
    return A
end

Q,E,B,ψ = make_meshfree_ops(r, w)
Q_skew = .5*(Q-transpose(Q))

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
M_inv = spdiagm(0 => 1 ./vec(diagm(w)*J))

"initial conditions"
u = @. exp(-25*x^2)

h = @. exp(-25*x^2)+2
# h = h*0 .+2
h = [0.43012     0.00283327   4.32944e-6  0.00587629   0.461943  0.989807  0.99953   0.999644;
 0.0737789   0.000115821  1.01984e-8  0.000110116  0.937357  0.99886   1.00033   1.0128;
 1.80755e-8  1.06873e-8   8.2332e-5   0.0724844    1.01279   1.00033   0.99886   0.937346;
 0.00550536  5.01346e-6   0.00173385  0.431363     0.999649  0.99953   0.989807  0.456408]
hu= [0.554957    0.00709513  -1.41038e-6   -0.00332459  -0.277515   -0.0101393     0.000147021   0.0195437;
  -0.00350728  9.98422e-5   0.0          -7.46314e-5  -0.0800892  -0.00114606   -0.000350967  -0.0246873;
   0.0         0.0         -4.36845e-5   -0.0474182    0.0246741   0.000350967   0.00114606    0.0801153;
   0.109488    2.80551e-6  -0.000938431  -0.279944    -0.0195245  -0.00014702    0.0101397     0.278559]
# h[:,1:convert(Int,K1D/2)] .= 1e-10
# h[:,convert(Int,K1D/2)+1:K1D] .= 1.0
# hu = h*0;

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dT = T/Nsteps

"pack arguments into tuples - will "
ops = (Q_skew,E,B,M_inv, Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)
nodemaps = (mapM, mapP)

function rhsx(u,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Q,E,B,M_inv, Vf = ops
    rxJ,J = vgeo
    nxJ = fgeo
    (mapM,mapP) = nodemaps

    uM = Vf*u # can replace with nodal extraction
    dudx = rxJ.*(Q*u)
    rhsu = dudx + transpose(E)*(.5*nxJ.*uM[mapP])
    return rhsu
end

Qx = build_rhs_matrix(rhsx,size(u,1),size(u,2),ops,vgeo,fgeo,nodemaps)


function rhs(h,hu,M_inv, Qx)
    # unpack args
    # Q,E,B,M_inv,Vf  = ops
    # rxJ,J = vgeo
    # nxJ = fgeo
    #
    # uM = Vf*u # can replace with nodal extraction
    # du = uM[mapP]-uM

    rows = rowvals(Qx)
    vals = nonzeros(Qx)

    rhsh = zero.(h)
    rhshu = zero.(hu)
    for j = 1:size(Qx,2) # loop over columns
        hj = h[j]
        huj = hu[j]
        for index in nzrange(Qx,j) # loop over rows
            i = rows[index]
            Qxij = vals[index]
            lambdaj = abs(huj./hj)+sqrt.(g.*hj)
            lambdai = abs(hu[i]./h[i])+sqrt.(g.*h[i])
            dij = abs.(Qxij)*max(lambdaj, lambdai)
            @show dij
            Fx1, Fx2 = fS1D_LF((hj, huj), (hj, huj))
            rhsh[i]  += Qxij*Fx1  - dij * (hj - h[i])
            rhshu[i] += Qxij*Fx2 - dij * (huj- hu[i])
        end
    end
    # @show size(M_inv), size(rhsu)
    return -M_inv*rhsh, -M_inv*rhshu

    # rhsu = Q*u
    # for e = 1:K1D
    #     ux, uy  = meshgrid(u[:,e] );
    #     # rhsu[:,e] = 2*(sum(QNx.*FxV,dims = 2) )#+ sum(QNy.*FxV,dims = 2));
    #     rhsu[:,e] -= transpose(sum(abs.(Q)*(uy-ux), dims = 1))
    #
    # end
    # rhsu = zero.(u)
    # for e = 1:K1D # loop over columns
    #     for j = 1:length(u[:,e])
    #             uj = u[j]
    #             for i = 1:length(u[:,e]) # loop over rows
    #                 Qij = Q[i,j]
    #                 dij = abs.(Qij)
    #                 rhsu[i,e] += Qij*uj - dij * (uj-u[i])
    #             end
    #     end
    # end
    # for e = 1:K1D # loop over columns
    #     for i = 1:length(u[:,e])
    #             for j = 1:length(u[:,e]) # loop over rows
    #                 Qij = Q[i,j]
    #                 dij = abs.(Qij)
    #                 rhsu[i,e] += Qij*u[j,e] - dij * (u[j,e]-u[i,e])
    #             end
    #     end
    # end
    # rhsu = Q*u -
    # for e = 1:K1D # loop over columns
    #     for i = 1:length(u[:,e])
    #             ui = u[i]
    #             for j = 1:length(u[:,e]) # loop over rows
    #                 Qij = Q[i,j]
    #                 dij = abs.(Qij)
    #                 rhsu[i,e] -= dij * (u[j,e]-u[i,e])
    #             end
    #     end
    # end
    # return -M_inv*rhsu
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*h)

resu = zeros(size(x))
h = vec(h)
hu = vec(hu)
t = 0
@gif for i = 1:100000
    @show i, t
    global h, hu, t
    h = vec(h)
    hu = vec(hu)
    # for INTRK = 1:1
    #     # rhsu = rhs(u,ops,vgeo,fgeo,mapP)
    #     rhsu = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsu
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    lambda = maximum(abs.(hu./h)+sqrt.(g.*h))
    dt1 = min(T-t, minimum(w)/(2*lambda), dT);
    rhsh1, rhshu1  = rhs(h,hu,M_inv,Qx)
    htmp  = h  + dt1*rhsh1
    hutmp = hu + dt1*rhshu1
    lambda = maximum(abs.(hutmp./htmp)+sqrt.(g.*htmp))
    dt2 = min(T-t, minimum(w)/(2*lambda), dT);
    while dt2<dt1
        dt1 = dt2
        htmp  = h  + dt1*rhsh1
        hutmp = hu + dt1*rhshu1
        lambda = maximum(abs.(hutmp./htmp)+sqrt.(g.*htmp))
        dt2 = min(T-t, minimum(w)/(2*lambda), dT);
    end
    dt = min(dt1, dt2)
    rhsh2 , rhshu2 = rhs(htmp, hutmp, M_inv,Qx)
    h  .+= .5*dt*(rhsh1 + rhsh2)
    hu .+= .5*dt*(rhshu1 + rhshu2)

    t +=dt
    if t>=T
            break
    end
    @show L1, L2
    i +=1
    if i%10==0
        h = reshape(h,size(x,1),size(x,2))
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*h,ylims=(-.1,2),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,h)
        # sleep(.0)
    end
end every 5

# plot(Vp*x,Vp*u,ylims=(-.1,1.1))
