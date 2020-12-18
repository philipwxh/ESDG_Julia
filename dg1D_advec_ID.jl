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

"Approximation parameters"
N   = 3 # The order of approximation
K1D = 4
CFL = 0.25
T   = 0.5 # endtime

"Mesh related variables"
VX = LinRange(-1,1,K1D+1)
EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]),2,K1D))

"Construct matrices on reference elements"
r,w = NodesAndModes.gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/V
M = inv(V*V')

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
u0 = @. exp(-25*x^2)

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

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


function rhs(u,M_inv, Qx)
    # unpack args
    # Q,E,B,M_inv,Vf  = ops
    # rxJ,J = vgeo
    # nxJ = fgeo
    #
    # uM = Vf*u # can replace with nodal extraction
    # du = uM[mapP]-uM

    rows = rowvals(Qx)
    vals = nonzeros(Qx)

    rhsu = zero.(u)
    for j = 1:size(Qx,2) # loop over columns
        uj = u[j]
        for index in nzrange(Qx,j) # loop over rows
            i = rows[index]
            Qxij = vals[index]
            dij = abs.(Qxij)
            rhsu[i] += Qxij*uj - dij * (uj-u[i])
        end
    end
    # @show size(M_inv), size(rhsu)
    return -M_inv*rhsu

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
plt = plot(Vp*x,Vp*u)

resu = zeros(size(x))
u = vec(u)
@gif for i = 1:Nsteps
    global u
    u = vec(u)
    # for INTRK = 1:1
    #     # rhsu = rhs(u,ops,vgeo,fgeo,mapP)
    #     rhsu = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsu
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    rhsu = rhs(u,M_inv,Qx)
    utmp = u + dt*rhsu
    u .+= .5*dt*(rhsu + rhs(utmp,M_inv,Qx))

    if i%5==0 || i==Nsteps
        u = reshape(u,size(x,1),size(x,2))
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*u,ylims=(-.1,1.1),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,u)
        # sleep(.0)
    end
end every 5

# plot(Vp*x,Vp*u,ylims=(-.1,1.1))
