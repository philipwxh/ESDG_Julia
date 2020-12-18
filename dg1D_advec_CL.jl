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
K1D = 32
CFL = 0.5
T   = 2 # endtime

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

Q_ID,E,B,ψ = make_meshfree_ops(r, w)
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
# M_inv = spdiagm(0 => 1 ./vec(diagm(w)*J))

"initial conditions"
u = @. exp(-25*x^2)
u0 = @. exp(-25*x^2)
# u = x*0 .+1
u[:,1:convert(Int,K1D/2)] .= 1e-10
u[:,convert(Int,K1D/2)+1:K1D] .= 1.0

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples - will "
ops = (Q_IDskew,Q_ESskew,E,M_inv, Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)
nodemaps = (mapM, mapP)

function rhs(u,ops,vgeo,fgeo,mapP,dt)
        Q_ID,Q_ES,E,M_inv,Vf = ops
        rxJ,J = vgeo
        nxJ = fgeo

        uM = Vf*u # can replace with nodal extraction
        uP = uM[mapP]
        du = uP - uM

        #ES part
        Fs = ((uM + uP)/2).*nxJ
        tau = 1
        f1 = transpose(E)*(Fs - .5*tau*abs.(nxJ).*du)
        rhsu_ES = zeros(size(u))
        for e = 1:size(u,2)
            Fx = zeros(size(u,1), size(u,1))
            for i = 1:size(u,1)
                for j = 1:size(u,1)
                    Fx[i,j] = ( u[i,e]+u[j,e] ) /2
                end
            end
            rhsu_ES[:,e] = 2*sum(Q_ES.*Fx,dims = 2)
        end
        rhsu_ES += f1

        # ID part
        rhsu_ID = rxJ.*Q_ID*u + transpose(E)*(.5*nxJ.*uM[mapP])
        rhsu_ID -= transpose(E)*(1/2*( uM[mapP]-uM + [transpose(u[2,:]);transpose(u[end-1,:])] - uM))
        # for e = 1:K1D # loop over columns
        for i = 2:length(u[:,1])-1
            rhsu_ID[i,:] -= 1/2 * ( (u[i-1,:]-u[i,:]) + (u[i+1,:]-u[i,:]) )
        end

        return -M_inv*rhsu_ES, -M_inv*rhsu_ID
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*u)
resu = zeros(size(x))
tol = 1e-15
@gif for i = 1:Nsteps
    global u
    # for INTRK = 1:5
    #     rhsu = rhs(u,ops,vgeo,fgeo,mapP)
    #     # rhsu = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsu
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    rhsu_ES, rhsu_ID = rhs(u,ops,vgeo,fgeo,mapP,dt)
    rhsu_Diff = rhsu_ES - rhsu_ID;
    rhsu1 = zeros(size(u))
    L = ones(1,size(u,2))
    for e = 1:size(u,2)
        for k = 1:size(u,1)
            if rhsu_Diff[k,e] > 0
                l_temp = 1;
            # elseif abs(rhsu_Diff[k,e] ) < tol
            #     l_temp = 0;
            else
                l_temp = -(u[k,e] + dt*rhsu_ID[k,e]-tol) / (dt*(rhsu_Diff[k,e]))
            end
            l_temp = max(l_temp,0)
            L[e] = min(L[e], l_temp);
        end
        if L[e] < 0 || L[e] > 1
            error("L[",e,"] = ", L[e] )
        end
        # L[e]  = 1
        rhsu1[:,e] = rhsu_ID[:,e] + rhsu_Diff[:,e]*L[e]
    end
    # @show L
    utmp = u + dt*rhsu1
    rhsu_ES, rhsu_ID = rhs(utmp,ops,vgeo,fgeo,mapP,dt)
    rhsu_Diff = rhsu_ES - rhsu_ID;
    rhsu2 = zeros(size(u))
    L = ones(1,size(u,2))
    for e = 1:size(u,2)
        for k = 1:size(u,1)
            if rhsu_Diff[k,e] > 0
                l_temp = 1
            else
                l_temp = -( 2*u[k,e] + dt*(rhsu_ID[k,e]+ rhsu1[k,e])-tol ) / (dt*rhsu_Diff[k,e] )
            end
            l_temp = max(l_temp,0)
            L[e] = min(L[e], l_temp);
        end
        if L[e] < 0 || L[e] > 1
            error("L[",e,"] = ", L[e] )
        end
        # L[e]  = 1
        rhsu2[:,e] = rhsu_ID[:,e] + rhsu_Diff[:,e]*L[e]
    end
    # @show L
    u .+= .5*dt*(rhsu1 + rhsu2)
    u_min, pos = findmin(u)
    if u_min < 0
       error("u_min<0  ", u_min, pos )
    end
    if i%10==0 || i==Nsteps
        # u = reshape(u,size(x,1),size(x,2))
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*u,ylims=(-.1,1.1),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,u)
        # sleep(.0)
    end
end every 10

# plot(Vp*x,Vp*u,ylims=(-.1,1.1))
