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
K1D = 16
CFL = 0.25
T   = 1 # endtime

"Mesh related variables"
VX = LinRange(-1,1,K1D+1)
EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]),2,K1D))

"Construct matrices on reference elements"
r,w = NodesAndModes.gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/V
M = inv(V*V')
M = diagm(w.*J[1])
Q = M*Dr
E = zeros(2,size(r,1))
E[1,1] = J[1]; E[end,end] = J[1];
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

"initial conditions"
u = @. exp(-25*x^2)
u = x*0 .+1
# u[:,1:convert(Int,K1D/2)] .= 1e-10
# u[:,convert(Int,K1D/2)+1:K1D] .= 1.0

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples - will "
ops = (Q_skew,E,M_inv,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)

function rhs(u,ops,vgeo,fgeo,mapP)
    # unpack args
    Q_skew,E,M_inv,Vf = ops
    rxJ,J = vgeo
    nxJ = fgeo

    uM = Vf*u # can replace with nodal extraction
    uP = uM[mapP]
    Fs = ((uM + uP)/2).*nxJ
    du = uP - uM

    # ux = rxJ*(Q*u)
    tau = 1# upwind penalty parameter
    f1 = transpose(E)*(Fs - .5*tau*abs.(nxJ).*du)
    # rhsu = ux + .5*LIFT*(@. du*nxJ - tau*abs(nxJ)*du)
    rhsu = zeros(size(u))
    for e = 1:size(u,2)
        Fx = zeros(size(u,1), size(u,1))
        for i = 1:size(u,1)
            for j = 1:size(u,1)
                Fx[i,j] = ( u[i,e]+u[j,e] ) /2
            end
        end
        rhsu[:,e] = 2*sum(Q_skew.*Fx,dims = 2)
    end
    rhsu += f1

    return -M_inv*rhsu
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*u)

resu = zeros(size(x))
@gif for i = 1:Nsteps
      rhsu = rhs(u,ops,vgeo,fgeo,mapP)
      utmp = u + dt*rhsu
      u .+= .5*dt*(rhsu + rhs(utmp,ops,vgeo,fgeo,mapP))
    # for INTRK = 1:5
    #     rhsu = rhs(u,ops,vgeo,fgeo,mapP)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsu
    #     @. u   += rk4b[INTRK]*resu
    # end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*u,ylims=(-.1,1.1),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,u)
        # sleep(.0)
    end
end every 5

# plot(Vp*x,Vp*u,ylims=(-.1,1.1))
