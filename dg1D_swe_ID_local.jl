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
function fS1D(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL
    uR = huR./hR
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*hL*hR
    return fxS1,fxS2
end

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

Q,E,B,ψ = make_meshfree_ops(r, w)
E = zeros(size(E)); E[1,1] = 1; E[end,end]=1;
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
# M_inv = spdiagm(0 => 1 ./vec(diagm(w)*J))

"initial conditions"
h = @. exp(-25*x^2)+2
# h = h*0 .+2
h[:,1:convert(Int,K1D/2)] .= 1e-10
h[:,convert(Int,K1D/2)+1:K1D] .= 1.0
# h = copy(h0)
hu = h*0;

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples - will "
# u = (h, hu)
ops = (Q_skew,E,B,M_inv, Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)
nodemaps = (mapM, mapP)

function rhs(h,hu,ops,vgeo,fgeo,mapP)
    # unpack args
    # h, hu = U
    Q,E,B,M_inv,Vf  = ops
    rxJ,J = vgeo
    nxJ = fgeo

    u = hu./h
    hM = Vf*h # can replace with nodal extraction
    hP = hM[mapP]
    # dh = hP-hM

    huM = Vf*hu
    huP = huM[mapP]

    UL = (hM, huM)
    UR = (hP, huP)
    fxS1,fxS2 = fS1D_LF(UR,UR)
    # lambda = abs.(u.*nxJ)+sqrt.(g.*h)
    lambda = abs.(u)+sqrt.(g.*h)

    # lambda1 = (u.*nxJ)+sqrt.(g.*h)
    # lambda2 = (u.*nxJ)-sqrt.(g.*h)
    # lambda  = max.(abs.(lambda1), abs.(lambda2))
    lambdaM = Vf*lambda
    lambdaP = lambdaM[mapP]
    c = [transpose(max.(lambda[1,:],lambda[2,:] )); transpose(max.(lambda[end,:],lambda[end-1,:] ))]
    # @show lambda c
    cij = 1/2
    rhsh = rxJ.*(Q*hu) + transpose(E)*(0.5*nxJ.*fxS1)
    rhsh -= transpose(E)*(cij*max.(lambdaP,lambdaM) .* ( hP - hM )  )
    rhsh -= transpose(E)*(cij*c.* ([transpose(h[2,:]);transpose(h[end-1,:])] - hM) )

    rhshu = rxJ.*(Q*(hu.*u+1/2*g*h.*h)) + transpose(E)*(0.5*nxJ.*fxS2)
    rhshu -= transpose(E)*(cij*max.(lambdaP,lambdaM).* ( huP -  huM) )
    rhshu -= transpose(E)*(cij*c.* ([transpose(hu[2,:]);transpose(hu[end-1,:])] - huM) )
    # for e = 1:K1D # loop over columns
        for i = 2:length(h[:,1])-1
            ci_1 = max.(lambda[i,:], lambda[i-1,:])
            ci_2 = max.(lambda[i,:], lambda[i+1,:])

            rhsh[i,:]  -= cij * ( ci_1.*(h[i-1,:]-h[i,:])  + ci_2.*(h[i+1,:]-h[i,:]) )
            rhshu[i,:] -= cij * ( ci_1.*(hu[i-1,:]-hu[i,:]) + ci_2.*(hu[i+1,:]-hu[i,:]) )
        end
    # end
    return -M_inv*rhsh, -M_inv*rhshu
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*h)

# resh = zeros(size(x))
# reshu = zeros(size(x))
t = 0
@gif for i = 1:100000
    @show i, t
    global h, hu, t
    # for INTRK = 1:1
    #     # rhsu = rhs(u,ops,vgeo,fgeo,mapP)
    #     rhsu = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsu
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    lambda = maximum(abs.(hu./h)+sqrt.(g.*h))
    dt1 = min(T-t, minimum(w)/(2*lambda), dT);
    rhsh1, rhshu1  = rhs(h,hu,ops,vgeo,fgeo,mapP)
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
    rhsh2 , rhshu2 = rhs(htmp,hutmp,ops,vgeo,fgeo,mapP)
    h  .+= .5*dt*(rhsh1 + rhsh2)
    hu .+= .5*dt*(rhshu1 + rhshu2)

    t +=dt
    if t>=T
            break
    end
    @show L1, L2
    i +=1
    if i%10==0 || i==Nsteps
        # u = reshape(u,size(x,1),size(x,2))
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*h,ylims=(-.1,4),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,h)
        # sleep(.0)
    end
end every 5

# plot(Vp*x,Vp*h,xlims=(-1,1), ylims=(-.1,4.1))
