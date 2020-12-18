#this version roll back to ID if the results negative water height

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
K1D = 16
CFL = 1/4
T   = 1 # endtime

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

"Mesh related variables"
VX = LinRange(-1,1,K1D+1)
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
# M_inv = spdiagm(0 => 1 ./vec(diagm(w)*J))

"initial conditions"
h = @. exp(-25*x^2)+2
# h = h*0 .+2
h[:,1:convert(Int,K1D/2)] .= 1e-10
h[:,convert(Int,K1D/2)+1:K1D] .= 1.0
hu = h*0;

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dT = T/Nsteps

"pack arguments into tuples - will "
u    = (h, hu)
ops  = (Q_IDskew,Q_ESskew,E,M_inv, Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)
nodemaps = (mapM, mapP)

function rhs(U,ops,vgeo,fgeo,mapP,dt)
        h, hu = U
        Q_ID,Q_ES,E,M_inv,Vf = ops
        rxJ,J = vgeo
        nxJ = fgeo

        u = hu./h
        hM = Vf*h # can replace with nodal extraction
        hP = hM[mapP]
        dh = hP-hM

        huM = Vf*hu
        huP = huM[mapP]
        dhu = huP - huM

        uM = Vf*u
        uP = uM[mapP]

        UL = (hM, huM)
        UR = (hP, huP)
        fxS1,fxS2 = fS1D(UL,UR)
        # lambda = abs.(u.*nxJ)+sqrt.(g.*h)
        lambda = abs.(u)+sqrt.(g.*h)
        lambdaM = Vf*lambda
        lambdaP = lambdaM[mapP]
        lambdaF = max.(lambdaM, lambdaP)

        #ES part
        Fs1 = fxS1.*nxJ
        Fs2 = fxS2.*nxJ
        tau = 1
        f1 = transpose(E)*(Fs1 - .5*tau*abs.(lambdaF).*dh)
        f2 = transpose(E)*(Fs2 - .5*tau*abs.(lambdaF).*dhu)
        rhsh_ES = zeros(size(h))
        rhshu_ES = zeros(size(h))
        for e = 1:size(h,2)
            Fv1 = zeros(size(h,1), size(h,1))
            Fv2 = zeros(size(h,1), size(h,1))
            for i = 1:size(h,1)
                for j = 1:size(h,1)
                    Ui = (h[i,e], hu[i,e])
                    Uj = (h[j,e], hu[j,e])
                    Fv1[i,j], Fv2[i,j] = fS1D(Ui,Uj)
                end
            end
            rhsh_ES[:,e]  = 2*sum(Q_ES.*Fv1,dims = 2)
            rhshu_ES[:,e] = 2*sum(Q_ES.*Fv2,dims = 2)
        end
        rhsh_ES  += f1
        rhshu_ES += f2

        # ID part
        c = [transpose(max.(lambda[1,:],lambda[2,:] )); transpose(max.(lambda[end,:],lambda[end-1,:] ))]
        # @show lambda c
        cij = 1/2
        rhsh_ID = rxJ.*(Q_ID*hu) + transpose(E)*(0.5*nxJ.*huP)
        rhsh_ID -= transpose(E)*(cij*max.(lambdaP,lambdaM) .* ( hP - hM )  )
        rhsh_ID -= transpose(E)*(cij*c.* ([transpose(h[2,:]);transpose(h[end-1,:])] - hM) )

        rhshu_ID = rxJ.*(Q_ID*(hu.*u+1/2*g*h.*h)) + transpose(E)*(0.5*nxJ.*huP.*uP+1/2*g*hP.*hP)
        rhshu_ID -= transpose(E)*(cij*max.(lambdaP,lambdaM).* ( huP - huM ) )
        rhshu_ID -= transpose(E)*(cij*c.* ([transpose(hu[2,:]);transpose(hu[end-1,:])] - huM) )
        # for e = 1:K1D # loop over columns
        for i = 2:length(h[:,1])-1
            ci_1 = max.(lambda[i,:], lambda[i-1,:])
            ci_2 = max.(lambda[i,:], lambda[i+1,:])

            rhsh_ID[i,:]  -= cij * ( ci_1.*(h[i-1,:]-h[i,:])  + ci_2.*(h[i+1,:]-h[i,:]) )
            rhshu_ID[i,:] -= cij * ( ci_1.*(hu[i-1,:]-hu[i,:]) + ci_2.*(hu[i+1,:]-hu[i,:]) )
        end
        # end

        return -M_inv*rhsh_ES, -M_inv*rhshu_ES, -M_inv*rhsh_ID, -M_inv*rhshu_ID, maximum(lambda)
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*h)
resu = zeros(size(x))
tol = 1e-8
t = 0
@gif for i = 1:100000
    @show i, t
    global h, hu, u, t
    # for INTRK = 1:5
    #     rhsh = rhs(u,ops,vgeo,fgeo,mapP)
    #     # rhsh = rhs(u,M_inv,Qx)
    #     @. resu = rk4a[INTRK]*resu + dt*rhsh
    #     @. u   += rk4b[INTRK]*resu
    # end
    # Heun's method - this is an example of a 2nd order SSP RK method
    rhsh_ES1, rhshu_ES1, rhsh_ID1, rhshu_ID1, lambda = rhs(u,ops,vgeo,fgeo,mapP,0)
    # dt1 = min(T-t, minimum(w)/(2*lambda), dT);
    rhsh_Diff1 = rhsh_ES1 - rhsh_ID1; rhshu_Diff1 = rhshu_ES1 - rhshu_ID1;
    rhsh1  = zeros(size(h))
    rhshu1 = zeros(size(h))
    L1 = ones(1,size(h,2))
    for e = 1:size(h,2)
        for k = 1:size(h,1)
            l_k = 1
            if rhsh_Diff1[k,e] < 0
                l_k = -(h[k,e] + dt*rhsh_ID1[k,e]-tol) / (dt*(rhsh_Diff1[k,e]))
                l_k = min(l_k, 1)
            end
            l_k = max(l_k,0)
            L1[e] = min(L1[e], l_k);
            if h[k,e] < tol || norm(rhsh_Diff1)<tol
                L1[e] = 0
            end
        end
        rhsh1[:,e]  = rhsh_ID1[:,e]  + rhsh_Diff1[:,e] *L1[e]
        rhshu1[:,e] = rhshu_ID1[:,e] + rhshu_Diff1[:,e]*L1[e]
    end
    # if dt<dT
    #     rhsh1 = rhsh_ID1;
    #     rhshu1 = rhshu_ID1;
    # end
    htmp  = h  + dt*rhsh1
    hutmp = hu + dt*rhshu1
    utmp = (htmp, hutmp)
    @show L1, htmp dt

    h_min, pos = findmin(htmp)
    if h_min < 0
       error("htmp_min<0 ", h_min, pos, "iteration ", i )
    end
    rhsh_ES2, rhshu_ES2, rhsh_ID2, rhshu_ID2, lambda = rhs(utmp,ops,vgeo,fgeo,mapP,0)
    # dt2 = min(T-t, minimum(w)/(2*lambda), dT);
    # while dt2<dt1
    #     dt1 = dt2
    #     htmp  = h  + dt1*rhsh1
    #     hutmp = hu + dt1*rhshu1
    #     utmp = (htmp, hutmp)
    #     rhsh_ES2, rhshu_ES2, rhsh_ID2, rhshu_ID2, lambda = rhs(utmp,ops,vgeo,fgeo,mapP,0)
    #     dt2 = min(T-t, minimum(w)/(2*lambda), dT);
    # end
    rhsh_Diff2 = rhsh_ES2 - rhsh_ID2; rhshu_Diff2 = rhshu_ES2 - rhshu_ID2;
    # dt = min(dt1, dt2)
    rhsh2 = zeros(size(h))
    rhshu2 = zeros(size(h))
    L2 = ones(1,size(h,2))
    for e = 1:size(h,2)
        for k = 1:size(h,1)
            l_k = 1
            if rhsh_Diff2[k,e] < 0
                # l_k = -( 2*h[k,e] + dt*(rhsh_ID2[k,e]+ rhsh1[k,e])-tol ) / (dt*rhsh_Diff2[k,e] )
                l_k = -(htmp[k,e] + dt*rhsh_ID2[k,e]-tol) / (dt*(rhsh_Diff2[k,e]))
                l_k = min(l_k, 1);
            end
            l_k = max(l_k,0)
            L2[e] = min(L2[e], l_k);
            if h[k,e] < tol || htmp[k,e] < tol || norm(rhsh_Diff2)<tol
                L2[e] = 0
            end
        end
        rhsh2[:,e]  = rhsh_ID2[:,e]  + rhsh_Diff2[:,e] *L2[e]
        rhshu2[:,e] = rhshu_ID2[:,e] + rhshu_Diff2[:,e]*L2[e]
    end
    # dt = min(T-t, minimum(w)/(2*lambda), dT);
    # if dt< dT
    #     rhsh2 = rhsh_ID2;
    #     rhshu2 = rhshu_ID2;
    # end
    h  .+= .5*dt*(rhsh1  + rhsh2)
    hu .+= .5*dt*(rhshu1 + rhshu2)
    @show L2, h, dt
    h_min, pos = findmin(h)
    if h_min < 0
        h  .-= .5*dt*(rhsh1  + rhsh2)
        hu .-= .5*dt*(rhshu1 + rhshu2)
        htmp  = h  + dt*rhsh_ID1
        hutmp = hu + dt*rhshu_ID1
        utmp = (htmp, hutmp)

        rhsh_ES2, rhshu_ES2, rhsh_ID2, rhshu_ID2, lambda = rhs(utmp,ops,vgeo,fgeo,mapP,0)
        h  .+= .5*dt*(rhsh_ID1  + rhsh_ID2)
        hu .+= .5*dt*(rhshu_ID1 + rhshu_ID2)
        # @show maximum(hu./h)
        # error("h_min<0 ", h_min, pos, "iteration ", i )
    end
    u = (h,hu)
    t += dt
    if t>=T
            break
    end
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
end #every 10

# plot(Vp*x,Vp*u,ylims=(-.1,1.1))
