using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using SetupDG

"Approximation parameters"
N   = 3 # The order of approximation

# initialize ref element and mesh
rd1D = init_reference_interval(N)
rd2D = init_reference_quad(N,gauss_lobatto_quad(0,0,N))

@unpack Vq = rd1D
@unpack Vq = rd2D

function VU(h,hu)
    u = @. hu/h
    v1 = @. h - .5*u^2
    v2 = @. u
    return v1,v2
end

function VU(h,hu,hv)
    u = @. hu/h
    v = @. hv/h
    v1 = @. h - .5*(u^2+v^2)
    v2 = @. u
    v3 = @. v
    return v1,v2,v3
end

# assume g = 1
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

function init1D(N,rd1D::RefElemData)
    @unpack M,Dr,Vf,nrJ = rd1D
    Q = M*Dr
    S = Q-Q'
    VfTW = transpose(Vf) # W = I in 1D

    function rhs1D(U,flux)
        h,hu = U
        f1,f2 = flux
        Uf = (x->Vf*x).(U)
        UP = (x->x[[2;1]]).(Uf)
        hL,hR   = meshgrid(h)
        huL,huR = meshgrid(hu)
        F1,F2 = fS1D((hL,huL),(hR,huR))
        rhs1 = sum(S.*F1,dims=2) + VfTW*(nrJ.*f1)
        rhs2 = sum(S.*F2,dims=2) + VfTW*(nrJ.*f2)
        return rhs1,rhs2
    end
    return rhs1D
end

function init2D(N,rd2D::RefElemData)
    @unpack M,Dr,Ds,nrJ,nsJ,Vf,r,s,rf,sf,wf = rd2D
    Sr,Ss = (A->(A-A')).((M*Dr,M*Ds))
    VfTW  = transpose(Vf)*diagm(wf)
    mapM  = reshape(1:(N+1)*4,N+1,4)
    mapP  = vec(mapM[:,[3,4,1,2]])

    function rhs2D(U,fx,fy)
        h,hu = U
        Uf = (x->Vf*x).(U)
        hL,hR   = meshgrid(h)
        huL,huR = meshgrid(hu)
        hvL,hvR = meshgrid(hv)
        Fx,Fy = fS2D((hL,huL,hvL),(hR,huR,hvR))
        rhs1 = sum(Sr.*Fx[1]+Ss.*Fy[1],dims=2) + VfTW*(nrJ.*vec(fx[1])+nsJ.*vec(fy[1]))
        rhs2 = sum(Sr.*Fx[2]+Ss.*Fy[2],dims=2) + VfTW*(nrJ.*vec(fx[2])+nsJ.*vec(fy[2]))
        rhs3 = sum(Sr.*Fx[3]+Ss.*Fy[3],dims=2) + VfTW*(nrJ.*vec(fx[3])+nsJ.*vec(fy[3]))
        return rhs1,rhs2,rhs3
    end
end

# initialize rhs with operators inherited inside init1D/2D
rhs1D = init1D(N,rd1D)
rhs2D = init2D(N,rd2D)

# ========= initialize variables to some arbitrary functions ==========

@unpack r = rd1D
h1D  = @. 2 + exp(-r^2)
hu1D = @. r + exp(-r^2)
# h1D  = 4 .+ r*0
# hu1D = r*0

@unpack r,s,rf,sf,Vf,wf = rd2D
wf = wf[1:N+1] # get a single face
h  = @. 2 .+ .5*exp(-(r^2+s^2))
hu = @. r + .5*exp(-(r^2+s^2))
hv = .01*randn(length(r))
# h  = @. 3 .+ r*0
# hu = @. r*0
# hv = @. r*0

# =============== coupled version ================

# coupling 1D->2D
e = ones(N+1)
hf,huf,hvf = (x->reshape(Vf*x,N+1,4)).((h,hu,hv))

hP,huP,hvP = copy.((hf,huf,hvf))

# impose wall BCs in y in 2D region
hP[:,[1,3]]  = hf[:,[1,3]]
huP[:,[1,3]] = huf[:,[1,3]]
hvP[:,[1,3]] = -hvf[:,[1,3]]

# compute exterior values for 1D/2D coupling
hP[:,4]  = e*h1D[N+1]
huP[:,4] = e*hu1D[N+1]
hP[:,2]  = e*h1D[1]
huP[:,2] = e*hu1D[1]
hvP[:,[2,4]] = -0*hvf[:,[2,4]] # impose zero flow in the orthogonal direction
fx,fy = fS2D((hf,huf,hvf),(hP,huP,hvP))
@show fx
@show fy
bmult = (x,y)->x.*y # broadcasted mult

vu_1D = VU(h1D,hu1D)
vu_2D = VU(h,hu,hv)

# 1D / 2D rhs
f1L = wf'*fx[1][:,2]
f1R = wf'*fx[1][:,4]
f2L = wf'*fx[2][:,2]
f2R = wf'*fx[2][:,4]
f1 = [f1L;f1R]/sum(wf)
f2 = [f2L;f2R]/sum(wf)
@show f1
@show f2
rhs_1D = rhs1D((h1D,hu1D),(f1,f2))
rhs_2D = rhs2D((h,hu,hv),fx,fy)
rhstest = sum(wf)*sum(sum(bmult.(vu_1D,rhs_1D))) + sum(sum(bmult.(vu_2D,rhs_2D)))

@show rhstest
