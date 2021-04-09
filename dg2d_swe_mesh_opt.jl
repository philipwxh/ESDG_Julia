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

function init_reference_tri_sbp_GQ(N, qnode_choice)
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
    if qnode_choice == "GQ"
        r1D, w1D = gauss_quad(0,0,N)
    elseif qnode_choice == "GL" || qnode_choice == "tri_diage"
        r1D, w1D = gauss_lobatto_quad(0,0,N+1)
    end
    Nfp = length(r1D) # number of points per face
    e = ones(Nfp) # vector of all ones
    z = zeros(Nfp) # vector of all zeros
    rf = [r1D; -r1D; -e];
    sf = [-e; r1D; -r1D];
    wf = vec(repeat(w1D,3,1));
    nrJ = [z; e; -e]
    nsJ = [-e; e; z]
    @pack! rd = rf,sf,wf,nrJ,nsJ

    if qnode_choice == "GQ"
        rq,sq,wq = GQ_SBP[N];
    elseif qnode_choice == "GL"
        rq,sq,wq = GL_SBP[N];
    elseif qnode_choice == "tri_diage"
        rq,sq,wq = Tri_diage[N];
    end
    # rq,sq,wq = GQ_SBP[N]
    # rq,sq,wq = GL_SBP[N]
    # rq,sq,wq = Tri_diage[N]
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

function ESDG_bottom(QNr_sbp, QNs_sbp,btm,vgeo,g)
    rxJ,sxJ,ryJ,syJ,J = vgeo
    gQNxb = zeros(size(btm))
    gQNyb = zeros(size(btm))
    for e = 1:size(h,2)
        QNx = 1/2*( diagm(rxJ[:,e])*QNr_sbp + QNr_sbp*diagm(rxJ[:,e]) + diagm(sxJ[:,e])*QNs_sbp + QNs_sbp*diagm(sxJ[:,e]) );
        QNy = 1/2*( diagm(ryJ[:,e])*QNr_sbp + QNr_sbp*diagm(ryJ[:,e]) + diagm(syJ[:,e])*QNs_sbp + QNs_sbp*diagm(syJ[:,e]) );
        gQNxb[:,e] += g*QNx*btm[:,e];
        gQNxb[:,e] += g*QNy*btm[:,e];
    end
    return gQNxb, gQNyb
end
