avg(a,b) = .5*(a+b)
function fS1D(UL,UR,g)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL
    uR = huR./hR
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*hL*hR
    return fxS1,fxS2
end

function fS1D_LF(UL,UR,g)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL
    uR = huR./hR
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL*uL,huR*uR) + .5*g*avg(hL*hL,hR*hR)
    return fxS1,fxS2
end

function convex_limiter(rhsh_ES, rhshu_ES, rhsh_ID, rhshu_ID, hh, htmp, tol, dt)
    rhsh_Diff = rhsh_ES - rhsh_ID; rhshu_Diff = rhshu_ES - rhshu_ID;
    rhsh = zeros(size(h)); rhshu = zeros(size(h));
    L = ones(1,size(h,2));
    for e = 1:size(h,2)
        for k = 1:size(h,1)
            l_k = 1
            if rhsh_Diff[k,e] < 0
                l_k = -(hh[k,e] + dt*rhsh_ID[k,e]-tol) / (dt*(rhsh_Diff[k,e]))
                l_k = min(l_k, 1)
            end
            l_k = max(l_k,0)
            l_k = min(1, l_k)
            if hh[k,e] <= tol || htmp[k,e] <= tol || norm(rhsh_Diff)<=tol
                L[e] = 0
                if e == 1
                    L[e+1] = 0
                    L[end] = 0
                elseif  e == K1D
                    L[e-1] = 0
                    L[1] = 0
                else
                    L[e+1] = 0
                    L[e-1] = 0
                end
            end
            l_k = min(L[e], l_k);
            rhsh[k,e]  = rhsh_ID[k,e]  + rhsh_Diff[k,e] *l_k
            rhshu[k,e] = rhshu_ID[k,e] + rhshu_Diff[k,e]*l_k
        end

    end
    return rhsh, rhshu, L
end

function swe_1d_esdg_surface(UL, UR, dU, E, nxJ, c, g)::Tuple{Array{Float64,2},Array{Float64,2}}
    (fxS1,fxS2) = fS1D(UL,UR,g)
    (hf, huf) = UL; (hP, huP) = UR; (dh, dhu) = dU;
    Fs1 = fxS1.*nxJ; Fs2 = fxS2.*nxJ;
    tau = 1
    f1_ES = E' * (Fs1 - .5*tau*c.*dh)
    f2_ES = E' * (Fs2 - .5*tau*c.*dhu)
    return f1_ES, f2_ES
end

function swe_1d_esdg_vol(UL_E, UR_E, ops, vgeo, i, j, btm, g)::Tuple{Float64,Float64,Float64,Float64}
    Q_ID, Qb_ID, Q_ES, Qb_ES, E, M_inv, Mf_inv = ops
    rxJ,J = vgeo
    (FxV1,FxV2)= fS1D(UL_E,UR_E,g)
    h_i, hu_i = UL_E; h_j, hu_j = UR_E;

    QNx_ij = Q_ES[i,j]*2;
    QNb_ij = Qb_ES[i,j];

    fv1_ES   =  QNx_ij*FxV1;
    fv2_i_ES =  QNx_ij*FxV2 + g*h_i*QNb_ij*btm[j];
    fv2_j_ES = -QNx_ij*FxV2 - g*h_j*QNb_ij*btm[i];
    return fv1_ES, fv2_i_ES, -fv1_ES, fv2_j_ES

end

function swe_1d_ID_surface(UL, UR, dU, E, nxJ, c, g)::Tuple{Array{Float64,2},Array{Float64,2}}
    # (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D_LF(UL,UR,g)
    (fxS1,fxS2) = fS1D(UL,UR,g)
    (hf, huf) = UL; (hP, huP) = UR; (dh, dhu) = dU;
    fs1 = fxS1.*nxJ; fs2 = fxS2.*nxJ;
    tau = 1
    f1_ID = E' * (0.5*fs1 .- .5*tau*c.*dh)
    f2_ID = E' * (0.5*fs2 .- .5*tau*c.*dhu)
    return f1_ID, f2_ID
end

function swe_1d_ID_vol(UL_E, ops, vgeo, i, j, btm, g)::Tuple{Float64,Float64}
    Q_ID, Qb_ID, Q_ES, Qb_ES, E, M_inv, Mf_inv = ops
    rxJ,J = vgeo
    # (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    (fxV1,fxV2)= fS1D(UL_E,UL_E,g)
    h_i, hu_i = UL_E;

    QNx_ij = Q_ID[i,j];QNb_ij = Qb_ID[i,j];

    fv1_ID = QNx_ij*fxV1;
    fv2_ID = QNx_ij*fxV2 + g*h_i*QNb_ij*btm[j];
    return fv1_ID, fv2_ID
end

function swe_1d_ID_h(UL_E, Q_ID, i, j, g)::Float64
    # (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    # (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    (fxV1,fxV2) = fS1D(UL_E,UL_E,g)
    Q_ID_ij = Q_ID[i,j]
    fv1_ID  = Q_ID_ij*fxV1
    return fv1_ID
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
