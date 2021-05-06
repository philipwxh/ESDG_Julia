avg(a,b) = .5*(a+b)
function fS1D(UL,UR,g, tol)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL
    uR = huR./hR
    if hL< tol
        uL = 0;
    end
    if hR<tol
        uR = 0;
    end
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*hL*hR
    return fxS1,fxS2
end

function fS1D_sur(UL,UR,g, tol)
    hL,huL = UL
    hR,huR = UR
    uL = huL./hL; uL[findall(x->x<2*tol, hL)] .= 0;
    uR = huR./hR; uR[findall(x->x<2*tol, hR)] .= 0;
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*hL*hR
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

function swe_1d_esdg_surface(UL, UR, dU, E, nxJ, c, g, tol)::Tuple{Array{Float64,2},Array{Float64,2}}
    (fxS1,fxS2) = fS1D_sur(UL,UR,g, tol)
    (hf, huf) = UL; (hP, huP) = UR; (dh, dhu) = dU;
    Fs1 = fxS1.*nxJ; Fs2 = fxS2.*nxJ;
    tau = 1
    f1_ES = E' * (Fs1 - .5*tau*c.*dh)
    f2_ES = E' * (Fs2 - .5*tau*c.*dhu)
    return f1_ES, f2_ES
end

function swe_1d_esdg_vol(UL_E, UR_E, ops, vgeo, i, j, btm, g, tol)
    Q_ID, Qb_ID, Q_ES, Qb_ES, E, M_inv, Mf_inv = ops
    rxJ,J = vgeo
    (FxV1,FxV2)= fS1D(UL_E,UR_E,g, tol)
    h_i, hu_i = UL_E; h_j, hu_j = UR_E;

    QNx_ij = Q_ES[i,j]*2;
    QNb_ij = Qb_ES[i,j];

    fv1_ES   =  QNx_ij*FxV1;
    fv2_i_ES =  QNx_ij*FxV2 + g*h_i*QNb_ij*btm[j];
    fv2_j_ES = -QNx_ij*FxV2 - g*h_j*QNb_ij*btm[i];
    return fv1_ES, fv2_i_ES, -fv1_ES, fv2_j_ES

end

function swe_1d_ID_surface(UL, UR, dU, E, nxJ, c, g, tol)::Tuple{Array{Float64,2},Array{Float64,2}}
    # (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D_LF(UL,UR,g)
    (fxS1,fxS2) = fS1D_sur(UL,UR,g, tol)
    (hf, huf) = UL; (hP, huP) = UR; (dh, dhu) = dU;
    fs1 = fxS1.*nxJ; fs2 = fxS2.*nxJ;
    tau = 1
    f1_ID = E' * (0.5*fs1 .- .5*tau*c.*dh)
    f2_ID = E' * (0.5*fs2 .- .5*tau*c.*dhu)
    return f1_ID, f2_ID
end

function swe_1d_ID_vol(UL_E, ops, vgeo, i, j, btm, g, tol)
    Q_ID, Qb_ID, Q_ES, Qb_ES, E, M_inv, Mf_inv = ops
    rxJ,J = vgeo
    # (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    (fxV1,fxV2)= fS1D(UL_E,UL_E,g, tol)
    h_i, hu_i = UL_E;

    QNx_ij = Q_ID[i,j];QNb_ij = Qb_ID[i,j];

    fv1_ID = QNx_ij*fxV1;
    fv2_ID = QNx_ij*fxV2 + g*h_i*QNb_ij*btm[j];
    return fv1_ID, fv2_ID
end

function swe_1d_ID_h(UL_E, Q_ID, i, j, g, tol)
    # (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    # (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    (fxV1,fxV2) = fS1D(UL_E,UL_E,g, tol)
    Q_ID_ij = Q_ID[i,j]
    fv1_ID  = Q_ID_ij*fxV1
    return fv1_ID
end
