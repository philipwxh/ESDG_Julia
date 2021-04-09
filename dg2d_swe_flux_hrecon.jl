avg(a,b) = .5*(a+b)
function fS2D(UL,UR,tol,g)
    # hL,huL,hvL,bL = UL
    # hR,huR,hvR,bR = UR
    # uL,vL = (x->x./hL).((huL,hvL))
    # uR,vR = (x->x./hR).((huR,hvR))
    # # hLs = max.(0,hL .- tol + bL - max.(bL,bR))
    # # hRs = max.(0,hR .- tol + bR - max.(bL,bR))
    # # huL, hvL = (x->x.*hLs).((uL,vL))
    # # huR, hvR = (x->x.*hRs).((uR,vR))
    # fxS1 = @. avg(huL,huR)
    # fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*hL*hR
    # fxS3 = @. avg(huL,huR)*avg(vL,vR)
    #
    # fyS1 = @. avg(hvL,hvR)
    # fyS2 = @. avg(hvL,hvR)*avg(uL,uR)
    # fyS3 = @. avg(hvL,hvR)*avg(vL,vR) + .5*g*hL*hR
    # return (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3)

    # hL,huL,hvL,bL = UL
    # hR,huR,hvR,bR = UR
    # uL,vL = (x->x./hL).((huL,hvL))
    # uR,vR = (x->x./hR).((huR,hvR))
    # hLs = max.(0,hL .- tol + bL - max.(bL,bR))
    # hRs = max.(0,hR .- tol + bR - max.(bL,bR))
    # huL, hvL = (x->x.*hLs).((uL,vL))
    # huR, hvR = (x->x.*hRs).((uR,vR))
    # fxS1 = @. avg(huL,huR)
    # fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*hLs*hRs + .5*g*(hL.^2 - hLs.^2);
    # fxS3 = @. avg(huL,huR)*avg(vL,vR)
    #
    # fyS1 = @. avg(hvL,hvR)
    # fyS2 = @. avg(hvL,hvR)*avg(uL,uR)
    # fyS3 = @. avg(hvL,hvR)*avg(vL,vR) + .5*g*hLs*hRs + .5*g*(hL.^2 - hLs.^2);
    # return (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3)

    hL,huL,hvL,bL = UL
    hR,huR,hvR,bR = UR
    uL,vL = (x->x./hL).((huL,hvL))
    uR,vR = (x->x./hR).((huR,hvR))
    hLs = max.(0,hL .- tol + bL - max.(bL,bR))
    hRs = max.(0,hR .- tol + bR - max.(bL,bR))
    huL, hvL = (x->x.*hLs).((uL,vL))
    huR, hvR = (x->x.*hRs).((uR,vR))
    fxS1 = @. avg(huL,huR)
    fxS2 = @. avg(huL,huR)*avg(uL,uR) + .5*g*avg(hL*hL,hR*hR) + .5*g*(hL.^2 - hLs.^2);
    fxS3 = @. avg(huL,huR)*avg(vL,vR)

    fyS1 = @. avg(hvL,hvR)
    fyS2 = @. avg(hvL,hvR)*avg(uL,uR)
    fyS3 = @. avg(hvL,hvR)*avg(vL,vR) + .5*g*avg(hL*hL,hR*hR) + .5*g*(hL.^2 - hLs.^2);
    return (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3)
end

function swe_2d_esdg_surface(UL, UR, dU, Pf, fgeo, c, tol, g)::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}}
    nxJ,nyJ,sJ,nx,ny = fgeo
    (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D(UL,UR,tol,g)
    dh, dhu, dhv = dU
    (hf, huf, hvf)=UL;
    (hP, huP, hvP)=UR;
    fs1 = @. fxS1*nxJ + fyS1*nyJ;
    fs2 = @. fxS2*nxJ + fyS2*nyJ;
    fs3 = @. fxS3*nxJ + fyS3*nyJ;
    tau = 0
    f1_ES =  Pf*(fs1 .- .5*tau*c.*(hP.-hf).*sJ);
    f2_ES =  Pf*(fs2 .- .5*tau*c.*(huP.-huf).*sJ);
    f3_ES =  Pf*(fs3 .- .5*tau*c.*(hvP.-hvf).*sJ);
    return f1_ES, f2_ES, f3_ES
end

function swe_2d_esdg_vol(UL_E, UR_E, ops, vgeo_e, i, j, tol, g)
    Qr_ID,Qs_ID,Qrb_ID,Qsb_ID,Qr_ES,Qs_ES,QNr_sbp, QNs_sbp, E,M_inv,Pf= ops
    (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    (FxV1,FxV2,FxV3),(FyV1,FyV2,FyV3) = fS2D(UL_E,UR_E,tol,g)
    h_i, hu_i, hv_i, b_i = UL_E; h_j, hu_j, hv_j, b_j = UR_E;
    h_i = max(h_i-tol, 0); h_j = max(h_j-tol, 0);
    # QNx_ij = Qr_ES[i,j]*(rxJ[i,e]+ rxJ[j,e]) + Qs_ES[i,j]*(sxJ[i,e]+ sxJ[j,e]);
    # QNy_ij = Qr_ES[i,j]*(ryJ[i,e]+ ryJ[j,e]) + Qs_ES[i,j]*(syJ[i,e]+ syJ[j,e]);
    QNx_ij = Qr_ES[i,j]*(rxJ_i*2) + Qs_ES[i,j]*(sxJ_i*2);
    QNy_ij = Qr_ES[i,j]*(ryJ_i*2) + Qs_ES[i,j]*(syJ_i*2);

    QNxb_ij = QNr_sbp[i,j]*(rxJ_i) + QNs_sbp[i,j]*(sxJ_i);
    QNyb_ij = QNr_sbp[i,j]*(ryJ_i) + QNs_sbp[i,j]*(syJ_i);

    fv1_ES = (QNx_ij*FxV1 + QNy_ij*FyV1);
    if i ==1 && j ==2
        @show h_i, h_j, g*h_i*QNxb_ij*b_j, g*h_j*QNxb_ij*b_i
    end
    fv2_i_ES = (QNx_ij*FxV2 + QNy_ij*FyV2) + g*h_i*QNxb_ij*b_j;
    fv3_i_ES = (QNx_ij*FxV3 + QNy_ij*FyV3) + g*h_i*QNyb_ij*b_j;
    fv2_j_ES = -(QNx_ij*FxV2 + QNy_ij*FyV2) - g*h_j*QNxb_ij*b_i;
    fv3_j_ES = -(QNx_ij*FxV3 + QNy_ij*FyV3) - g*h_j*QNyb_ij*b_i;
    return fv1_ES, fv2_i_ES, fv3_i_ES, -fv1_ES, fv2_j_ES, fv3_j_ES
end

function swe_2d_ID_surface(UL, UR, dU, Pf, fgeo, c, tol, g)::Tuple{Array{Float64,2},Array{Float64,2},Array{Float64,2}}
    nxJ,nyJ,sJ,nx,ny = fgeo
    # (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D_LF(UL,UR,g)
    (fxS1,fxS2,fxS3),(fyS1,fyS2,fyS3) = fS2D(UL,UR,tol, g)
    dh, dhu, dhv = dU
    fs1 = @. fxS1*nxJ + fyS1*nyJ;
    fs2 = @. fxS2*nxJ + fyS2*nyJ;
    fs3 = @. fxS3*nxJ + fyS3*nyJ;
    tau = 1
    f1_ID = Pf * (0.5*fs1 .- .5*tau*c.*dh.*sJ)
    f2_ID = Pf * (0.5*fs2 .- .5*tau*c.*dhu.*sJ)
    f3_ID = Pf * (0.5*fs3 .- .5*tau*c.*dhv.*sJ)
    # f1_ID = Pf * (fs1 .- .5*tau*c.*dh.*sJ)
    # f2_ID = Pf * (fs2 .- .5*tau*c.*dhu.*sJ)
    # f3_ID = Pf * (fs3 .- .5*tau*c.*dhv.*sJ)
    # f1_ID = Pf * (0.5*fs1) - transpose(E)*(Cf.*c.*dh)
    # f2_ID = Pf * (0.5*fs2) - transpose(E)*(Cf.*c.*dhu)
    # f3_ID = Pf * (0.5*fs3) - transpose(E)*(Cf.*c.*dhv)
    return f1_ID, f2_ID, f3_ID
end

function swe_2d_ID_vol(UL_E, ops, vgeo_e, i, j, b_j, tol, g)
    Qr_ID,Qs_ID,Qrb_ID,Qsb_ID,Qr_ES,Qs_ES,QNr_sbp, QNs_sbp, E,M_inv,Pf= ops
    (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    # (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D(UL_E,UL_E,tol, g)
    h_i, hu_i, hv_i, b_i = UL_E;

    QNx_ij = Qr_ID[i,j]*rxJ_i + Qs_ID[i,j]*sxJ_i;
    QNy_ij = Qr_ID[i,j]*ryJ_i + Qs_ID[i,j]*syJ_i;

    QNxb_ij = Qrb_ID[i,j]*rxJ_i + Qsb_ID[i,j]*sxJ_i;
    QNyb_ij = Qrb_ID[i,j]*ryJ_i + Qsb_ID[i,j]*syJ_i;

    fv1_ID = QNx_ij*fxV1 + QNy_ij*fyV1;
    fv2_ID = QNx_ij*fxV2 + QNy_ij*fyV2 + g*h_i*QNxb_ij*b_j;
    fv3_ID = QNx_ij*fxV3 + QNy_ij*fyV3 + g*h_i*QNyb_ij*b_j;
    return fv1_ID, fv2_ID, fv3_ID
end

function swe_2d_ID_h(UL_E, Qr_ID, Qs_ID, vgeo_e, i, j, tol, g)
    (rxJ_i, sxJ_i, ryJ_i, syJ_i) = vgeo_e;
    # (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D_LF(UL_E,UL_E,g)
    (fxV1,fxV2,fxV3),(fyV1,fyV2,fyV3) = fS2D(UL_E,UL_E,tol,g)
    Qr_ID_ij = Qr_ID[i,j]; Qs_ID_ij = Qs_ID[i,j];
    dhdx  = rxJ_i*(Qr_ID_ij*fxV1) + sxJ_i*(Qs_ID_ij*fxV1)
    dhdy  = ryJ_i*(Qr_ID_ij*fyV1) + syJ_i*(Qs_ID_ij*fyV1)
    fv1_ID = dhdx  + dhdy
    return fv1_ID
end
