using Basis2DTri
# using SetupDG
# using UniformTriMesh
using NodesAndModes
using NodesAndModes.Tri
using UnPack
function SWE_vortex( x, y, t )
    # % This function calculate the analytical solution of vortex translate
    # % across the domain without changing its shape .
    # % H = H_inf - ?^2 / (32 * pi^2) * e^( -2(r^2 - 1) ),
    # % u = u_inf - ?   / (2 * pi)    * e^(  -(r^2 - 1) ) * yt
    # % v = v_inf + ?   / (2 * pi)    * e^(  -(r^2 - 1) ) * xt,
    H_inf = 1;
    u_inf = 1;
    v_inf = 0;
    xc = 0;
    yc = 0;
    xt = x .- xc .- u_inf*t;
    yt = y .- yc .- v_inf*t;
    r_sq = xt.^2 + yt.^2;
    beta = 5;
    g = 2;
    H = H_inf .- beta^2 / (32 * pi^2) * exp.( -2 * (r_sq .- 1) );
    u = u_inf .- beta   / (2 * pi)    * exp.(  -( r_sq .- 1) ) .* yt;
    v = v_inf .+ beta   / (2 * pi)    * exp.(  -( r_sq .- 1) ) .* xt;
    return H,u,v
end

function L2_error_quadrature_2D(X, Y, hxy, huxy, hvxy, Wxy, t)
    Hxy, Uxy, Vxy = SWE_vortex(X,Y,t);
    HUxy = Hxy.*Uxy;
    HVxy = Hxy.*Vxy;
    EHxy = hxy - Hxy;
    EHUxy = huxy - HUxy;
    EHVxy = hvxy - HVxy;
    res = sum(sum(Wxy.*(EHxy.^2)));
    res = res + sum(sum(Wxy.*(EHUxy.^2)));
    res = res + sum(sum(Wxy.*(EHVxy.^2)));
    res = sqrt(res);
end

function LInf_error_quadrature_2D(X, Y, h, hu, hv, t)

    hex, huex, hvex = SWE_vortex(X,Y,t);

    Eh  = abs.(hex  - h);
    Ehu = abs.(huex - hu);
    Ehv = abs.(hvex - hv);

    res = maximum([Eh;Ehu;Ehv]);
end

function SWE_2D_Error_quad_sbp(N, V, Vq, x, y, h, hu, hv, t)

rqe,sqe,wqe = quad_nodes_2D(2*(N+2));
Vqe = Tri.vandermonde(N,rqe,sqe)/V;
Vqe_x = Vqe*x;
Vqe_y = Vqe*y;
hex, uex, vex = SWE_vortex( Vqe_x, Vqe_y, t );
huex = hex .* uex;
hvex = hex .* vex;
he = Vqe*h;
hue = Vqe*hu;
hve = Vqe*hv;


L2_err = L2_error_quadrature_2D(Vqe_x, Vqe_y, he, hue, hve, wqe, t)
Linf_err = LInf_error_quadrature_2D(Vqe_x, Vqe_y, he, hue, hve, t)
@show L2_err, Linf_err
end
