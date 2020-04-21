

include("../tests/example_helpers/test_densities.jl")
include("../src/misc/declarations.jl")

include("../src/KR/engine.jl")
include("../src/misc/normal_distribution.jl")
include("../src/misc/utilities.jl")
include("../src/integration/numerical.jl")
include("../src/KR/dG.jl")
include("../src/KR/d2G.jl")
include("../src/KR/single_KR.jl")
include("../tests/verification/differential_verification.jl")
include("../src/kernel_centers/initial.jl")
include("../src/kernel_centers/subsequent.jl")
include("../src/kernel_centers/kDPP.jl")
include("../src/fit/fit_adaptive.jl")
include("../src/fit/RKHS.jl")
include("../src/misc/declarations.jl")
include("../src/kernel_centers/front_end.jl")

include("../src/kernel_centers/inference_kDPP.jl")

include("../src/Taylor_inverse/front_end.jl")
include("../src/Taylor_inverse/Taylor_inverse_helpers.jl")

include("../src/quantile/setupTaylorquantile.jl")
include("../src/quantile/quantile_engine.jl")
include("../src/Taylor_inverse/ROC_check.jl")
include("../src/Taylor_inverse/RQ_Taylor_quantile.jl")
include("../src/integration/double_exponential.jl")

include("../src/KR/transport.jl")
include("../src/KR/unbundled/adaptive/KR_isonormal.jl")
include("../src/misc/chain_rule.jl")
include("../src/KR/setupallKR.jl")

include("../src/KR/cached_derivatives/RQ_adaptive.jl")
include("../src/KR/cached_derivatives/d2g.jl")
include("../src/KR/cached_derivatives/dg.jl")

include("../src/KR/isonormal_common.jl")

include("../tests/verification/fit_example_copulae.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(205)

# initial_divisions = 1
# zero_tol = 1e-9
#
# gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
#                         max_integral_evals = KR_config.max_integral_evals,
#                         initial_divisions = initial_divisions)
# #
# fq_array, updatey_array, fetchZ_array, fetchv_array,
#     fetchy1d_array = getfvarray( c_array, 𝓧_array, θ_array;
#                 zero_tol = zero_tol,
#                 initial_divisions = initial_divisions,
#                 max_integral_evals = KR_config.max_integral_evals)
#
#
#
# #
# d_select = 2
#
# # assess correctness.
# y0 = rand(2)
#
# updatey_array[d_select](y0, d_select)
# a = fq_array[d_select](y0[d_select])
# println("a = ", a)
#
# b = gq_array[d_select](y0)
# println("b = ", b)
# println("norm(a-b) = ", norm(a-b))
#
# # derive dG and d2G from this.
# @assert 1==2
#
# # timing.
# @btime updatey_array[d_select](y0, d_select);
# @btime fq_array[d_select](y0)
#
# @btime gq_array[d_select](y0)
#
# @assert 1==2


#
μ = randn(D)
σ_array = rand(D) .* 5.9
src_dist = Distributions.MvNormal(μ, diagm(σ_array))

initial_divisions = 1



# # I am here. too slow and inaccurate.
# #println("AN: df_via_y(y0) = ", df_via_y(y0))
# #println("discrepancy = ", norm(df_via_y(y0)-dQ_u0_ND))
# println()
#
# testfunc = yy->evaldg99(updatev_array, dg_array, yy)
#
#
# z = testfunc(y0)
# display(z)
#
# ref = dQinv(y0)
# display(ref)
#
# #ref2 = convertdFinvtolowertriangular(ref)
# @assert 1==2
#
# @btime testfunc(y0)
# @btime testfunc(y0)
#
# @btime dQinv(y0)
# @btime dQinv(y0)
#
# @btime updatev_array[2](y0[1:2],2)

### non-simultaneous version.
# T_map, dT, CDF_map, quantile_map, dCDF,
#   dQinv,  dg_array, updatev_array = setupTmapfromisonormal( KR_config,
#                                     c_array,
#                                     θ_array,
#                                     𝓧_array,
#                                     dφ_array,
#                                     d2φ_array,
#                                     μ,
#                                     σ_array,
#                                     limit_a,
#                                     limit_b;
#                                     initial_divisions = initial_divisions)


# simultaneous updates and compute for transport and derivatives.
T_map, dT, d2T, CDF_map, quantile_map,
      dCDF, d2CDF, dg_array, d2g_array,
      updatedfbuffer_array,
      fq_v_array,
      gq_array,
      dQ_via_y, d2Q_via_y,
      ∂f_∂x_array,
      d2G = setupTmapfromisonormal2( KR_config,
                                  c_array,
                                  θ_array,
                                  𝓧_array,
                                  dφ_array,
                                  d2φ_array,
                                  dϕ_array,
                                  d2ϕ_array,
                                  μ,
                                  σ_array,
                                  limit_a,
                                  limit_b;
                                  initial_divisions = initial_divisions)

##

dT_ND = xx->FiniteDiff.finite_difference_jacobian(aa->T_map(aa)[1], xx)
dCDF_ND = xx->FiniteDiff.finite_difference_jacobian(CDF_map, xx)
dQ_ND = xx->FiniteDiff.finite_difference_jacobian(aa->quantile_map(aa)[1], xx)


### compare values.

updatemyfunc = yy->collect( updatedfbuffer_array[d](yy[1:d]) for d = 1:D )
myfunc = yy->collect( dg_array[d](yy[d]) for d = 1:D )
d2G_AN = yy->collect( d2g_array[d](yy[d]) for d = 1:D )

println("Test derivative of quantile")
x0 = rand(src_dist)
y0, err_y0 = T_map(x0)
u0 = CDF_map(x0)
updatemyfunc(y0)
K = myfunc(y0)

K_mat = convertdFinvtolowertriangular(K)
dQ_AN_via_y_y0 = inv(K_mat)
display(dQ_AN_via_y_y0)

dQ_u0_ND = dQ_ND(u0)

println("AN: dQinv(y0) = ", dQ_AN_via_y_y0)
println("ND: dQ(u0) = ", dQ_u0_ND)
println("discrepancy = ", norm(dQ_AN_via_y_y0-dQ_u0_ND))
println()

### repeat the test.
println("Test derivative of quantile")
x0 = rand(src_dist)
y0, err_y0 = T_map(x0)
u0 = CDF_map(x0)
updatemyfunc(y0)
K = myfunc(y0)

K_mat = convertdFinvtolowertriangular(K)
dQ_AN_via_y_y0 = inv(K_mat)
display(dQ_AN_via_y_y0)

dQ_u0_ND = dQ_ND(u0)

println("AN: dQinv(y0) = ", dQ_AN_via_y_y0)
println("ND: dQ(u0) = ", dQ_u0_ND)
println("discrepancy = ", norm(dQ_AN_via_y_y0-dQ_u0_ND))
println()


### test for full transport.
println("test for full transport.")
x0 = rand(src_dist)

y0, err_y0 = T_map(x0)
dT_x0_AN = dT(x0,y0)
dT_x0_ND = dT_ND(x0)

println("AN: dT(x0) = ", dT_x0_AN)
println("ND: dT(x0) = ", dT_x0_ND)
println("discrepancy = ", norm(dT_x0_AN-dT_x0_ND))
println()

println("Test derivative of CDF")
dCDF_x0_AN = diagm(dCDF(x0))
dCDF_x0_ND = dCDF_ND(x0)

println("AN: dCDF(x0) = ", dCDF_x0_AN)
println("ND: dCDF(x0) = ", dCDF_x0_ND)
println("discrepancy = ", norm(dCDF_x0_AN-dCDF_x0_ND))
println()

############ second-order.
println()
println("Second-order tests.")
println()

# CDF_array should be defined in first_part.jl
d2G_ND = xx->collect( FiniteDiff.finite_difference_hessian(CDF_array[d], xx[1:d]) for d = 1:D )



x0 = rand(src_dist)
y0, err_y0 = T_map(x0)
d2G_y0_AN = d2G(y0)
println("AN: d2G(y0):")
display(d2G_y0_AN)
println()


d2G_y0_ND = d2G_ND(y0)
println("ND: d2G(y0):")
display(d2G_y0_ND)
println()

y0, err_y0 = T_map(x0)
dT_x0_AN = dT(x0,y0)
d2G_y0_my = d2G_AN(y0)
println("speed up AN: d2G(y0):")
display(d2G_y0_my)
println()

println("discrepancy mine vs. ND: ", sum(norm.(d2G_y0_my-d2G_y0_ND)))
println()


## again.
println(" again ")
x0 = rand(src_dist)

y0, err_y0 = T_map(x0)
dT_x0_AN = dT(x0,y0)
d2G_y0_my = d2G_AN(y0)
println("speed up AN: d2G(y0):")
display(d2G_y0_my)
println()

d2G_y0_ND = d2G_ND(y0)
println("ND: d2G(y0):")
display(d2G_y0_ND)
println()

println("discrepancy mine vs. ND: ", sum(norm.(d2G_y0_my-d2G_y0_ND)))
println()

##### d2T.


## second-order derivaitves of the component maps.
Q_components = collect( uu->quantile_map(uu)[1][d] for d = 1:D )
d2Q_components_ND = collect( uu->FiniteDiff.finite_difference_hessian(Q_components[d], uu) for d = 1:D )


## test.


# I am here. get the notation symbols correct.
d2g_y0_AN = d2G_AN(y0)
d2G_x0_full = packaged2G(d2g_y0_AN)
dF_u0_mat = dQ_via_y(y0)

k_select = 2
RHS = sum( dF_u0_mat[k_select,l] .* d2G_x0_full[l] for l = 1:D )
RHS2 = evalRHSinvfuncTHMlowertriangular(d2g_y0_AN, dF_u0_mat, k_select)

A = -(dF_u0_mat'*RHS*dF_u0_mat)
println("A:")
display(A)
println()

A2 = applyinvfuncTHMd2(d2g_y0_AN, dF_u0_mat)
d2Q_u = applyinvfuncTHMd2lowertriangular(d2g_y0_AN, dF_u0_mat)
println("discrepancy test for applyinvfuncTHMd2() is ", norm(A-A2[k_select]))
println("discrepancy test for applyinvfuncTHMd2lowertriangular() is ", norm(A-d2Q_u[k_select]))
println()

u0 = CDF_map(x0)
B = d2Q_components_ND[k_select](u0)
println("B:")
display(B)
println()

println("discrepancy between ND and AN for d2Q_u is:")
display(norm.(collect(d2Q_components_ND[k](u0) for k = 1:D)-d2Q_u))
println()

println()
println("debug dQ")
println()

#x0 = rand(src_dist)
x0 = [2.2878340847830425;
 2.289536801444754]
u0 = CDF_map(x0)
y0, err_y0 = T_map(x0)
dT_x0_AN = dT(x0,y0)

d2Q_u0_ND = collect(d2Q_components_ND[k](u0) for k = 1:D)

d2Q_u0_AN = d2Q_via_y(y0)

println("AN: d2Q(u0)")
display(d2Q_u0_AN)
println()

println("ND: d2Q(u0)")
display(d2Q_u0_ND)
println()

println("d2Q(u0) discrepancy ratio: ",
      evalKRd2Tratiodiscrepancy(d2Q_u0_AN, d2Q_u0_ND))
println()


#@assert 1==2

## second-order derivaitves of the component maps.
println()
println("transport, d2:")
println()
Tmap_components = collect( xx->T_map(xx)[1][d] for d = 1:D )
d2Tmap_ND = collect( xx->FiniteDiff.finite_difference_hessian(Tmap_components[d], xx) for d = 1:D )

d2T_x0_ND = collect( d2Tmap_ND[d](x0) for d = 1:D )
println("ND: d2T(x0)")
display(d2T_x0_ND)
println()

d2CDF_x = d2CDF(x0)
dCDF_x = diagm(dCDF(x0))
d2Q_u = d2Q_via_y(y0)

dQ_u = dQ_via_y(y0)
d2T_x0_AN = applychainruled2(dCDF_x, d2CDF_x,
                       dQ_u, d2Q_u)

println("AN: d2T(x0):")
display(d2T_x0_AN)
println()

println("discrepancy between ND and AN for d2T is:")
display(norm.(d2T(x0,y0)-d2T_x0_ND))
println()

A = d2T(x0,y0)
B = d2T_x0_ND
println("d2T discrepancy ratio: ",
      evalKRd2Tratiodiscrepancy(A, B))
println()


@assert 5==4



# about the same time. 424 ms.
@btime d2G(y0)
@btime d2G_ND(y0)
@btime d2G_AN(y0)

@btime dT(x0,y0)

#go back to original plan. no simultaneous. too hard to debug.

# I am here. next, plot histogram to visually sanity-check if transport map is "correct"
# then, code 2nd derivatives. then, verify the derivatives with ND.

N_viz = 10000


# need to generate a colletion of src particles, draw from src_dist.
