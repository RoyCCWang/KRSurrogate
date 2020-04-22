


@everywhere include("../tests/example_helpers/test_densities.jl")
@everywhere include("../src/misc/declarations.jl")

@everywhere include("../src/KR/engine.jl")
@everywhere include("../src/misc/normal_distribution.jl")
@everywhere include("../src/misc/utilities.jl")
@everywhere include("../src/integration/numerical.jl")
@everywhere include("../src/KR/dG.jl")
@everywhere include("../src/KR/d2G.jl")
@everywhere include("../src/KR/single_KR.jl")
@everywhere include("../tests/verification/differential_verification.jl")
@everywhere include("../src/kernel_centers/initial.jl")
@everywhere include("../src/kernel_centers/subsequent.jl")
@everywhere include("../src/kernel_centers/kDPP.jl")
@everywhere include("../src/fit/fit_adaptive_bp.jl")
@everywhere include("../src/fit/RKHS.jl")
@everywhere include("../src/misc/declarations.jl")
@everywhere include("../src/kernel_centers/front_end.jl")

@everywhere include("../src/kernel_centers/inference_kDPP.jl")

@everywhere include("../src/Taylor_inverse/front_end.jl")
@everywhere include("../src/Taylor_inverse/Taylor_inverse_helpers.jl")

@everywhere include("../src/quantile/setupTaylorquantile.jl")
@everywhere include("../src/quantile/quantile_engine.jl")
@everywhere include("../src/Taylor_inverse/ROC_check.jl")
@everywhere include("../src/Taylor_inverse/RQ_Taylor_quantile.jl")
@everywhere include("../src/integration/double_exponential.jl")

@everywhere include("../src/KR/transport.jl")


@everywhere include("../src/misc/chain_rule.jl")
@everywhere include("../src/KR/setupallKR.jl")


@everywhere include("../src/KR/cached_derivatives/RQ_adaptive.jl")
@everywhere include("../src/KR/cached_derivatives/d2g.jl")
@everywhere include("../src/KR/cached_derivatives/dg.jl")
@everywhere include("../src/KR/cached_derivatives/generic.jl")

@everywhere include("../src/KR/bundled/adaptive/KR_isonormal.jl")
@everywhere include("../src/KR/isonormal_common.jl")
@everywhere include("../src/misc/visualize.jl")


@everywhere include("../src/misc/parallel_utilities.jl")

@everywhere include("../tests/verification/fit_example_copulae.jl")

@everywhere include("../tests/verification/helpers.jl")
@everywhere include("../tests/example_helpers/demo_densities.jl")
@everywhere include("../tests/verification/adaptive/demo_cases.jl")

@everywhere include("../tests/verification/fit_example_gmm_cauchy.jl")
@everywhere include("../tests/verification/transport_derivative_helpers.jl")

@everywhere include("../src/KR/eval_KR_density.jl")

@everywhere include("../src/fresh/setup_quantile.jl")


PyPlot.close("all")
fig_num = 1

Random.seed!(25)


# set up transport map and derivatives..
initial_divisions = 1
T_map, ds_T_map, dT, quantile_map1,
    CDF_map1 = setupbundledKR( KR_config,
                                  c_array,
                                  θ_array,
                                  𝓧_array,
                                  dφ_array,
                                  d2φ_array,
                                  dϕ_array,
                                  d2ϕ_array,
                                  src_μ,
                                  src_σ_array,
                                  limit_a,
                                  limit_b;
                                  f_target = f,
                                  df_target = df_target,
                                  d2f_target = d2f_target,
                                  initial_divisions = initial_divisions)

# verify derivatives.
verifytransportderivatives(src_dist, T_map, ds_T_map, dT, length(limit_a))

# stress test verifytransport derivatives.

#include("helmet_second.jl")

CDF_map = xx->transportisonormaltoCDFspace(xx, src_μ, src_σ_array)

src_dist_array = collect( Distributions.Normal(src_μ[d], src_σ_array[d]) for d = 1:D )
CDF_ref = xx->collect( Distributions.cdf(src_dist_array[d], xx[d]) for d = 1:D )



T0 = uu->evalKR(   1,
                c_array,
                θ_array,
                𝓧_array,
                KR_config.max_integral_evals,
                KR_config.x_ranges,
                dφ_array,
                d2φ_array,
                KR_config.N_nodes_tanh,
                KR_config.m_tsq,
                KR_config.quantile_err_tol,
                KR_config.max_traversals,
                KR_config.N_predictive_traversals,
                KR_config.correction_epoch,
                KR_config.quantile_max_iters,
                KR_config.quantile_convergence_zero_tol,
                KR_config.n_limit,
                KR_config.n0,
                uu)


### test.
x0 = rand(src_dist)
u0 = CDF_ref(x0)
y0, discrepancy_y0 = T0(u0)

y1, discrepancy_y1 = T_map(x0)
u1 = CDF_map1(x0)
q1, discrepancy_q1 = quantile_map1(u1)

println("discrepancy between u0, u1 is ", norm(u0-u1))
println("discrepancy between q1, y1 is ", norm(y1-q1))
println("discrepancy between y0, y1 is ", norm(y1-y0[1]))
println()


### test again
x0 = rand(src_dist)
u0 = CDF_ref(x0)
y0, discrepancy_y0 = T0(u0)

y1, discrepancy_y1 = T_map(x0)
u1 = CDF_map1(x0)
q1, discrepancy_q1 = quantile_map1(u1)

println("discrepancy between u0, u1 is ", norm(u0-u1))
println("discrepancy between q1, y1 is ", norm(y1-q1))
println("discrepancy between y0, y1 is ", norm(y1-y0[1]))
println()

#@assert 1==2

# ### to see whether T0 is correct or T_map, look at first component only.
# # need CDF.
#
# g1 = gq_array[1]
# Z1 = evalintegral(g1, limit_a[1], limit_b[1])
#
# ### the y coord.
# a1 = y0[1][1]
# b1 = y1[1]
#
# # CDF evaluated at y.
# u_a1 = evalintegral(g1, limit_a[1], a1)/Z1
# u_b1 = evalintegral(g1, limit_a[1], b1)/Z1
#
# println("u0[1] = ", u0[1])
# println("u_a1 = ", u_a1)
# println("u_b1 = ", u_b1)
# println()
#
# @assert 1==2

N_viz = 10000
X0 = collect( rand(src_dist) for n = 1:N_viz )
X_idea = collect( randn(D) for n = 1:N_viz )

U0 = collect( collect( drawstdnormalcdf() for d = 1:D ) for n = 1:N_viz )

U1 = CDF_ref.(X0)

mean_U0 = Statistics.mean(U0)
mean_U1 = Statistics.mean(U1)

var_U0 = Statistics.var(U0)
var_U1 = Statistics.var(U1)

println("mean_U0 = ", mean_U0)
println("mean_U1 = ", mean_U1)
println("mean should be near 0.5")
println()

println("var_U0 = ", var_U0)
println("var_U1 = ", var_U1)
println("var should be near 1/12 = ", 1/12)
println()

## T_map.
println("Timing T_map, ", N_viz, " times.")
@time tmp1 = T_map.(X0)
println()

xp_array1 = collect( tmp1[n][1] for n = 1:N_viz)
disc_xp_array1 = collect( tmp1[n][2] for n = 1:N_viz)

# ## T0
# println("Timing T0, ", N_viz, " times.")
# @time tmp0 = T0.(U0)
# println()
#
# xp_array0 = collect( tmp0[n][1][1] for n = 1:N_viz)
# disc_xp_array0 = collect( tmp0[n][2][1] for n = 1:N_viz)

## T_map.
println("Timing quantile_map1, ", N_viz, " times.")
@time tmp3 = quantile_map1.(U1)
println()

xp_array3 = collect( tmp3[n][1] for n = 1:N_viz)
disc_xp_array3 = collect( tmp3[n][2] for n = 1:N_viz)

# @assert 1==233

xp_array = xp_array3
disc_xp_array1 = disc_xp_array3

max_val, max_ind = findmax(norm.(disc_xp_array))
println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(disc_xp_array)))
println("largest l-1 discrepancy is ", max_val)
println("At that case: x = ", xp_array[max_ind])
println()

n_bins = N_array[1] *10
fig_num = visualize2Dhistogram(fig_num, xp_array, limit_a, limit_b;
                                use_bounds = true, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "xp, bounds")
