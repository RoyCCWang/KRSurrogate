# based on bundled derivatives.

include("../../tests/example_helpers/test_densities.jl")
include("../../src/misc/declarations.jl")

include("../../src/KR/engine.jl")
include("../../src/misc/normal_distribution.jl")
include("../../src/misc/utilities.jl")
include("../../src/integration/numerical.jl")
include("../../src/KR/dG.jl")
include("../../src/KR/d2G.jl")
include("../../src/KR/single_KR.jl")
include("../../tests/verification/differential_verification.jl")
include("../../src/kernel_centers/initial.jl")
include("../../src/kernel_centers/subsequent.jl")
include("../../src/kernel_centers/kDPP.jl")
include("../../src/fit/fit_adaptive.jl")
include("../../src/fit/RKHS.jl")
include("../../src/misc/declarations.jl")
include("../../src/kernel_centers/front_end.jl")

include("../../src/kernel_centers/inference_kDPP.jl")

include("../../src/Taylor_inverse/front_end.jl")
include("../../src/Taylor_inverse/Taylor_inverse_helpers.jl")

include("../../src/quantile/setupTaylorquantile.jl")
include("../../src/quantile/quantile_engine.jl")
include("../../src/Taylor_inverse/ROC_check.jl")
include("../../src/Taylor_inverse/RQ_Taylor_quantile.jl")
include("../../src/integration/double_exponential.jl")

include("../../src/KR/transport.jl")
#include("../../src/KR/unbundled/adaptive/KR_isonormal.jl")
include("../../src/misc/chain_rule.jl")
include("../../src/KR/setupallKR.jl")


include("../../src/KR/cached_derivatives/RQ_adaptive.jl")
include("../../src/KR/cached_derivatives/d2g.jl")
include("../../src/KR/cached_derivatives/dg.jl")
include("../../src/KR/cached_derivatives/generic.jl")

include("../../src/KR/bundled/adaptive/KR_isonormal.jl")
include("../../src/KR/isonormal_common.jl")

include("../../tests/verification/fit_example_copulae.jl")

# non-adaptive.
include("../../src/KR/bundled/non_adaptive_isonormal.jl")
include("../../src/kernels/RQ_non_adaptive.jl")
include("../../src/KR/setupallKR_non_adaptive.jl")
include("../../src/KR/transport_non_adaptive.jl")
include("../../src/KR/misc_non_adaptive.jl")

#PyPlot.close("all")
#fig_num = 1

Random.seed!(205)


μ = randn(D)
σ_array = rand(D) .* 5.9
src_dist = Distributions.MvNormal(μ, diagm(σ_array))

initial_divisions = 1

## bundled map.

df_target = xx->ForwardDiff.gradient(f_target, xx)
d2f_target = xx->ForwardDiff.hessian(f_target, xx)

# force compile once.
y_test = (limit_a + limit_b) ./ 2
df_target(y_test)
d2f_target(y_test)

T_map, ds_T_map, dT = setupbundledKRnonadaptive( KR_config,
                                  c_array,
                                  θ_array,
                                  𝓧_array,
                                  μ,
                                  σ_array,
                                  limit_a,
                                  limit_b;
                                  f_target = f_target,
                                  df_target = df_target,
                                  d2f_target = d2f_target,
                                  initial_divisions = initial_divisions)

##
x0 = rand(src_dist)

y0, err_y0 = T_map(x0)

#@assert 3333==4

# ## visualize transport.
# N_viz = 1000
# X = collect( rand(src_dist) for n = 1:N_viz )
#
# println("Timing: transport.")
# @time results = T_map.(X)
# println("end timing.")
# println()
#
# x_array = collect( results[n][1] for n = 1:N_viz)
# discrepancy_array = collect( results[n][2] for n = 1:N_viz)
#
#
# max_val, max_ind = findmax(norm.(discrepancy_array))
# println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(discrepancy_array)))
# println("largest l-1 discrepancy is ", max_val)
# println("At that case: x = ", x_array[max_ind])
# println()
#
#
#
# #### visualize histogram.
#
# plot_flag = true
# n_bins = 500
# use_bounds = true
# bounds = [[limit_a[2], limit_b[2]], [limit_a[1], limit_b[1]]]
#
# if plot_flag
#         PyPlot.figure(fig_num)
#         fig_num += 1
#         p1 = collect(x_array[n][2] for n = 1:N_viz)
#         p2 = collect(x_array[n][1] for n = 1:N_viz)
#
#         if use_bounds
#             PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="jet")
#         else
#             PyPlot.plt.hist2d(p1, p2, n_bins, cmap="jet" )
#         end
#
#
#         # PyPlot.figure(fig_num)
#         # fig_num += 1
#         # p1 = collect(randn() for n = 1:N_viz)
#         # p2 = collect(randn() for n = 1:N_viz)
#         #
#         # if use_bounds
#         #     PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="Greys")
#         # else
#         #     PyPlot.plt.hist2d(p1, p2, n_bins, cmap="Greys" )
#         # end
#         # #PyPlot.plt.axis("equal")
#
# end
#
# @assert 5555==4

dT_ND = xx->FiniteDiff.finite_difference_jacobian(aa->T_map(aa)[1], xx)

T_map_components = collect( xx->T_map(xx)[1][d] for d = 1:D )
d2T_components_ND = collect( xx->FiniteDiff.finite_difference_hessian(T_map_components[d], xx) for d = 1:D )
d2T_ND = xx->collect( d2T_components_ND[d](xx) for d = 1:D )

## test.

println(" test ds_T_map ")
x0 = rand(src_dist)

y0, err_y0 = T_map(x0)
dT_x0_AN, d2T_x0_AN = ds_T_map(x0,y0)

dT_x0_ND = dT_ND(x0)
println("ND: dT(x0)")
display(dT_x0_ND)
println()

println("AN: dT(x0)")
display(dT_x0_AN)
println()

println("dT discrepancy: ", norm(dT_x0_AN-dT_x0_ND))
println()

d2T_x0_ND = d2T_ND(x0)
println("ND: d2T(x0)")
display(d2T_x0_ND)
println()

println("AN: d2T(x0)")
display(d2T_x0_AN)
println()

disc_ratio = evalKRd2Tratiodiscrepancy(d2T_x0_AN, d2T_x0_ND)
println("d2T discrepancy ratio: ", disc_ratio)
println()

ratio_tol = 1e-2
println("Are the ratios close to one, within a tol of ", ratio_tol, ", per dim?")
println( ratiodiscrepancywithintol(disc_ratio; tol = ratio_tol) )

### again.
println()
println()
println(" again ds_T_map ")
x0 = rand(src_dist)

y0, err_y0 = T_map(x0)
dT_x0_AN, d2T_x0_AN = ds_T_map(x0,y0)

dT_x0_ND = dT_ND(x0)
println("ND: dT(x0)")
display(dT_x0_ND)
println()

println("AN: dT(x0)")
display(dT_x0_AN)
println()

println("dT discrepancy: ", norm(dT_x0_AN-dT_x0_ND))
println()

d2T_x0_ND = d2T_ND(x0)
println("ND: d2T(x0)")
display(d2T_x0_ND)
println()

println("AN: d2T(x0)")
display(d2T_x0_AN)
println()

disc_ratio = evalKRd2Tratiodiscrepancy(d2T_x0_AN, d2T_x0_ND)
println("d2T discrepancy ratio: ", disc_ratio)
println()

ratio_tol = 1e-2
println("Are the ratios close to one, within a tol of ", ratio_tol, ", per dim?")
println( ratiodiscrepancywithintol(disc_ratio; tol = ratio_tol) )

# write official test scripts for the 4 KR cases.
# skip last, adaptive
# fit all, adaptive
# skip last, non-adaptive
# fit all, non-adaptive

@assert 987 == 123


# next, do fitall RQ non-adaptive. make sure it is fast.

# about the same time. 424 ms.
@btime T_map(x0)
@btime ds_T_map(x0,y0)


# uregent:
# 1. next, verify everything in D = 3 and 4, with made-up fit.
# 3. T_map should return density evaluation as a bundle.

# non-adaptive RQ KR-related.
# 4. T_map, ds_T_map for RQ non-adaptive.
# 4.1 show RQ non-adaptive on a grid have problems (does it?),
#     thus, want adaptive kernel for higher accuracy.

# Fitting-related:
# 2. NNLS for massive number non-adaptive RQ fit. for publication,
      # fix RQ bandmwidth or do hyperparam optim on single a_RQ.
#     do particle swarm hyperparam optim on the non-adaptive RQ bandwidth. (or use Bayesian optim)

# complete 1-4, and we are ready to integrate KR to GFlow.
# 5. speed up GFlow code. stabilize, speed up.

# run against data.
# 6. compare GFlow-non-adaptive against MCMC.
# 7. compare GFlow-adaptive-seed-MCMC against MCMC.
# 8. verify KR-adaptive is doing its job.

# finally: refactor code.

# paper motivation guideline:
# exp(fit) is ill-conditioned for regions of x such that f(x) ≈ 0.
# (fit)^2 has zero-crossings.
# direct fit of quantile is costly.

# paper synthetic examples guideline.
# visualize toy distributions.
# do Eyam, 2D. expensive to evaluate points.
# do strange distribution on dirichlet or von Mis.

# future:
# -  estimate warp map by fitting non-adaptive RQ, see where
#     fit is bad for a kernel.. if increasing improves region
# - better warpmap strategies.
# - 2nd derivatives and SDEs. WHy does it work so well as a warpmap for 1D cases?
