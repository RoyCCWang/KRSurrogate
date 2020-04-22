using Distributed
@everywhere using SharedArrays

#@everywhere import JLD
@everywhere import FileIO

@everywhere import NearestNeighbors

@everywhere import Printf
@everywhere import PyPlot
@everywhere import Random
@everywhere import Optim

@everywhere using LinearAlgebra

@everywhere using FFTW

@everywhere import Statistics

@everywhere import Distributions
@everywhere import HCubature
@everywhere import Interpolations
@everywhere import SpecialFunctions

@everywhere import SignalTools
@everywhere import RKHSRegularization
@everywhere import Utilities

@everywhere import Convex
@everywhere import SCS

@everywhere import Calculus
@everywhere import ForwardDiff
@everywhere import FiniteDifferences

@everywhere import VisualizationTools

@everywhere import Printf
@everywhere import GSL
@everywhere import FiniteDiff

@everywhere using BenchmarkTools

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

#@everywhere include("../tests/verification/fit_example_copulae.jl")

@everywhere include("../tests/verification/helpers.jl")
@everywhere include("../tests/example_helpers/demo_densities.jl")
@everywhere include("../tests/verification/adaptive/demo_cases.jl")

#@everywhere include("../tests/verification/fit_example_gmm_cauchy.jl")
@everywhere include("../tests/verification/transport_derivative_helpers.jl")

@everywhere include("../src/KR/eval_KR_density.jl")

# for non-adaptive.
@everywhere include("../src/fit/fit_non_adaptive.jl")

@everywhere include("../src/KR/bundled/non_adaptive_isonormal.jl")
@everywhere include("../src/kernels/RQ_non_adaptive.jl")
@everywhere include("../src/KR/setupallKR_non_adaptive.jl")
@everywhere include("../src/KR/transport_non_adaptive.jl")
@everywhere include("../src/KR/misc_non_adaptive.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


# set up target.
D = 2

# Y_pdf, Y_dist, x_ranges = setupskipDgmm3D(limit_a, limit_b, N_array)

#scene_tag = "squiggle"
scene_tag = "helmet"

f, x_ranges, limit_a, limit_b, fig_num = setuporaclefromimage(scene = scene_tag)
N_array = collect( length(x_ranges[d]) for d = 1:D )



f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

f_ratio_tol = 1e-8
status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges, f_ratio_tol)
@assert status_flag

# set up kernel centers.
# downsample by 2.
downsample_factor = 1
Np_array = collect( div(N_array[d], downsample_factor) for d = 1:2 )
Xp_ranges = collect( LinRange(limit_a[d], limit_b[d], Np_array[d] ) for d = 1:D )
Xp = Utilities.ranges2collection(Xp_ranges, Val(D))

𝑋 = vec(Xp)
#𝑋 = collect( rand(Y_dist) for n = 1:N_realizations)

N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
#N_X[1] = 25
println("N_X = ", N_X)

fig_num = imshowgrayscale(fig_num, f.(Xp), "f.(Xp)")


# fit density via Bayesian optimization.


#a_array = [0.07; 1.0] # show how we cannot achieve good fit since we're restricting coeffs to be positive.
a_array = [1.0; 1.0] #good for downsample_factor = 1

σ_array = [1e-6; 1e-6]

D_fit = D
X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D_fit )
@time f_X_array = collect( f_joint.(X_array[d]) for d = 1:D_fit )


skip_flag = true
#skip_flag = false

max_fit_iters = 2000 #15000

c_array, 𝓧_array, θ_array, f, limit_a, limit_b, x_ranges,
  src_μ, src_σ_array, f_joint = demohelmetnonadaptive(a_array, σ_array;
                        downsample_factor = downsample_factor,
                        skip_flag = skip_flag,
                        max_iters = max_fit_iters,
                        scene_tag = scene_tag)

#
#fq = xx->RKHSRegularization.evalquery(xx, c_array[2], 𝓧_array[2], θ_array[2])

# visualize.


# visualize the surrogate density, full joint.
max_integral_evals = 10000 #typemax(Int)
initial_divisions = 1
gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                        max_integral_evals = max_integral_evals,
                        initial_divisions = initial_divisions)

pdf_g = setupKRdensityeval(gq_array, limit_a, limit_b;
                        initial_divisions = initial_divisions,
                        max_integral_evals = max_integral_evals)

#
if !skip_flag
    fig_num = visualizefit2Dmarginal(fig_num, c_array, 𝓧_array, θ_array,
                                limit_a, limit_b, f_joint)

    #
    #
    Nv_array = collect( round(Int, Np_array[d] .* 1.1) for d = 1:D )
    xv_ranges = collect( LinRange(limit_a[d], limit_b[d], Nv_array[d]) for d = 1:2 )
    Xv_nD = Utilities.ranges2collection(xv_ranges, Val(2))


    # println("Timing, pdf_g.(Xv_nD)")
    # @time g_Xv_nD = reshape(parallelevals(pdf_g, vec(Xv_nD)), size(Xv_nD))
    # println("End timing.")

    # fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
    #                   g_Xv_nD, [], "x", fig_num, "g")

    #imshowgrayscale(g_Xv_nD,)
end

#@assert 1==2

Random.seed!(205)


src_dist, df_target, d2f_target,
    KR_config = setupquantilefordemo(f, x_ranges, limit_a, limit_b,
                                        src_μ, src_σ_array;
                                        quantile_err_tol = 1e-6,
                                        quantile_convergence_zero_tol = 1e-8)

# set up transport map and derivatives..
initial_divisions = 1
T_map, ds_T_map, dT = setupbundledKRnonadaptive( KR_config,
                                  c_array,
                                  θ_array,
                                  𝓧_array,
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

#N_visualization = 1000
N_visualization = 100000 # for helmet. use multiple cores.

if scene_tag != "helmet"
    N_visualization = 1000
end

println("batch transport timing:")

### sequential version.
# @time x_array, discrepancy_array,
#         x_src_array = runbatchtransport( src_dist,
#                                         T_map,
#                                         ds_T_map, dT;
#                                         N_visualization = N_visualization)

### parallel version.
N_batches = 15 # must be larger than N_batch.
@time x_array, discrepancy_array = runbatchtransportparallel( N_visualization,
                                        N_batches,
                                        src_dist,
                                        T_map,
                                        ds_T_map, dT;
                                        N_visualization = N_visualization)
println("end timing.")
println()



# #### visualize histogram.
n_bins = N_array[1] *10
#n_bins = N_array[1] *3
fig_num = visualize2Dhistogram(fig_num, x_array, limit_a, limit_b;
                                use_bounds = true, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "xp, bounds",
                                flip_vertical_flag = true)

fig_num = visualize2Dhistogram(fig_num, x_array, limit_a, limit_b;
                                use_bounds = false, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "xp, no bounds",
                                flip_vertical_flag = true)

#### TODO: check that skip_flag = true also works. Then non-adaptive skip and no skip.
# refactor code.
# work on coupla example, 4-D.
