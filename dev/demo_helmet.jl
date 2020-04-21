@everywhere using Distributed
@everywhere using SharedArrays

@everywhere import JLD
@everywhere import FileIO

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


@everywhere import stickyHDPHMM

@everywhere import Printf
@everywhere import GSL

@everywhere import NNLS

@everywhere include("../src/approximation/approx_helpers.jl")
@everywhere include("../src/approximation/final_approx_helpers.jl")
@everywhere include("../src/approximation/analytic_cdf.jl")
@everywhere include("../src/approximation/fit_mixtures.jl")
@everywhere include("../src/approximation/optimization.jl")

@everywhere include("../src/integration/numerical.jl")
@everywhere include("../src/integration/adaptive_helpers.jl")
@everywhere include("../src/integration/DEintegrator.jl")

@everywhere include("../src/density/density_helpers.jl")
@everywhere include("../src/density/fit_density.jl")

@everywhere include("../src/misc/final_helpers.jl")
@everywhere include("../src/misc/test_functions.jl")
@everywhere include("../src/misc/utilities.jl")
@everywhere include("../src/misc/parallel_utilities.jl")
@everywhere include("../src/misc/declarations.jl")

@everywhere include("../src/quantile/derivative_helpers.jl")
@everywhere include("../src/quantile/Taylor_inverse_helpers.jl")
@everywhere include("../src/quantile/numerical_inverse.jl")

@everywhere include("../src/splines/quadratic_itp.jl")

@everywhere include("../src/KR/engine_Taylor.jl")
@everywhere include("../src/derivatives/RQ_Taylor_quantile.jl")
@everywhere include("../src/derivatives/RQ_derivatives.jl")
@everywhere include("../src/derivatives/traversal.jl")
@everywhere include("../src/derivatives/ROC_check.jl")

@everywhere include("../src/derivatives/RQ_sq_Taylor_quantile_multiwarp.jl")
@everywhere include("../src/derivatives/RQ_Taylor_quantile_multiwarp.jl")
#@everywhere include("../src/derivatives/RQ_Taylor_quantile.jl")

@everywhere include("../src/KR/engine.jl")
@everywhere include("../src/misc/normal_distribution.jl")
@everywhere include("../src/KR/engine_irregular_w2_sq.jl")
@everywhere include("../src/KR/Taylor_2x_warp_sq.jl")
@everywhere include("../src/KR/irregular_warpmaps.jl")
@everywhere include("../src/KR/engine_fit_density.jl")

@everywhere include("../src/DPP/DPP_helpers.jl")
@everywhere include("../src/DPP/inference_kDPP.jl")

@everywhere include("../src/warpmap/bandpass.jl")
@everywhere include("../src/warpmap/bspline2_derivatives.jl")

include("../src/diagnostics/moments.jl")
include("../src/diagnostics/visualize.jl")
include("../src/diagnostics/integral_probability_metrics.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

max_integral_evals_marginalization = 10000 #500 #typemax(Int)

## select dimension.
D = 2
N_array_factor = 0.3

f, x_ranges, limit_a, limit_b,
    fig_num = getk05helmetgrayscale(fig_num, 1, N_array_factor)

println("limit_a = ", limit_a)
println("limit_b = ", limit_b)
println()


N_realizations = 100

## generate dataset.
println("preparing X_nD")
@time X_nD = Utilities.ranges2collection(x_ranges, Val(D))
println()

# visualize full joint density.
if D == 2
    println("computing f(X)")
    @time f_X_nD = f.(X_nD)
    println()

    fig_num = myfunc(x_ranges, f_X_nD, [], ".", fig_num,
    #fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, f_X_nD, vec(X_nD), ".", fig_num,
                                            "f, full joint, D = 2")
end

###### fit KR, adaptive kernels with RQ as canonical kernel.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 50000
σ² = 1e-9 #1e-5
σ²_array = σ² .* ones(Float64, D)
max_integral_evals = 10000 #500 #typemax(Int)

#a_array = 0.7 .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
a_array = 0.7*2 .* ones(Float64, D)

# number of importance samples to draw from.
N_IS = 10000

# number of drawn importance samples.
N_𝑖 = 5000

# minimum distance between drawn importance samples.
ϵ = 1e-2

# a function of the total number of samples. Returns a cluster size.
avg_num_clusters_func = xx->max(round(Int,xx/20), 3)

N_iters_DPGMM = 1000
N_candidates = 1000

a_LP = 3.0 *2
a_HPc = 0.7 *2
multiplier_warpmap_BP = 180.0 # to do: a way to tune this.
multiplier_warpmap_LP = 1/30 * 300.0
multiplier_warpmap_HP  = 1/15 *300.0
ϵ_warpmap = 1e-7
#warp_weights = zeros(Float64, D)
warp_weights = ones(Float64, D) .* 0.3

N_kDPP_draws = 1
N_kDPP_per_draw = 500 #50
self_gain = 1.0
RQ_a_kDPP = 1.0
kDPP_warp_weights = ones(Float64, D)

N_𝑗 = 500 #50

w_config0 = IrregularWarpMapConfigType(N_IS, N_𝑖, ϵ, avg_num_clusters_func,
                N_iters_DPGMM, N_candidates, a_LP, a_HPc,
                multiplier_warpmap_LP, multiplier_warpmap_HP,
                multiplier_warpmap_BP, ϵ_warpmap, warp_weights,
                N_kDPP_draws, N_kDPP_per_draw, self_gain, RQ_a_kDPP,
                kDPP_warp_weights, N_𝑗, prune_tol)

# w_config_array = Vector{IrregularWarpMapConfigType{Float64}}(undef, D)
# w_config_array[1] = w_config0
# w_config_array[2] = w_config0

w_config_array = collect( w_config0 for d = 1:D )

# to do: there is a black band for some reason in the histogram...

println("Timing: fitdensity")
@time c_array, 𝓧_array, θ_array, d_ϕ_array, d2_ϕ_array,
        d_ψ_array, d2_ψ_array, X_array,
        f_oracle_array, f_d_multiplier_array, fq_array,
        c_LP_array, X_LP_array,
      c_HPc_array, X_HPc_array,
      c_f_proxy_array, X_f_proxy_array,
      a_f_proxy_array  = fitdensity(  f,
                                        max_integral_evals_marginalization,
                                        w_config_array,
                                        a_array,
                                        σ²_array)

println("End timing.")
println()



### store the fitted density.
# I am here. store and test reconstructed θ_array.
tag_string = "gmm3"

save_string = Printf.@sprintf("../output/fitted_density_dim_%d.jld", D)
#JLD.save(save_string, "c_array", c_array, "X_array", 𝓧_array,  "theta_array", θ_array)

# # debug the conditionals.
# include("check_fit.jl")
# @assert 1==2
#

## for publication.
# n_bins = 50
# PyPlot.figure(fig_num)
# fig_num += 1
# PyPlot.plt.hist(randn(5000), density=true, n_bins, [-2.5, 14])
# PyPlot.title("X~N(0,1)")
### end publication

## visualize.
if D == 2

    #
    f_oracle_X_nD = f_oracle_array[2].(X_nD)
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, f_oracle_X_nD, [], "x", fig_num,
                                            "Target Density")

    fq = xx->RKHSRegularization.evalquery(xx, c_array[2], 𝓧_array[2], θ_array[2])
    gq = xx->evalsq(xx, fq)
    X_nD = Utilities.ranges2collection(x_ranges, Val(D))

    gq_X_nD = gq.(X_nD)
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, gq_X_nD, [], "x", fig_num,
                                            "Fitted Density")
    #
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, gq_X_nD, 𝓧_array[end], "x", fig_num,
                                            "Fitted Density")

    fq = xx->RKHSRegularization.evalquery(xx, c_array[1], 𝓧_array[1], θ_array[1])
    gq = xx->evalsq(xx, fq)

    PyPlot.figure(fig_num)
    fig_num += 1

    xq = LinRange(limit_a[1], limit_b[1], 500)
    Xq = collect( [xq[n]] for n = 1:length(xq) )

    PyPlot.plot(Xq, gq.(Xq))
    PyPlot.plot(𝓧_array[1], gq.(𝓧_array[1]), "x")

    PyPlot.title("Fitted Density")
    PyPlot.legend()

    #PyPlot.savefig("destination_path.eps", format="eps")
end

@assert 1==2

# batch-related parameters.
N_viz = 6000
N_batches = 15 #100

# quantile-related parameters.
max_integral_evals = 10000
quantile_iters = 1000
N_nodes_tanh = 200
m_tsq = 7
quantile_err_tol = 0.01
max_traversals = 50
N_predictive_traversals = 40
correction_epoch = 10
quantile_max_iters = 500
quantile_convergence_zero_tol = 1e-4
n_limit = 10
n0 = 5

KR_config = KRTaylorConfigType( x_ranges,
                                quantile_iters,
                                max_integral_evals,
                                N_nodes_tanh,
                                m_tsq,
                                quantile_err_tol,
                                max_traversals,
                                N_predictive_traversals,
                                correction_epoch,
                                quantile_max_iters,
                                quantile_convergence_zero_tol,
                                n_limit,
                                n0 )

#
# parallelapplyadaptiveKRviaTaylorW2

@time x_array,
    discrepancy_array = parallelapplyadaptiveKRviaTaylorW2irregularsq( c_array,
                                        θ_array,
                                        𝓧_array,
                                        d_ϕ_array,
                                        d2_ϕ_array,
                                        d_ψ_array,
                                        d2_ψ_array,
                                        N_viz,
                                        N_batches,
                                        KR_config)
#

max_val, max_ind = findmax(norm.(discrepancy_array))
println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(discrepancy_array)))
println("largest l-1 discrepancy is ", max_val)
println("At that case: x = ", x_array[max_ind])
println()



#### visualize histogram.
if D == 2
    plot_flag = true
    n_bins = 500
    use_bounds = true
    bounds = [[limit_a[2], limit_b[2]], [limit_a[1], limit_b[1]]]

    if plot_flag
        PyPlot.figure(fig_num)
        fig_num += 1
        p1 = collect(x_array[n][2] for n = 1:N_viz)
        p2 = collect(x_array[n][1] for n = 1:N_viz)

        if use_bounds
            PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="Greys")
        else
            PyPlot.plt.hist2d(p1, p2, n_bins, cmap="Greys" )
        end
        #PyPlot.plt.axis("equal")
        #PyPlot.plt.colorbar()
        #PyPlot.title("x_array")



        PyPlot.figure(fig_num)
        fig_num += 1
        p1 = collect(randn() for n = 1:N_viz)
        p2 = collect(randn() for n = 1:N_viz)

        # if use_bounds
        #     PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="Greys")
        # else
            PyPlot.plt.hist2d(p1, p2, n_bins, cmap="Greys" )
        # end
        #PyPlot.plt.axis("equal")

    end
end
