@everywhere using Distributed
@everywhere using SharedArrays


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

# 
# @everywhere import stickyHDPHMM

@everywhere import Printf
@everywhere import GSL



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

@everywhere include("../src/KR/engine.jl")
@everywhere include("../src/misc/normal_distribution.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


## select dimension.
D = 2
N_array_factor = 0.3

f, x_ranges, limit_a, limit_b,
    fig_num = getk05helmetgrayscale(fig_num, 1, N_array_factor)

println("limit_a = ", limit_a)
println("limit_b = ", limit_b)
println()




## select other parameters for synthetic dataset.
N_components = 3
N_full = 1000
N_realizations = 100

## generate dataset.
𝑋 =
#𝑋 = collect( rand(Y_dist, 1) for n = 1:N_realizations)
#

# prepare all marginal joint densities.
f_joint = xx->evaljointpdf(xx,f,D)[1]

# visualize full joint density.

    X_nD = Utilities.ranges2collection(x_ranges, Val(D))

    f_X_nD = f_joint.(X_nD)
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, f_X_nD, vec(X_nD), ".", fig_num,
                                            "oracle density, full joint, D = 2")


###### fit KR, adaptive kernels with RQ as canonical kernel.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 5000
σ² = 1e-5
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5

a_array = 0.7 .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
N_X[1] = 25

X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
f_X_array = collect( f_joint.(X_array[d]) for d = 1:D )






# # warp map parameter set up.
# reciprocal_cut_off_percentages = ones(N_bands) ./collect(LinRange(1.0,0.2,N_bands))
# ω_set = collect( π/(reciprocal_cut_off_percentages[i]*sqrt(2*log(attenuation_factor_at_cut_off))) for i = 1:length(reciprocal_cut_off_percentages) )
# pass_band_factor = abs(ω_set[1]-ω_set[2])*0.2
#
# # construct warp map.
# X_nD = Utilities.ranges2collection(x_ranges[1:d_select], Val(d_select))
# Y_nD = f.(X_nD)
#
# ϕ = RKHSRegularization.getRieszwarpmapsamples(Y_nD, Val(:simple), Val(:uniform), ω_set, pass_band_factor)
# ϕ_map_func, d_ϕ_map_func, d2_ϕ_map_func = getwarpmap(ϕ, x_ranges[1:d_select], amplification_factor)
#
# # make adaptive kernel.
# θ_canonical = RKHSRegularization.RationalQuadraticKernelType(a)
# θ_a = RKHSRegularization.AdaptiveKernelType(θ_canonical, ϕ_map_func)
# K = constructkernelmatrix(X, θ_a)
#
# @assert 1==2

println("Timing: fitadaptiveKRviaTaylorW2")
@time c_array, 𝓧_array, θ_array, d_ϕ_array, d2_ϕ_array = fitadaptiveKRviaTaylorW2(f_X_array, X_array,
                                            x_ranges, f_joint,
                                            max_iters_RKHS, a_array, σ²,
                                            amplification_factor, N_bands,
                                            attenuation_factor_at_cut_off,
                                            zero_tol_RKHS, prune_tol, max_integral_evals)

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

@time x_array, discrepancy_array = parallelapplyadaptiveKRviaTaylorW2(   c_array,
                                        θ_array,
                                        𝓧_array,
                                        d_ϕ_array,
                                        d2_ϕ_array,
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
    #bounds = [[-15, 15], [-15, 15]]
    bounds = [[limit_a[1], limit_b[1]], [limit_a[2], limit_b[2]]]
    bounds = [[limit_a[2], limit_b[2]], [limit_a[1], limit_b[1]]]

    if plot_flag
        PyPlot.figure(fig_num)
        fig_num += 1
        p1 = collect(x_array[n][2] for n = 1:N_viz)
        p2 = collect(x_array[n][1] for n = 1:N_viz)
        # p1 = collect(x_array[n][1] for n = 1:N_viz)
        # p2 = collect(x_array[n][2] for n = 1:N_viz)

        if use_bounds
            PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="Greys" )
        else
            PyPlot.plt.hist2d(p1, p2, n_bins, cmap="Greys" )
        end
        PyPlot.plt.axis("equal")
        PyPlot.title("x_array")

        # if savefig_flag
        #     save_name = Printf.@sprintf("./outputs/%d-histogram.png",n)
        #     PyPlot.savefig(save_name)
        #     sleep(save_delay)
        #     PyPlot.close(fig_num-1)
        #     fig_num -= 1
        # end
    end
end

## visualize.
if D == 2

    #
    X_nD = Utilities.ranges2collection(x_ranges, Val(D))

    f_X_nD = f.(X_nD)
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, f_oracle_X_nD, [], "x", fig_num,
                                            "Target Density")

    fq = xx->RKHSRegularization.evalquery(xx, c_array[2], 𝓧_array[2], θ_array[2])

    fq_X_nD = fq.(X_nD)
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, fq_X_nD, [], "x", fig_num,
                                            "Fitted Density")
    #
    fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, fq_X_nD, 𝓧_array[end], "x", fig_num,
                                            "Fitted Density")
end
