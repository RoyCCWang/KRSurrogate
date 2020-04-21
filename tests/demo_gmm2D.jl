# packaged up version of fast phi.


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


#@everywhere import stickyHDPHMM

@everywhere import Printf
@everywhere import GSL
@everywhere import FiniteDiff

@everywhere include("../tests/example_helpers/test_densities.jl")
@everywhere include("../src/misc/declarations.jl")

@everywhere include("../src/KR/engine.jl")
@everywhere include("../src/misc/normal_distribution.jl")
@everywhere include("../src/misc/utilities.jl")
@everywhere include("../src/integration/numerical.jl")
@everywhere include("../src/KR/dG.jl")
@everywhere include("../src/KR/d2G.jl")
@everywhere include("../src/KR/single_KR.jl")
@everywhere include("../src/KR/parallel_KR.jl")
@everywhere include("../tests/verification/differential_verification.jl")
@everywhere include("../src/kernel_centers/initial.jl")
@everywhere include("../src/kernel_centers/subsequent.jl")
@everywhere include("../src/kernel_centers/kDPP.jl")
@everywhere include("../src/fit/fit_adaptive_bp.jl")
include("../src/fit/RKHS.jl")
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

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

# ## select dimension.
D = 2
N_array = [100; 100]
limit_a = [-14.0; -15.0]
limit_b = [16.0; 15.0]

# D = 3
# #N_array = [100; 100; 100]
# #N_array = [50; 50; 50]
# N_array = [20; 20; 20]
# limit_a = [-14.0; -16.0; -11.0]
# limit_b = [16.0; 15.0; 5.0]

## select other parameters for synthetic dataset.
N_components = 3
N_realizations = 100

## generate dataset.

Y_dist, f, x_ranges = setupsyntheticdataset(N_components,
                                N_array, limit_a, limit_b,
                                getnDGMMrealizations, Val(D))
#
X_nD = Utilities.ranges2collection(x_ranges, Val(D))

#
f_X_nD = f.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, f_X_nD, [], ".", fig_num,
                            "f, unnormalized density")

### get fit locations.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 10000
σ_array = sqrt(1e-5) .* ones(Float64, D)
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 30.0 #50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5

N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
candidate_truncation_factor = 0.01 #0.005
candidate_zero_tol = 1e-12

base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
kDPP_zero_tol = 1e-12
N_kDPP_draws = 1
N_kDPP_per_draw = 50

close_radius_tol = 1e-4#0.3 #1e-3 #0.3
N_refinements = 50 #20

initial_divisions = 1

center_config = KernelCenterConfigType( amplification_factor,
                attenuation_factor_at_cut_off,
                N_bands,
                N_preliminary_candidates,
                candidate_truncation_factor,
                candidate_zero_tol,
                base_gain,
                kDPP_zero_tol,
                N_kDPP_draws,
                N_kDPP_per_draw,
                zero_tol_RKHS,
                prune_tol,
                close_radius_tol,
                N_refinements,
                max_iters_RKHS,
                #σ²,
                σ_array[end]^2,
                initial_divisions)

#
canonical_a = 0.7
θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)


X_ref = vec(X_nD)

Y_nD = f.(X_nD)

θ_a, c_history, X_history, error_history,
    X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                        Y_nD,
                        x_ranges,
                        X_ref,
                        f,
                        limit_a,
                        limit_b,
                        center_config)
#
min_error, min_ind = findmin(error_history)
𝑋 = X_history[min_ind]

#𝑋 = collect( vec(rand(Y_dist, 1)) for n = 1:N_realizations)
#

# prepare all marginal joint densities.
#f_joint = xx->evaljointpdf(xx,f,D)[1]
# f_joint = xx->evaljointpdf(xx,f,D,
#                 max_integral_evals = typemax(Int),
#                 initial_divisions = 3)[1] # higher accuracy.
f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

###### fit KR, adaptive kernels with RQ as canonical kernel.


a_array = 0.7 .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.

X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
removeclosepositionsarray!(X_array, close_radius_tol)
# to do: auto fuse points with self-adapting close_radius_tol in fit.jl
#       this is to avoid semidef error for the QP solver.

f_X_array = collect( f_joint.(X_array[d]) for d = 1:D )


println("Timing: fitproxydensities")
@time c_array, 𝓧_array, θ_array,
        dφ_array, d2φ_array,
        dϕ_array, d2ϕ_array,
        Y_array = fitmarginalsadaptive(f_X_array, X_array,
                                    x_ranges, f,
                                    max_iters_RKHS, a_array, σ_array,
                                    amplification_factor, N_bands,
                                    attenuation_factor_at_cut_off,
                                    zero_tol_RKHS, prune_tol, max_integral_evals)

println("Number of kernel centers kept, per dim:")
println(collect( length(c_array[d]) for d = 1:D))

gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                        max_integral_evals = max_integral_evals)

#
# visualize.
g2 = gq_array[2]
g2_X_nD = g2.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, g2_X_nD, 𝓧_array[2], "x", fig_num,
                            "g2, markers at kernel centers")



PyPlot.figure(fig_num)
fig_num += 1

xq = LinRange(limit_a[1], limit_b[1], 500)
Xq = collect( [xq[n]] for n = 1:length(xq) )

fq_Xq = gq_array[1].(Xq)
fq_𝓧 = gq_array[1].(𝓧_array[1])
#f_Xq = f_joint.(Xq)

PyPlot.plot(xq, fq_Xq, label = "fq")
PyPlot.plot(𝓧_array[1], fq_𝓧, "x", label = "fq kernel centers")
#PyPlot.plot(xq, f_Xq, "--", label = "f")
PyPlot.plot(x_ranges[1], Y_array[1], "^", label = "Y")

PyPlot.title("f vs. fq")
PyPlot.legend()

# try visualizing with histogram, if oracle density is easy to sample from.
# Y_array too low of a resolution.
#


# batch-related parameters.
#N_viz = 10000
N_viz = 1000
#N_viz = 1000 # much be larger than N_batch.
N_batches = 15 #100

# quantile-related parameters.
max_integral_evals = 10000
quantile_iters = 1000
N_nodes_tanh = 200
m_tsq = 7
quantile_err_tol = 1e-3 #0.01  both gave l-2 norm( abs(u-u_rec) ), summed over all dimensions: 0.024788099072235982
max_traversals = 50
N_predictive_traversals = 40
correction_epoch = 10
quantile_max_iters = 500
quantile_convergence_zero_tol = 1e-6 #1e-4
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



Printf.@printf("Timing for parallel application of KR, %d particles.\n", N_viz)
@time x_array, discrepancy_array = parallelevalKR( c_array,
                                            θ_array,
                                            𝓧_array,
                                            dφ_array,
                                            d2φ_array,
                                            N_viz,
                                            N_batches,
                                            KR_config)
#

#

max_val, max_ind = findmax(norm.(discrepancy_array))
println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(discrepancy_array)))
println("largest l-1 discrepancy is ", max_val)
println("At that case: x = ", x_array[max_ind])
println()



#### visualize histogram.

plot_flag = true
n_bins = 500
use_bounds = false
bounds = [[limit_a[2], limit_b[2]], [limit_a[1], limit_b[1]]]

if plot_flag
        PyPlot.figure(fig_num)
        fig_num += 1
        p1 = collect(x_array[n][2] for n = 1:N_viz)
        p2 = collect(x_array[n][1] for n = 1:N_viz)

        if use_bounds
            PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="jet")
        else
            PyPlot.plt.hist2d(p1, p2, n_bins, cmap="jet" )
        end


        # PyPlot.figure(fig_num)
        # fig_num += 1
        # p1 = collect(randn() for n = 1:N_viz)
        # p2 = collect(randn() for n = 1:N_viz)
        #
        # if use_bounds
        #     PyPlot.plt.hist2d(p1, p2, n_bins, range = bounds, cmap="Greys")
        # else
        #     PyPlot.plt.hist2d(p1, p2, n_bins, cmap="Greys" )
        # end
        # #PyPlot.plt.axis("equal")

end
