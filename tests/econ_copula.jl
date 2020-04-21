# packaged up version of fast phi.
# try to reduce the number of kernel centers as much as we can.

using Distributed
@everywhere using SharedArrays

#@everywhere import JLD
@everywhere import FileIO

import NearestNeighbors

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

## select dimension.
D = 2
#N_array = [400; 400]
#N_array = [40; 40] # itp still good...
#N_array = [20; 20]
N_array = [16; 16]

τ = 1e-2
limit_a = [τ; τ]
limit_b = [1-τ; 1-τ]



# oracle probability density.
𝑝, x_ranges = getmixture2Dbetacopula1(τ, N_array)

X_nD = Utilities.ranges2collection(x_ranges, Val(D))

# visualization positions.
Nv = 100
xv_ranges = collect( LinRange(limit_a[d],limit_b[d],Nv) for d = 1:D )
Xv_nD = Utilities.ranges2collection(xv_ranges, Val(D))

# unnormalized probability density.
𝑝_X_nD = 𝑝.(X_nD)
f_scale_factor = maximum(𝑝_X_nD)
f = xx->𝑝(xx)/f_scale_factor

# compare against itp.
f_X_nD = f.(X_nD)

f_itp, _, _ = Utilities.setupcubicitp(f_X_nD, x_ranges,
                    1.0)

# to do: can't saw grid-warpmap + postive constrained RKHS is an improvement.
#       so, do irregular warpmap + "" vs. itp.
# to do: comparison of positive constrained RKHS to NNLS.
# ψ = evalBspline2
# warpmap_LP, c_LP = fitregressionNNLSnoprune(f_𝓧, 𝓧, ψ, a_LP)

fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges, f_itp.(Xv_nD), [], ".", fig_num,
                            "f_itp, unnormalized density")

f_Xv_nD = f.(Xv_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges, f_Xv_nD, [], ".", fig_num,
                            "f, unnormalized density")
#@assert 1==2
### get fit locations.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 10000
#σ_array = sqrt(1e-3) .* ones(Float64, D)
σ_array = [sqrt(1e-8); sqrt(1e-3)]
max_integral_evals = 10000 #500 #typemax(Int)
#amplification_factor = 5.0 # good for N > 20.
amplification_factor = 5.0 # good for N > 20.
attenuation_factor_at_cut_off = 2.0
N_bands = 5

N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
candidate_truncation_factor = 0.01 #0.005
candidate_zero_tol = 1e-12

base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
kDPP_zero_tol = 1e-12
N_kDPP_draws = 1
N_kDPP_per_draw = 20

close_radius_tol = 1e-5
N_refinements = 10 #10
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
# larger means larger bandwidth.
# canonical_a = 0.03 # for N > 20.
canonical_a = 0.03

θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

X_nD = Utilities.ranges2collection(x_ranges, Val(D))
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


a_array = canonical_a .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
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


# visualize.
g2 = gq_array[2]
g2_Xv_nD = g2.(Xv_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges, g2_Xv_nD, 𝓧_array[2], "x", fig_num,
                            "g2, markers at kernel centers")


PyPlot.figure(fig_num)
fig_num += 1

xq = LinRange(limit_a[1], limit_b[1], Nv)
Xq = collect( [xq[n]] for n = 1:length(xq) )

fq_Xq = gq_array[1].(Xq)
fq_𝓧 = gq_array[1].(𝓧_array[1])
f_Xq = f_joint.(Xq)

PyPlot.plot(xq, fq_Xq, label = "fq")
PyPlot.plot(𝓧_array[1], fq_𝓧, "x", label = "fq kernel centers")
PyPlot.plot(xq, f_Xq, "--", label = "f")
PyPlot.plot(x_ranges[1], Y_array[1], "^", label = "Y")

PyPlot.title("f vs. fq")
PyPlot.legend()

@assert 1==2

# try visualizing with histogram, if oracle density is easy to sample from.
# Y_array too low of a resolution.
#


# batch-related parameters.
N_viz = 10000
#N_viz = 100 # much be larger than N_batch.
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
