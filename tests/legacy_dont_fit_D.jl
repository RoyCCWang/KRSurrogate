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
@everywhere include("../src/KR/unbundled/adaptive/KR_isonormal.jl")
@everywhere include("../src/misc/chain_rule.jl")
@everywhere include("../src/misc/parallel_utilities.jl")

@everywhere include("../tests/verification/fit_example_copulae.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

# ## select dimension.
# D = 2
# N_array = [100; 100]
# limit_a = [-14.0; -15.0]
# limit_b = [16.0; 15.0]

D = 3
N_array = [100; 100; 100]
#N_array = [50; 50; 50]
#N_array = [20; 20; 20]
limit_a = [-14.0; -16.0; -11.0]
limit_b = [16.0; 15.0; 5.0]

## select other parameters for synthetic dataset.
N_components = 3
N_realizations = 40

## generate dataset.

Y_dist, f_syn, x_ranges = setupsyntheticdataset(N_components,
                                N_array, limit_a, limit_b,
                                getnDGMMrealizations, Val(D))
#Y_dist, f, x_ranges, X_nD, t0, offset = lowdimsetup(N_components, N_array, 15.0, Val(D))

𝑋 = collect( rand(Y_dist, 1) for n = 1:N_realizations)

X_nD = Utilities.ranges2collection(x_ranges, Val(D))

# I gotta use a different density with heavier tails.
## make sure f(x) is numerically above zero..
#f = xx->(f_syn(xx)+1e-6)

μ_cauchy = randn(D).*3
σ_cauchy = rand(D) .+ 2.5
cauchy_dists = collect( Distributions.Cauchy(μ_cauchy[d], σ_cauchy[d]) for d = 1:D )
cauchy_pdf = xx->exp(sum( Distributions.logpdf(cauchy_dists[d], xx[d]) for d = 1:D))
f = xx->(f_syn(xx)+cauchy_pdf(xx))
#f = cauchy_pdf
#f = f_syn

#f_X_nD = f.(X_nD)
#f2_X_nD = f2.(X_nD)
#all(isfinite.(norm.(vec(f_X_nD))))
#all(isfinite.(norm.(vec(f2_X_nD))))

f_ratio_tol = 1e-8
status_flag, min_f_x, max_f_x = dynamicrangecheck(f, x_ranges,
                                        f_ratio_tol)
@assert status_flag # to do: handle this failure gracefully.


# prepare all marginal joint densities.
f_joint = xx->evaljointpdf(xx,f,D)[1]

# visualize.
Nv = 100
xv_ranges = collect( LinRange(limit_a[d],limit_b[d],Nv) for d = 1:2 )
Xv_nD = Utilities.ranges2collection(xv_ranges, Val(2))

println("Timing, f_joint.(Xv_nD)")
#@time f_Xv_nD = f_joint.(Xv_nD)
@time f_Xv_nD = reshape(parallelevals(f_joint, vec(Xv_nD)), size(Xv_nD))
println("End timing.")
fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
                        f_Xv_nD, [], "x", fig_num,
                            "f, numerical integration")

#@assert 1==2


###### fit KR, adaptive kernels with RQ as canonical kernel.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 5000
σ_array = sqrt(1e-5) .* ones(Float64, D-1)
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 1.0 #50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5
a_array = 0.1 .* ones(Float64, D-1)

#a_array = 0.7 .* ones(Float64, D-1)

N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
N_X[1] = 25

X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
f_X_array = collect( f_joint.(X_array[d]) for d = 1:D-1 )

println("Timing: fitproxydensities")
@time c_array, 𝓧_array, θ_array,
        dφ_array, d2φ_array,
        dϕ_array, d2ϕ_array,
        Y_array = fitmarginalsadaptive(f_X_array, X_array,
                                    x_ranges[1:D-1], f_joint,
                                    max_iters_RKHS, a_array, σ_array,
                                    amplification_factor, N_bands,
                                    attenuation_factor_at_cut_off,
                                    zero_tol_RKHS, prune_tol,
                                    max_integral_evals)

println("Number of kernel centers kept, per dim:")
println(collect( length(c_array[d]) for d = 1:D-1))

gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                        max_integral_evals = max_integral_evals)


# visualize.

d_select = 2

# visualization positions.


g2 = gq_array[d_select]
g2_Xv_nD = g2.(Xv_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
                        g2_Xv_nD, 𝓧_array[d_select], "x", fig_num,
                            "g2, markers at kernel centers")
#


###

PyPlot.figure(fig_num)
fig_num += 1

Nv_1D = 300
xq = LinRange(limit_a[1], limit_b[1], Nv_1D)
Xq = collect( [xq[n]] for n = 1:length(xq) )

fq_Xq = gq_array[1].(Xq)
fq_𝓧 = gq_array[1].(𝓧_array[1])

# this is slow.
println("Timing, f(Xq)")
#f_Xq = f_joint.(Xq)
@time f_Xq = reshape(parallelevals(f_joint, vec(Xq)), size(Xq))

PyPlot.plot(xq, fq_Xq, label = "fq")
PyPlot.plot(𝓧_array[1], fq_𝓧, "x", label = "fq kernel centers")
PyPlot.plot(xq, f_Xq, "--", label = "f")
PyPlot.plot(x_ranges[1], Y_array[1], "^", label = "Y")

PyPlot.title("f vs. fq")
PyPlot.legend()



#### transport.
f_target = f

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
