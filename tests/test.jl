# packaged up version of fast phi.


using Distributed
using SharedArrays

#import JLD
import FileIO

import NearestNeighbors

import Printf
import PyPlot
import Random
import Optim

using LinearAlgebra

using FFTW

import Statistics

import Distributions
import HCubature
import Interpolations
import SpecialFunctions

import SignalTools
import RKHSRegularization
import Utilities

import Convex
import SCS

import Calculus
import ForwardDiff


#import stickyHDPHMM

import FiniteDiff

import Printf
import GSL
import FiniteDiff

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
include("../src/fit/fit_adaptive_bp.jl")
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

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

#f = tt->sin(2*tt[1])*exp(-0.2*tt[1]^2)
f = xx->sinc(4*xx[1])*xx[1]^3

t_range = LinRange(-5, 3, 100)

θ = RKHSRegularization.RationalQuadraticKernelType(0.7)
σ² = 1e-5

N_realizations = 15
X = collect( randn(1) for n = 1:N_realizations )
f_X = f.(X)
η, α = RKHSRegularization.fitnDdensity(vec(f_X), vec(X), σ², θ,
                                            zero_tol_RKHS, max_iters_RKHS)

fq = xx->RKHSRegularization.evalquery(xx, η.c, η.X, η.θ)

#
PyPlot.figure(fig_num)
fig_num += 1

xq = LinRange(t_range[1], t_range[end], 500)
Xq = collect( [xq[n]] for n = 1:length(xq) )



f_Xq = f.(Xq)
fq_Xq = fq.(Xq)

PyPlot.plot(xq, fq_Xq, label = "fq")
PyPlot.plot(xq, fq_Xq, "x")

PyPlot.plot(xq, f_Xq, label = "f")

PyPlot.title("f vs. fq")
PyPlot.legend()

@assert 1==222

## select dimension.
D = 2
N_array = [100; 100]

τ = 1e-2
limit_a = [τ; τ]
limit_b = [1-τ; 1-τ]



# oracle probability density.
𝑝, x_ranges = getmixture2Dbetacopula1(τ, N_array)

X_nD = Utilities.ranges2collection(x_ranges, Val(D))



# unnormalized probability density.
𝑝_X_nD = 𝑝.(X_nD)
f_scale_factor = maximum(𝑝_X_nD)
f = xx->𝑝(xx)/f_scale_factor

# I am here. next, add 2D histogram.
f_X_nD = f.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, f_X_nD, [], ".", fig_num,
                            "f, unnormalized density")

### get fit locations.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 10000
σ_array = sqrt(1e-3) .* ones(Float64, D)
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 5.0 #1.0 #50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5

N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
candidate_truncation_factor = 0.01 #0.005
candidate_zero_tol = 1e-12

base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
kDPP_zero_tol = 1e-12
N_kDPP_draws = 1
N_kDPP_per_draw = 50

close_radius_tol = 1e-5
N_refinements = 10 #50

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



fitRKHSRegularizationdensity(  f_X_array[end],
                                X_array[end],
                                x_ranges,
                                Y_nD,
                                max_iters_RKHS::Int,
                                a::T,
                                σ²::T,
                                amplification_factor::T,
                                N_bands::Int,
                                attenuation_factor_at_cut_off::T,
                                zero_tol_RKHS::T,
                                prune_tol::T)


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
g2_X_nD = g2.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges, g2_X_nD, 𝓧_array[2], "x", fig_num,
                            "g2, markers at kernel centers")

###
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
