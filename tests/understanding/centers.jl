using Distributed
using SharedArrays

#import JLD
import FileIO

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

import Printf
import GSL
import FiniteDiff


include("../src/approximation/approx_helpers.jl")
include("../src/approximation/final_approx_helpers.jl")
include("../src/approximation/analytic_cdf.jl")
include("../src/approximation/fit_mixtures.jl")
include("../src/approximation/optimization.jl")

include("../src/integration/numerical.jl")
include("../src/integration/adaptive_helpers.jl")
include("../src/integration/DEintegrator.jl")

include("../src/density/density_helpers.jl")
include("../src/density/fit_density.jl")

include("../src/misc/final_helpers.jl")
include("../src/misc/test_functions.jl")
include("../src/misc/utilities.jl")
include("../src/misc/declarations.jl")

include("../src/quantile/derivative_helpers.jl")
include("../src/quantile/Taylor_inverse_helpers.jl")
include("../src/quantile/numerical_inverse.jl")

include("../src/splines/quadratic_itp.jl")

include("../src/KR/engine_Taylor.jl")
include("../src/derivatives/RQ_Taylor_quantile.jl")
include("../src/derivatives/RQ_derivatives.jl")
include("../src/derivatives/traversal.jl")
include("../src/derivatives/ROC_check.jl")

include("../src/fresh/synthetic_data_generators.jl")
include("../src/fresh/engine.jl")
include("../src/fresh/utilities.jl")
include("../src/fresh/integration/numerical.jl")
include("../src/fresh/integration/numerical.jl")
include("../src/fresh/dG.jl")
include("../src/fresh/d2G.jl")
include("../src/fresh/single_KR.jl")
include("../src/fresh/kernel_centers/initial.jl")
include("../src/fresh/kernel_centers/subsequent.jl")
include("../src/fresh/fit.jl")
include("../src/fresh/misc/declarations.jl")
include("../src/fresh/kernel_centers/front_end.jl")

include("../src/KR/engine_irregular_w2.jl")
include("../src/DPP/DPP_helpers.jl")
include("../src/DPP/inference_kDPP.jl")


PyPlot.close("all")
fig_num = 1

Random.seed!(25)

# ## select dimension.
D = 2
N_array = [100; 100]
limit_a = [-14.0; -15.0]
limit_b = [16.0; 15.0]


## select other parameters for synthetic dataset.
N_components = 3
N_realizations = 100

## generate dataset.

Y_dist, f, x_ranges = setupsyntheticdataset(N_components,
                                N_array, limit_a, limit_b,
                                getnDGMMrealizations, Val(D))
#Y_dist, f, x_ranges, X_nD, t0, offset = lowdimsetup(N_components, N_array, 15.0, Val(D))

𝑋 = collect( rand(Y_dist, 1) for n = 1:N_realizations)
#

# prepare all marginal joint densities.
#f_joint = xx->evaljointpdf(xx,f,D)[1]
# f_joint = xx->evaljointpdf(xx,f,D,
#                 max_integral_evals = typemax(Int),
#                 initial_divisions = 3)[1] # higher accuracy.
f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

X_nD = Utilities.ranges2collection(x_ranges, Val(D))
Y_nD = f.(X_nD)


# query poitns are used for plotting in the script.
Nq_array = [200; 200]
xq_ranges = collect( LinRange(  limit_a[d],
                                limit_b[d],
                                Nq_array[d]) for d = 1:D )
Xq_nD = Utilities.ranges2collection(xq_ranges, Val(D))


###### fit KR, adaptive kernels with RQ as canonical kernel.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 10000
σ² = 1e-5
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 1.0 #50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5

N_preliminary_candidates = 10000
candidate_truncation_factor = 0.01 #0.005
candidate_zero_tol = 1e-12

base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
kDPP_zero_tol = 1e-12
N_kDPP_draws = 1
N_kDPP_per_draw = 50

close_radius_tol = 1e-6
N_refinements = 50 #20

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




####

canonical_a = 0.7

# # df2 experiment.
# canonical_a = 0.07
# σ² = 1e-2 # experiment with d2f warpmap.
# # end experiment.

θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

X_ref = vec(X_nD)

θ_a, c_history, X_history, error_history,
    X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                        Y_nD,
                        x_ranges,
                        X_ref,
                        f,
                        limit_a,
                        limit_b,
                        center_config)


####

f_Xq_nD = f.(Xq_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(xq_ranges,
                f_Xq_nD, X_pool, "x",
                fig_num, "initial pool", "x1", "x2")

# we need to pick minimum since the adaptive kernel
# does not guarantee monotonic decrease in error as
#   we increase data points, only asymtotically converge
#   as #  of data points approach inf.
# say we leave kernel center selection to future works.
min_error, min_ind = findmin(error_history)
X_star = X_history[min_ind]
c_star = c_history[min_ind]
g_star = xx->evalquery(xx, c_star, X_star, θ_a)

g_star_Xq_nD = g_star.(Xq_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(xq_ranges,
                f_Xq_nD, X_star, "x",
                fig_num, "g_star, markers at kernel centers", "x1", "x2")

#
fig_num = VisualizationTools.visualizemeshgridpcolor(xq_ranges,
                abs.(f_Xq_nD-g_star_Xq_nD), [], "x",
                fig_num, "abs(g_star-f)", "x1", "x2")
#
println("l-2 discrepancy of g_star vs. f is ",
            norm(f_Xq_nD-g_star_Xq_nD))
println()




PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(error_history, "x")

PyPlot.title("f vs. g. l-2 error history")
PyPlot.legend()
