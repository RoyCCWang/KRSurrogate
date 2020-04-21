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

@everywhere include("../tests/verification/fit_example_copulae.jl")

@everywhere include("../tests/verification/helpers.jl")
@everywhere include("../tests/example_helpers/demo_densities.jl")
@everywhere include("../tests/verification/adaptive/demo_cases.jl")

@everywhere include("../tests/verification/fit_example_gmm_cauchy.jl")
@everywhere include("../tests/verification/transport_derivative_helpers.jl")

# for non-adaptive.
@everywhere include("../src/fit/fit_non_adaptive.jl")

include("../src/KR/bundled/non_adaptive_isonormal.jl")
include("../src/kernels/RQ_non_adaptive.jl")
include("../src/KR/setupallKR_non_adaptive.jl")
include("../src/KR/transport_non_adaptive.jl")
include("../src/KR/misc_non_adaptive.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)



#skip_flag = true # bad d2T.
skip_flag = false # good d2T.

c_array, 𝓧_array, θ_array, f, limit_a, limit_b, x_ranges,
          src_μ, src_σ_array,
          f_joint = democgmm3Dnonadaptive(; skip_flag = skip_flag,
                                                N_realizations = 100)


#
# visualize.
fig_num = visualizefit2Dmarginal(fig_num, c_array, 𝓧_array, θ_array,
                            limit_a, limit_b, f_joint)

###
Random.seed!(205)


src_dist, df_target, d2f_target,
    KR_config = setupquantilefordemo(f, x_ranges, limit_a, limit_b,
                                        src_μ, src_σ_array)

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

#@assert 1==2
N_visualization = 1000
x_array, discrepancy_array, _ = runbatchtransport( src_dist,
                                                T_map,
                                                ds_T_map, dT;
                                                N_visualization = N_visualization)

#### visualize histogram.
fig_num = visualize2Dhistogram(fig_num, x_array, limit_a, limit_b)
