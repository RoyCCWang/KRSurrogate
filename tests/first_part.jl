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



import Printf
import GSL
import FiniteDiff

using BenchmarkTools

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
include("../src/fit/fit_adaptive.jl")
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

include("../src/KR/transport.jl")
include("../src/KR/unbundled/adaptive/KR_isonormal.jl")
include("../src/misc/chain_rule.jl")

include("../tests/verification/fit_example_copulae.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

σ = rand()
μ = -abs(randn())

### Isotropic normal PDF.
f = xx->exp(-0.5*((xx-μ)/σ)^2)/(σ*sqrt(2*π))

φ = xx->exp(-0.5*((xx-μ)/σ)^2)/(σ*sqrt(2*π))


df_AD = xx->ForwardDiff.derivative(f,xx)
df_AN =  xx->(-(xx-μ)/(sqrt(2*π)*σ^3))*exp(-0.5*(xx-μ)^2/σ^2)

x0 = randn()

df_x0_AD = df_AD(x0)
df_x0_AN = df_AN(x0)

println("AD: df(x0) = ", df_x0_AD)
println("AN: df(x0) = ", df_x0_AN)
println("discrepancy: ", norm(df_x0_AD-df_x0_AN))
println()


### test transport map.

D = 2
# N_array = [100; 100]
# c_array, 𝓧_array, θ_array,
#       dφ_array, d2φ_array,
#       dϕ_array, d2ϕ_array,
#       Y_array, KR_config,
#       limit_a, limit_b,
#       f_target = fitexamplebetacopuladensity( Val(D), N_array )
#

N_array = [40; 40]

c_array, 𝓧_array, θ_array,
      dφ_array, d2φ_array,
      dϕ_array, d2ϕ_array,
      Y_array, KR_config,
      limit_a, limit_b,
      f_target = fitexamplebetacopuladensityecon1( Val(D), N_array )
#

### visualize fitted density.
initial_divisions = 1
gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                      max_integral_evals = KR_config.max_integral_evals,
                      initial_divisions = initial_divisions)
#
# visualization positions.
Nv = 100
xv_ranges = collect( LinRange(limit_a[d],limit_b[d],Nv) for d = 1:D )
Xv_nD = Utilities.ranges2collection(xv_ranges, Val(D))

f_Xv_nD = gq_array[end].(Xv_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges, f_Xv_nD, [], ".", fig_num,
                            "fitted density g[end]")

println()
#
