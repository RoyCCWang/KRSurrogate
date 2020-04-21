
module KRTransportMap

using Distributed
using SharedArrays

import JLD
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


import stickyHDPHMM

import GSL

import NNLS

include("./approximation/approx_helpers.jl")
include("./approximation/final_approx_helpers.jl")
include("./approximation/analytic_cdf.jl")
include("./approximation/fit_mixtures.jl")
include("./approximation/optimization.jl")

include("./integration/numerical.jl")
include("./integration/adaptive_helpers.jl")
include("./integration/DEintegrator.jl")

include("./density/density_helpers.jl")
include("./density/fit_density.jl")

include("./misc/final_helpers.jl")
include("./misc/test_functions.jl")
include("./misc/utilities.jl")
include("./misc/parallel_utilities.jl")
include("./misc/declarations.jl")

include("./quantile/derivative_helpers.jl")
include("./quantile/Taylor_inverse_helpers.jl")
include("./quantile/numerical_inverse.jl")

include("./splines/quadratic_itp.jl")

include("./KR/engine_Taylor.jl")
include("./derivatives/RQ_Taylor_quantile.jl")
include("./derivatives/RQ_derivatives.jl")
include("./derivatives/traversal.jl")
include("./derivatives/ROC_check.jl")

include("./derivatives/RQ_sq_Taylor_quantile_multiwarp.jl")
include("./derivatives/RQ_Taylor_quantile_multiwarp.jl")
#include("./derivatives/RQ_Taylor_quantile.jl")

include("./KR/engine.jl")
include("./KR/engine_irregular_w2_sq.jl")
include("./KR/Taylor_2x_warp_sq.jl")
include("./KR/irregular_warpmaps.jl")
include("./KR/engine_fit_density.jl")

include("./DPP/DPP_helpers.jl")
include("./DPP/inference_kDPP.jl")

include("./warpmap/bandpass.jl")
include("./warpmap/bspline2_derivatives.jl")

include("../src/diagnostics/moments.jl")
include("../src/diagnostics/visualize.jl")
include("../src/diagnostics/integral_probability_metrics.jl")

include("../src/misc/quantile_frontend.jl")
include("../src/misc/fit_density_frontend.jl")

export fitRKHSRegularizationdensity1D,
        fitRKHSRegularizationunnormalizeddensity,
        fitsqrtdensity,
        selectkernelcenters,
        getwarpmapsqrtspline2,
        getcandidatepnodes,
        evalquerySqExp2xwarpsq,
        fitRKHSRegularizationunnormalizeddensityconfig,
        getadaptivekernel,
        getwarpmapspline2




end
