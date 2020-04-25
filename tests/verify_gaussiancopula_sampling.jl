import Random
import PyPlot
import BSON

using LinearAlgebra

import Printf
import Utilities
import Distributions
import VisualizationTools

import SpecialFunctions

PyPlot.close("all")
fig_num = 1

Random.seed!(25)


# data = BSON.load("debug.bson")
# L = data[:L_full]
#
# println("typeof(L) = ", typeof(L))
# println()
#
# isposdef(L) # takes a while if in @everywhere?
# eigen(L) # takes a while

include("../src/misc/visualize.jl")
include("./example_helpers/test_densities.jl")


τ = 1e-2
N_array = [100; 100]
D = length(N_array)
# h, x_range = generaterandombetacopula( τ, N_array)
#
# x0 = rand(D)
# h(x0)


limit_a = [τ; τ]
limit_b = [1-τ; 1-τ]
h, x_ranges, pdf_β, β_dists, R_β, w_β,
    pdf_γ, γ_dists, R_γ, w_γ = generaterandombetacopula( τ, N_array)

Nv_array = [400; 400]
xv_ranges = collect( LinRange(limit_a[d], limit_b[d], Nv_array[d]) for d = 1:D )
Xv_nD = Utilities.ranges2collection(xv_ranges, Val(D))
h_Xv_nD = h.(Xv_nD)

fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
                  h_Xv_nD, [], "x", fig_num, "h")

N_viz = 100000
Y = drawfromdemocopula(N_viz, β_dists, R_β, w_β, γ_dists, R_γ, w_γ )

n_bins = 100
fig_num = visualize2Dhistogram(fig_num, Y, limit_a, limit_b;
                                use_bounds = true, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "Y, bounds")
