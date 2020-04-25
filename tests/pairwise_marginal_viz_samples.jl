using Distributed
using SharedArrays

import Random
import PyPlot
import BSON

using LinearAlgebra

import Printf
import Utilities
import Distributions
import VisualizationTools

import SpecialFunctions

import HCubature

#import Plots

include("../src/misc/visualize.jl")
include("./example_helpers/test_densities.jl")


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





τ = 1e-2
N_array = [50; 50; 50; 50]
D = length(N_array)
# h, x_range = generaterandombetacopula( τ, N_array)
#
# x0 = rand(D)
# h(x0)

limit_a = ones(Float64, D) .* τ
limit_b = ones(Float64, D) .* (1-τ)
h, x_ranges, pdf_β, β_dists, R_β, w_β,
    pdf_γ, γ_dists, R_γ, w_γ = generaterandombetacopula( τ, N_array)




N_viz = 1000000
println("Timing: draw ", N_viz, "realizations from oracle distribution.")
@time Y = drawfromdemocopula(N_viz, β_dists, R_β, w_β, γ_dists, R_γ, w_γ )
println()

Y_components = separatecomponents(Y)

n_bins = 100
p1 = Y_components[1]
p2 = Y_components[2]
PyPlot.plt.hist2d(p1, p2, n_bins, cmap="jet" )

@assert 1==2

function plotpairwise(fig_num,
            Y_components, limit_a::Vector{T}, limit_b, i, j;
            n_bins = 50) where T

    Y_display = Vector{Vector{T}}(undef,2)
    Y_display[1] = Y_components[j]
    Y_display[2] = Y_components[i]

    limit_display_a = [limit_a[j]; limit_a[i]]
    limit_display_b = [limit_b[j]; limit_b[i]]


    fig_num = visualize2Dhistogram(fig_num, Y_display, limit_display_a, limit_display_b; n_bins = n_bins)

    return fig_num
end

i = 1
j = 2
fig_num = plotpairwise(fig_num, Y_components, limit_a, limit_b, i, j)

@assert 5==4

import StatsBase

using Plots; pyplot() # other backend OK if necessary
using StatsBase
using Distributions
using StatsFuns

# N = 100000
# z = rand(MvNormal([1.0, 0.0], [1.0 0.5; 0.5 0.5]), N);
# x = logistic.(z[1, :]);
# y = logistic.(z[2, :]);

x = Y_components[1]
y = Y_components[2]
h = fit(Histogram, (x, y), closed = :left, nbins = (50, 50));

gap = trunc(Int, maximum(h.weights) / 9)
gap = trunc(Int, maximum(h.weights) / 3)
levels = 0:gap:(10*gap)
#level_labels = map(x -> "$(x*10)%", 0:10)

## filled.
handle = contourf(midpoints(h.edges[1]),
         midpoints(h.edges[2]),
         h.weights';
         levels = levels)       # how to put level_labels in?
#


# ## not filled.
# handle = contour(midpoints(h.edges[1]),
#          midpoints(h.edges[2]),
#          h.weights';
#          levels = levels)       # how to put level_labels in?

# figure this out again. https://discourse.julialang.org/t/labels-for-levels-in-contour-plot-plots-jl/8266
display(handle)

@assert 1==2

output_folder_name = "/home/roy/MEGAsync/outputs/debug"
dummy_fig = Plots.plot()
plotallpairwisecontours(Y_components, dummy_fig, 50, 50,
                        output_folder_name)

@assert 1==2
