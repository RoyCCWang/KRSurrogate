using Distributed
using SharedArrays

@everywhere import Random
@everywhere import PyPlot
@everywhere import BSON

@everywhere using LinearAlgebra

@everywhere import Printf
@everywhere import Utilities
@everywhere import Distributions
@everywhere import VisualizationTools

@everywhere import SpecialFunctions

@everywhere import HCubature


@everywhere include("../src/misc/visualize.jl")
@everywhere include("./example_helpers/test_densities.jl")
@everywhere include("../src/misc/parallel_utilities.jl")


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


#
d_i = 2
d_j = 3
z_min = copy(limit_a)
deleteat!(z_min, sort([d_i; d_j]))

z_max = copy(limit_b)
deleteat!(z_max, sort([d_i; d_j]))

println("Timing: evalgridpairwisemarginal")
initial_divisions = 1
max_integral_evals = 10000
@time h_X = evalgridpairwisemarginal(h, x_ranges, d_i, d_j, z_min, z_max;
                            initial_divisions = initial_divisions,
                            max_integral_evals = max_integral_evals)
println("end timing.")
#

# Nv_array = [400; 400]
# xv_ranges = collect( LinRange(limit_a[d], limit_b[d], Nv_array[d]) for d = 1:D )
# Xv_nD = Utilities.ranges2collection(xv_ranges, Val(D))
# h_Xv_nD = h.(Xv_nD)

xv_ranges = collect( x_ranges[d_i] for d = 1:2 )
xv_ranges[2] = x_ranges[d_j]
fig_num = VisualizationTools.visualizemeshgridpcolor(xv_ranges,
                  h_X, [], "x", fig_num, "h")

@assert 1==2

# TODO develop visualize pair-wse marginals.


import Plots

X_array = Vector{Vector{Float64}}(undef, 4)
X_array[1] = b1_MCMC
X_array[2] = b2_MCMC
X_array[3] = b3_MCMC
X_array[4] = b0_MCMC

output_folder_name = "/home/roy/MEGAsync/outputs/debug"
dummy_fig = Plots.plot()
plotallpairwisecontours(X_array, dummy_fig, 50, 50,
                        output_folder_name)

@assert 1==2
