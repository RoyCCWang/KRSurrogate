
import PyPlot
import Distributions

import Random
using LinearAlgebra

import SpecialFunctions
import Statistics

include("../src/misc/normal_distribution.jl")
include("../src/misc/visualize.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

D = 2

# src_μ = randn(D) .* 5.0
# src_σ_array = rand(D) .* 5.0
src_μ = zeros(D)
src_σ_array = ones(D) .* 0.7
src_dist = Distributions.MvNormal(src_μ, diagm(src_σ_array))

pdf = xx->Distributions.pdf(src_dist, xx)

### actual joint CDF>
# # presend its inifinite.
# x_min = -20.0 .* ones(D)
# #x_max = 20.0 .* ones(D)
#
# max_integral_evals = 10000
# initial_divisions = 2
#
# CDF_NI = xx->HCubature.hcubature(pdf, x_min, xx;
#                 maxevals = max_integral_evals,
#                 initdiv = initial_divisions)

### We want the cdf of the univariate conditionals.


## I suspect this CDF is wrong.
src_dist_array = collect( Distributions.Normal(src_μ[d], src_σ_array[d]) for d = 1:D )
CDF_ref = xx->collect( Distributions.cdf(src_dist_array[d], xx[d]) for d = 1:D )


CDF2 = xx->collect( evalunivariatenormalcdf(xx[d], src_μ[d], sqrt(src_σ_array[d]) ) for d = 1:D )

#
x0 = rand(src_dist)
ans1 = CDF_ref(x0)


#
N_viz = 10000
X0 = collect( rand(src_dist) for n = 1:N_viz )

U0 = collect( collect( drawstdnormalcdf() for d = 1:D ) for n = 1:N_viz )

U1 = CDF_ref.(X0)
U2 = CDF2.(X0)

println("discrepancy between U1, U2 is ", norm(U1-U2))
println()

mean_U0 = Statistics.mean(U0)
mean_U1 = Statistics.mean(U1)

var_U0 = Statistics.var(U0)
var_U1 = Statistics.var(U1)

println("mean_U0 = ", mean_U0)
println("mean_U1 = ", mean_U1)
println("mean should be near 0.5")
println()

println("var_U0 = ", var_U0)
println("var_U1 = ", var_U1)
println("var should be near 1/12 = ", 1/12)
println()


n_bins = 50
limit_a = zeros(D)
limit_b = ones(D)

fig_num = visualize2Dhistogram(fig_num, U0, limit_a, limit_b;
                                use_bounds = true, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "U0")
#
fig_num = visualize2Dhistogram(fig_num, U2, limit_a, limit_b;
                                use_bounds = false, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "U1")
