

import DifferentialEquations

import Printf
import PyPlot
import Random
import Optim

using LinearAlgebra

using FFTW

import Utilities
import Distributions
import VisualizationTools

include("basic_model.jl")
include("parsers.jl")
include("../data/epidemiology.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

infection_data1 = eyam2()
infection_data2 = eyam2()

## just the second year.
infection_data = infection_data2

## full dataset.
#infection_data = [infection_data1; infection_data2]
#infection_data = [1.0; infection_data]

N_pop = 261.0
I_0 = convert(Float64, infection_data[1])
R_0 = 0.0

S_0 = N_pop - I_0

discount_factor = 1.0

# simulate the differential equation for 2 more months than the data.
t0 = 0.0
t_end = convert(Float64, length(infection_data) + 2)



#### set up data for inference.
#N = 800 # gives a very degenerate posterior.
Y = convert(Vector{Float64}, infection_data[2:end])
N = length(Y)
𝓣 = collect( i*1.0 for i = 1:length(Y))

### inference.
D = 2
const_params = [ discount_factor, N_pop, R_0, S_0, I_0 ]

σ = 20.0 # fancy shape.
# σ = 2.0 # very condense region of space.
Σ_y = diagm( ones(N) .* σ^2 )


ln_f = xx->evalposteriorSIR(xx,
                    const_params,
                    parser2,
                    Σ_y,
                    Y,
                    𝓣;
                    t0 = 0.0,
                    t_end = 365.0,
                    β_prior_shape = 1.0,
                    β_prior_scale = 1.0,
                    γ_prior_shape = 1.0,
                    γ_prior_scale = 1.0)
#
f = xx->exp(ln_f(xx))


### visualize.

limit_a = [0.0; 0.0]
limit_b = [6.0; 6.0]

N_array = [100; 100]
x_ranges = collect( LinRange(limit_a[d], limit_b[d], N_array[d]) for d = 1:D )

X_nD = Utilities.ranges2collection(x_ranges, Val(D))

#
f_X_nD = f.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges,
                            f_X_nD, [], ".", fig_num,
                            "f, unnormalized density")
#
ln_f_X_nD = ln_f.(X_nD)
fig_num = VisualizationTools.visualizemeshgridpcolor(x_ranges,
                            ln_f_X_nD, [], ".", fig_num,
                            "ln_f, unnormalized density")
#

max_val, max_ind = findmax(ln_f_X_nD)
β_star, γ_star = X_nD[max_ind]


S_t, I_t, R_t = solveSIRnumerical(I_0, S_0, R_0,
                        β_star, γ_star, N_pop; t0 = t0, t_end = t_end)

Nq = 500
xq = LinRange(t0, t_end, Nq)

infected_xq = I_t.(xq)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(xq, infected_xq, label = "I(t)")
PyPlot.plot([0.0; 𝓣], [I_0; Y], "x", label = "data")

PyPlot.title("I(t) using max ln_f_X_nD solution.")
PyPlot.legend()
