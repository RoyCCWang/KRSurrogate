

import DifferentialEquations

import Printf
import PyPlot
import Random
import Optim

using LinearAlgebra

using FFTW

import Utilities
import Distributions

include("basic_model.jl")
include("parsers.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

N_pop = 10000.0
I_0 = 10.0
R_0 = 0.0

S_0 = N_pop - I_0

#β = 1.22*N_pop/6
β = 0.22
γ = 1/6

discount_factor = 1.0

println("oracle: β = ", β)
println("oracle: γ = ", γ)
println("oracle: S_0 = ", S_0)
println("oracle: I_0 = ", I_0)
println("oracle: R_0 = ", R_0)
println("oracle: k = ", discount_factor)
println("oracle: N_pop = ", N_pop)
println()

#θ = [N_pop; β; γ]
#u0 = [𝑆_0; 𝐼_0; 𝑅_0]

t0 = 0.0
t_end = 365.0
S_t, I_t, R_t = solveSIRnumerical(I_0, S_0, R_0,
                        β, γ, N_pop; t0 = t0, t_end = t_end)



Nq = 500
xq = LinRange(t0, t_end, Nq)

infected_xq = I_t.(xq)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(xq, infected_xq, label = "I(t)")

PyPlot.title("I(t)")
PyPlot.legend()


#### generate data.
#N = 800 # gives a very degenerate posterior.
N = 5 #80
𝓣 = collect( Utilities.convertcompactdomain(
                                rand(),
                                0.0,
                                1.0,
                                t0,
                                t_end) for n = 1:N)

σ_Y = 20.0
Y = collect( discount_factor * I_t(𝓣[i]) + randn() * σ_Y for i = 1:N )

# conclusion: posterior is too ill-posed for direct density fitting.
# remedy: Gaussian flow, or some factorization approach.

#### unnormalized posterior density of parameters from evaluations of Y.
# when N is large, the unnormalized posterior density should be concentrated
#       at the oracle parameters.
# This tests the identifiability claims.

D = 2
const_params = [ discount_factor, N_pop, R_0, S_0, I_0 ]

σ = σ_Y + 1e-5
Σ_y = diagm( ones(N) .* σ^2 )

# debug.
dist_β = Distributions.Gamma(1.0, 1.0)
dist_γ = Distributions.Gamma(1.0, 1.0)

ln_p_β = Distributions.logpdf(dist_β, β)
ln_p_γ = Distributions.logpdf(dist_γ, γ)


μ = discount_factor .* I_t.(𝓣)
dist_y = Distributions.MvNormal(μ, Σ_y)
ln_likelihood = Distributions.logpdf(dist_y, Y)

res = Y-μ
prop_ln_likelihood = -dot(res, Σ_y\res)/2

Z = -0.5*logdet(Σ_y) -length(Y)/2 *log(2*π)

A = prop_ln_likelihood+Z
println("ln_likelihood = ", ln_likelihood)
println("A = ", A) #should be same as ln_likelihood
println()

println("prop_ln_likelihood = ", prop_ln_likelihood)

@assert 1==2


# out, μ = evalposteriorSIR([β, γ],
#                 const_params,
#                 parser2,
#                 Σ_y,
#                 Y,
#                 𝓣;
#                 t0 = 0.0,
#                 t_end = 365.0,
#                 β_prior_shape = 1.0,
#                 β_prior_scale = 1.0,
#                 γ_prior_shape = 1.0,
#                 γ_prior_scale = 1.0)
#
# #
# println("out = ", out)
# println("discrepancy between μ and Y = ", norm(μ-Y))
# println()
#
# @assert 1==2

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

### sanity check.
x_max = [β, γ]
x_test = x_max + randn(D) .* 1e-3
println("ln_f(x_max)  = ", ln_f(x_max))
println("ln_f(x_test) = ", ln_f(x_test))
println()

### visualize.

limit_a = [0.0; 0.0]
limit_b = [0.5; 0.5]

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
println("oracle: β is ", β)
println("oracle: γ is ", γ)
println()

# conclusion: posterior is too ill-conditioned for density fitting.
# just selecting kernel centers will be troublesome.
