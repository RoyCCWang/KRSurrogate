
import HCubature
import Random

import Distributions

import PyPlot

import Utilities

using LinearAlgebra

include("../src/integration/numerical.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

D = 3



μ = randn(D)
Σ = Utilities.generaterandomposdefmat(D)

dist = Distributions.MvNormal(μ, Σ)

f = xx->Distributions.pdf(dist, xx)

limits_a = [-15.0; 14.0; 4.0]
limits_b = [7.0; 8.8; 13.2]

# I am here. glaring bug. need to do this on compact domain!!



f_joint = xx->evaljointpdfcompact(xx, f, limits_a, limits_b)

N_realizations = 100
X = collect( randn(2) for n = 1:N_realizations )

f_X = f_joint.(X)
