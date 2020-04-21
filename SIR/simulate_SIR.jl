

import DifferentialEquations

import Printf
import PyPlot
import Random
import Optim

using LinearAlgebra

using FFTW

import Utilities

include("basic_model.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

N = 10000.0
I_0 = 10.0
R_0 = 0.0

S_0 = N - I_0

#β = 1.22*N/6
β = 0.22
γ = 1/6

#θ = [N; β; γ]
#u0 = [𝑆_0; 𝐼_0; 𝑅_0]

t0 = 0.0
t_end = 365.0
S_t, I_t, R_t = solveSIRnumerical(I_0, S_0, R_0,
                        β, γ, N; t0 = t0, t_end = t_end)



Nq = 500
xq = LinRange(t0, t_end, Nq)

infected_xq = I_t.(xq)

PyPlot.figure(fig_num)
fig_num += 1

PyPlot.plot(xq, infected_xq, label = "I(t)")

PyPlot.title("I(t)")
PyPlot.legend()


# theory-check. Should be very close.
q = (β/N)/γ
I_max = I_0 + S_0 - (1+ log(q*S_0))/q

println("computed I_max is ", I_max)
println("max of infected_xq is ", maximum(infected_xq))
println()
