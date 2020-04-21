
# quick tests.

#import Utilities
using Distributed
using SharedArrays

#import JLD
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

import FiniteDiff
import ForwardDiff

#import stickyHDPHMM

import Printf
import GSL

include("../src/integration/numerical.jl")
include("../src/kernels/RQ_non_adaptive.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

D = 3
limit_a = [-14.0; -15.0]
limit_b = [16.0; 15.0]
a_RQ = 0.7
N = 100

X = collect( randn(D) for n = 1:N )

# test one kernel first.
z = X[1]
k = xx->(1/sqrt(a_RQ+dot(xx-z,xx-z))^3)
dk_AD = xx->FiniteDiff.finite_difference_gradient(k, xx)

#### test point.
x0 = randn(D)

#### zero-th order.

h = xx->evalintegral(tt->k([xx[1:end-1]; tt]), limit_a[end], xx[end])
w = xx->evalCDFnonadaptiveRQ(z, a_RQ, limit_a[end], xx)

println("w(x0) = ", w(x0))
println("h(x0) = ", h(x0))
println("discrepancy = ", abs(w(x0)-h(x0)))
println()

#### first-order

dh_ND = xx->FiniteDiff.finite_difference_gradient(h, xx)
dw_AD = xx->ForwardDiff.gradient(w, xx)
dw_AN = xx->[ collect( eval∂w∂vi(xx[1:end-1],
                z, a_RQ, limit_a[end], xx[end], d) for d = 1:D-1);
                k(xx)]

println("AN: dw(x0) = ", dw_AN(x0))
println("AD: dw(x0) = ", dw_AD(x0))
println("ND: dh(x0) = ", dh_ND(x0))
println("discrepancy between AN and AD, for dw = ", norm(dw_AD(x0)-dw_AN(x0)))
println("discrepancy between dw_AD and dh_ND = ", norm(dw_AD(x0)-dh_ND(x0)))
println()

### second-order.

d2h_ND = xx->FiniteDiff.finite_difference_hessian(h, xx)
d2w_AD = xx->ForwardDiff.hessian(w, xx)
d2w_AN = xx->collect( eval∂2w∂vij(      xx[1:D-1], z, a_RQ,
                                        limit_a[end],
                                        xx[end],
                                        i, j) for i = 1:D-1, j = 1:D-1 )

println("AN: d2w(x0) = ", d2w_AN(x0))
println("AD: d2w(x0) = ", d2w_AD(x0))
println("ND: d2h(x0) = ", d2h_ND(x0))
println("discrepancy between AN and AD, for d2w = ", norm(d2w_AD(x0)[1:D-1,1:D-1]-d2w_AN(x0)))
println("discrepancy between d2w_AD and d2h_ND = ", norm(d2w_AD(x0)-d2h_ND(x0)))
println()

#####
# repeat everything for query functions, and have persists.

println()
println()
println("Testing Query.")
println()

N = 100
c = rand(N)
X = collect( randn(D) for n = 1:N )
f = xx->evalnonadaptiveRQquery(xx, c, X, a_RQ)

h = xx->evalintegral(tt->f([xx[1:end-1]; tt]), limit_a[end], xx[end])
w = xx->evalnonadaptiveRQqueryCDF(c, X, a_RQ, limit_a[end], xx)

println("w(x0) = ", w(x0))
println("h(x0) = ", h(x0))
println("discrepancy = ", abs(w(x0)-h(x0)))
println()


#### first-order

dh_ND = xx->FiniteDiff.finite_difference_gradient(h, xx)
dw_AD = xx->ForwardDiff.gradient(w, xx)
dw_AN = xx->[ collect( eval∂w∂viquery( c, X, a_RQ, limit_a[end],
                                  xx, d) for d = 1:D-1); f(xx)]

println("AN: dw(x0) = ", dw_AN(x0))
println("AD: dw(x0) = ", dw_AD(x0))
println("ND: dh(x0) = ", dh_ND(x0))
println("discrepancy between AN and AD, for dw = ", norm(dw_AD(x0)-dw_AN(x0)))
println("discrepancy between dw_AD and dh_ND = ", norm(dw_AD(x0)-dh_ND(x0)))
println()


## second-order.

d2h_ND = xx->FiniteDiff.finite_difference_hessian(h, xx)
d2w_AD = xx->ForwardDiff.hessian(w, xx)
d2w_AN = xx->collect( eval∂2w∂vijquery( c, X, a_RQ,
                                        limit_a[end],
                                        xx,
                                        i, j) for i = 1:D-1, j = 1:D-1 )

println("AN: d2w(x0) = ", d2w_AN(x0))
println("AD: d2w(x0) = ", d2w_AD(x0))
println("ND: d2h(x0) = ", d2h_ND(x0))
println("discrepancy between AN and AD, for d2w = ", norm(d2w_AD(x0)[1:D-1,1:D-1]-d2w_AN(x0)))
println("discrepancy between d2w_AD and d2h_ND = ", norm(d2w_AD(x0)-d2h_ND(x0)))
println()



#### derivative test.

df = xx->ForwardDiff.gradient(f, xx)

#
df_AN = xx->collect( evalnonadaptiveRQ∂f∂xj(xx[end],
                        xx[1:end-1], c, X, a_RQ, j) for j = 1:D )


println("AD: df(x0) = ", df(x0))
println("AN: df(x0) = ", df_AN(x0))
println("discrepancy = ", norm(df(x0)-df_AN(x0)))
println()
