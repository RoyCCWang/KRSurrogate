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

import Calculus
import ForwardDiff


#import stickyHDPHMM

import Printf
import GSL



include("../src/approximation/approx_helpers.jl")
include("../src/approximation/final_approx_helpers.jl")
include("../src/approximation/analytic_cdf.jl")
include("../src/approximation/fit_mixtures.jl")
include("../src/approximation/optimization.jl")

include("../src/integration/numerical.jl")
include("../src/integration/adaptive_helpers.jl")
include("../src/integration/DEintegrator.jl")

include("../src/density/density_helpers.jl")
include("../src/density/fit_density.jl")

include("../src/misc/final_helpers.jl")
include("../src/misc/test_functions.jl")
include("../src/misc/utilities.jl")
include("../src/misc/declarations.jl")

include("../src/quantile/derivative_helpers.jl")
include("../src/quantile/Taylor_inverse_helpers.jl")
include("../src/quantile/numerical_inverse.jl")

include("../src/splines/quadratic_itp.jl")

include("../src/KR/engine_Taylor.jl")
include("../src/derivatives/RQ_Taylor_quantile.jl")
include("../src/derivatives/RQ_derivatives.jl")
include("../src/derivatives/traversal.jl")
include("../src/derivatives/ROC_check.jl")

include("./src/engine.jl")
include("./src/utilities.jl")
include("./src/dG.jl")
include("./src/d2G.jl")
include("./src/single_KR.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

## select dimension.
D = 2
N_array = [100; 100]
limit_a = [-14.0; -15.0]
limit_b = [16.0; 15.0]

# D = 3
# N_array = [100; 100; 100]
# limit_a = [-14.0; -1.0; -11.0]
# limit_b = [16.0; 15.0; 5.0]

## select other parameters for synthetic dataset.
N_components = 3
N_full = 1000
#N_realizations = 500 # 30 seconds to fit each dim.
N_realizations = 100 # 30 seconds to fit each dim.

## generate dataset.

Y_dist, f, x_ranges = setupsyntheticdataset(N_components,
                                N_array, limit_a, limit_b,
                                getnDGMMrealizations, Val(D))
#Y_dist, f, x_ranges, X_nD, t0, offset = lowdimsetup(N_components, N_array, 15.0, Val(D))

𝑋 = collect( rand(Y_dist, 1) for n = 1:N_realizations)
#

# prepare all marginal joint densities.
#f_joint = xx->evaljointpdf(xx,f,D)[1]
f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

###### fit KR, adaptive kernels with RQ as canonical kernel.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 5000
σ² = 1e-5
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5

a_array = 0.7 .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.
N_X[1] = 25

X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
f_X_array = collect( f_joint.(X_array[d]) for d = 1:D )


println("Timing: fitadaptiveKRviaTaylorW2")
@time c_array, 𝓧_array, θ_array,
        dφ_array, d2φ_array,
        dϕ_array, d2ϕ_array = fitadaptiveKRviaTaylorW2(f_X_array, X_array,
                                            x_ranges, f_joint,
                                            max_iters_RKHS, a_array, σ²,
                                            amplification_factor, N_bands,
                                            attenuation_factor_at_cut_off,
                                            zero_tol_RKHS, prune_tol, max_integral_evals)

println("Number of kernel centers kept, per dim:")
println(collect( length(c_array[d]) for d = 1:2))

# batch-related parameters.
# N_viz = 6000 # 35 seconds on Ryzen 7.
N_viz = 1
N_batches = 15 #100

# quantile-related parameters.
max_integral_evals = 10000
quantile_iters = 1000
N_nodes_tanh = 200
m_tsq = 7
quantile_err_tol = 1e-3 #0.01  both gave l-2 norm( abs(u-u_rec) ), summed over all dimensions: 0.024788099072235982
max_traversals = 50
N_predictive_traversals = 40
correction_epoch = 10
quantile_max_iters = 500
quantile_convergence_zero_tol = 1e-6 #1e-4
n_limit = 10
n0 = 5

KR_config = KRTaylorConfigType( x_ranges,
                                quantile_iters,
                                max_integral_evals,
                                N_nodes_tanh,
                                m_tsq,
                                quantile_err_tol,
                                max_traversals,
                                N_predictive_traversals,
                                correction_epoch,
                                quantile_max_iters,
                                quantile_convergence_zero_tol,
                                n_limit,
                                n0 )

#
# parallelapplyadaptiveKRviaTaylorW2

Printf.@printf("Timing for sequential application of KR, %d particles.\n", N_viz)
@time x_array, discrepancy_array = evalKR(N_viz,
                                            c_array,
                                            θ_array,
                                            𝓧_array,
                                            KR_config.max_integral_evals,
                                            KR_config.x_ranges,
                                            dφ_array,
                                            d2φ_array,
                                            KR_config.N_nodes_tanh,
                                            KR_config.m_tsq,
                                            KR_config.quantile_err_tol,
                                            KR_config.max_traversals,
                                            KR_config.N_predictive_traversals,
                                            KR_config.correction_epoch,
                                            KR_config.quantile_max_iters,
                                            KR_config.quantile_convergence_zero_tol,
                                            KR_config.n_limit,
                                            KR_config.n0)
#

max_val, max_ind = findmax(norm.(discrepancy_array))
println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(discrepancy_array)))
println("largest l-1 discrepancy is ", max_val)
println("At that case: x = ", x_array[max_ind])
println()

Tmap = uu->evalKR(1,
                                            c_array,
                                            θ_array,
                                            𝓧_array,
                                            KR_config.max_integral_evals,
                                            KR_config.x_ranges,
                                            dφ_array,
                                            d2φ_array,
                                            KR_config.N_nodes_tanh,
                                            KR_config.m_tsq,
                                            KR_config.quantile_err_tol,
                                            KR_config.max_traversals,
                                            KR_config.N_predictive_traversals,
                                            KR_config.correction_epoch,
                                            KR_config.quantile_max_iters,
                                            KR_config.quantile_convergence_zero_tol,
                                            KR_config.n_limit,
                                            KR_config.n0,
                                            uu)[1][1]

u0 = collect( convert(Float64, drawstdnormalcdf()) for d = 1:D )
x0 = Tmap(u0)


dTmap_ND = xx->Calculus.jacobian(Tmap,xx,:central)
dT_u0_ND = dTmap_ND(u0)
inv_dT_u0_ND = inv(dT_u0_ND)

println("dT_u0_ND = ")
display(dT_u0_ND)
println()

println("inverse dT_u0_ND = ")
display(inv_dT_u0_ND)
println()

d_select = 2

gq_array = Vector{Function}(undef, D)
gq_array[1] = xx->evalquery(xx[1], c_array[1], 𝓧_array[1], θ_array[1])
gq_array[2] = xx->evalquery(xx, c_array[2], 𝓧_array[2], θ_array[2])

CDF_array = Vector{Function}(undef, D)
CDF_array[1] = xx->evalCDFv(gq_array[1], xx, limit_a[1], limit_b[1];
                max_integral_evals = max_integral_evals )
CDF_array[2] = xx->evalCDFv(gq_array[2], xx, limit_a[2], limit_b[2];
                max_integral_evals = max_integral_evals )

Gmap = xx->collect( CDF_array[d](xx[1:d]) for d = 1:D )
u0_rec = Gmap(x0)
println("F(x0) = ", u0_rec)
println("u0 = ", u0)
println("discrepancy = ", norm(u0-u0_rec))
println()

# verify inverse function theorem.
dG_ND = xx->Calculus.jacobian(Gmap,xx,:central)

dG1_x0_eval_ND = Calculus.gradient(CDF_array[1], x0[1:1])
dG2_x0_eval_ND = Calculus.gradient(CDF_array[2], x0[1:2])
println("dG1_x0_eval_ND = ", dG1_x0_eval_ND)
println("dG2_x0_eval_ND = ", dG2_x0_eval_ND)
println()

# verify inverse function theorem.
dG_ND = xx->Calculus.jacobian(Gmap, xx, :central)

dG_x0_ND = dG_ND(x0)
println("ND: dG_x0 = ")
display(dG_x0_ND)
println()

fq = gq_array[2]

x_full = x0
v = x_full[1:D-1]
∂f_∂x_array, #∂f_loadx_array,
    f_v, Z_v = setupdGcomponent(fq, v,
                        x_full,
                        𝓧_array[d_select],
                        c_array[d_select],
                        θ_array[d_select].warpfunc,
                        dϕ_array[d_select],
                        θ_array[d_select].canonical_params.a,
                        limit_a[d_select],
                        limit_b[d_select];
                        max_integral_evals = max_integral_evals)



df = xx->Calculus.derivative(fq, xx)

df_dx_eval_ND = df(x_full)
df_dx_eval_AN = collect( ∂f_∂x_array[d](x_full[end]) for d = 1:D )

println("df_dx_eval_ND = ", df_dx_eval_ND)
println("df_dx_eval_AN = ", df_dx_eval_AN)
println()

### second derivative.


# # verify the multivariate Bruno formula.
# k = xx->RKHSRegularization.evalkernel(xx, 𝓧_array[d_select][3], θ_array[d_select])
# d2k_ND = xx->Calculus.hessian(k,xx)
#
# z = 𝓧_array[d_select][3]
# w = θ_array[d_select].warpfunc
# a = θ_array[d_select].canonical_params.a
#
# u = xx->(dot(xx-z, xx-z)+(w(xx)-w(z))^2)
# d2u = xx->ForwardDiff.hessian(u, xx)
# du = xx->ForwardDiff.gradient(u, xx)
#
# f = uu->(1/sqrt(a+uu)^3) # this is k in my notes.
# df = uu->ForwardDiff.derivative(f,uu)
# d2f = uu->ForwardDiff.derivative(df,uu)
# #df = uu->1.5/sqrt(θ_array[d_select].canonical_params+uu)^5
#
# x0 = randn(2)
# x0 = randn(2)
# x0 = randn(2)
# x0 = randn(2)
#
# u_x0 = u(x0)
#
#
#
# q2_1 = d2f(u_x0)
# q1_1 = df(u_x0)
#
# q1_2 = d2u(x0)[1,2]
# q2_2 = prod(du(x0))
# LHS12 = q1_1 * q1_2 + q2_1 * q2_2
#
# q1_2 = d2u(x0)[1,1]
# q2_2 = du(x0)[1]^2
# LHS11 = q1_1 * q1_2 + q2_1 * q2_2
#
# q1_2 = d2u(x0)[2,2]
# q2_2 = du(x0)[2]^2
# LHS22 = q1_1 * q1_2 + q2_1 * q2_2
#
# RHS = d2k_ND(x0) ./ sqrt(a)^3
#
# println("LHS11 = ", LHS11)
# println("LHS12 = ", LHS12)
# println("LHS22 = ", LHS22)
# println("RHS = ", RHS)
# println()
#
# #@assert 2222==3
#
# # verify second derivatives of a RQ kernel.
# #k = xx->RKHSRegularization.evalkernel(xx, 𝓧_array[d_select][3], θ_array[d_select])
# #d2k_ND = xx->Calculus.hessian(k,xx)
#
# #z = 𝓧_array[d_select][3]
# ϕ = θ_array[d_select].warpfunc
# dϕ = dϕ_array[d_select]
# d2ϕ = d2ϕ_array[d_select]
# #a = θ_array[d_select].canonical_params.a
#
# d2w = xx->ForwardDiff.hessian(w, xx)
# dw = xx->ForwardDiff.gradient(w, xx)
#
# d2w_ND = xx->Calculus.hessian(w, xx)
# d2w_AD = xx->ForwardDiff.hessian(w, xx)
#
# ∂u_∂x_x0_AN = eval∂2kwrt∂x2RQ(x0, z, ϕ, dϕ, d2ϕ, a)
#
# h = xx->ϕ(xx)*dϕ(xx)
# ∂u_∂x_ND = xx->ForwardDiff.jacobian(h,xx)
# ∂u_∂x_x0_ND = ∂u_∂x_ND(x0)
#
# # println("∂u_∂x_x0_AN = ", ∂u_∂x_x0_AN)
# # println("∂u_∂x_x0_ND = ", ∂u_∂x_x0_ND)
# # println()
# #
# # # println("d2w(x0) = ", d2w(x0))
# # # println("d2w_ND(x0) = ", d2w_ND(x0))
# # # println("d2w_AD(x0) = ", d2w_AD(x0))
# # # println()
# # #
# # #
# # # println("norm( d2w(x0) - d2w_AD(x0) ) = ", norm( d2w(x0) - d2w_AD(x0) ))
# #
# #
# # @assert 1==2333
# #
# # ∂2u_∂x2_x0_AN = eval∂2kwrt∂x2RQ(x0, z, ϕ, dϕ, d2ϕ, a)
# # ∂2u_∂x2_x0_AD = d2u(x0)
# #
# # println("∂2u_∂x2_x0_AN = ", ∂2u_∂x2_x0_AN)
# # println("∂2u_∂x2_x0_AD = ", ∂2u_∂x2_x0_AD)
# # println()
# #
# # @assert 1==2333
#
# d2k_x0_AN = eval∂2kwrt∂x2RQ(x0, z, ϕ, dϕ, d2ϕ, a)
# d2k_x0_AD = d2k_ND(x0) ./ sqrt(a)^3
#
# d2k_x0_AN2 = eval∂2kwrt∂x2RQviacomponents(x0, z, ϕ, dϕ, d2ϕ, a, 2, 2)
#
# println("d2k_x0_AN  = ", d2k_x0_AN)
# println("d2k_x0_AN2 = ", d2k_x0_AN2)
# println("d2k_x0_AD  = ", d2k_x0_AD)
# println()
#
# # to do: stress test this. hessian not symmetric sometimes?!
#
# @assert 1==2333

∂2f_∂x2_array, #∂f_loadx_array,
    f_v, Z_v = setupd2Gcomponent(fq, v,
                        x_full,
                        𝓧_array[d_select],
                        c_array[d_select],
                        θ_array[d_select].warpfunc,
                        dϕ_array[d_select],
                        d2ϕ_array[d_select],
                        θ_array[d_select].canonical_params.a,
                        limit_a[d_select],
                        limit_b[d_select];
                        max_integral_evals = max_integral_evals)

d2f = xx->Calculus.hessian(fq, xx)

d2f_dx_eval_ND = d2f(x_full)
#d2f_dx_eval_AN = collect( ∂2f_∂x2_array[d](x_full[end]) for d = 1:D )
d2f_dx_eval_AN = collect( ∂2f_∂x2_array[i,j](x_full[end]) for i = 1:D, j = 1:D )

println("d2f_dx_eval_ND = ", d2f_dx_eval_ND)
println("d2f_dx_eval_AN = ", d2f_dx_eval_AN)
println()


function evalhtilde(f, x_full, limit_a, limit_b, d)
    v = x_full[1:d-1]
    x = x_full[d]
    f_v = xx->f([v; xx])
    out = evalintegral(  f_v, limit_a[d], x_full[d])

    return out
end

function evalh(f, x_full, limit_a, limit_b, d)
    v = x_full[1:d-1]
    x = x_full[d]
    f_v = xx->f([v; xx])
    out = evalintegral(  f_v, limit_a[d], x_full[d])

    Z = evalintegral(f_v, limit_a[d], limit_b[d])
    return out/Z
end

x = x_full[d_select]
v = x_full[1:d_select-1]
a = limit_a[d_select]
b = limit_b[d_select]
f_v = xx->fq([v; xx])

h_tilde = xx->evalhtilde(fq, xx, limit_a, limit_b, d_select)
LHS = Calculus.gradient(h_tilde, x_full)

#∂f_loadx_array[1](x0)
RHS2 = evalintegral(∂f_∂x_array[1], a, x)

df1_v = xx->df([v;xx])[1]
RHS = evalintegral(df1_v, a, x)

println("LHS = ", LHS)
println("RHS = ", RHS)
println("RHS2 = ", RHS2)
println()

h = xx->evalh(fq, xx, limit_a, limit_b, d_select)
LHS = Calculus.gradient(h, x_full)

df1_v = xx->df([v;xx])[1]
h_x = evalintegral(f_v, a, x)
#Z_v = evalintegral(f_v, a, b)
∂h_x = evalintegral(∂f_∂x_array[1], a, x)
∂Z_v = evalintegral(∂f_∂x_array[1], a, b)
# ∂h_x = evalintegral(df1_v, a, x)
# ∂Z_v = evalintegral(df1_v, a, b)

numerator = ∂h_x*Z_v - h_x*∂Z_v
denominator = Z_v^2
RHS = numerator/denominator

println("LHS = ", LHS)
println("RHS = ", RHS)
println()

#@assert 1==2

∂Gd_∂x = computedGcomponent(    x_full[d_select],
                            f_v,
                            Z_v,
                            ∂f_∂x_array,
                            limit_a[d_select],
                            limit_b[d_select],
                            d_select)


println("∂Gd_∂x = ", ∂Gd_∂x)
println("dG2_x0_eval_ND = ", dG2_x0_eval_ND)
println()


# next: do 2nd order derivatives.
# do not fit full dimension.
# applications. This is the most uncertain task of this project.

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5643013/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5414245/
# https://www.hindawi.com/journals/bmri/2019/8304260/
