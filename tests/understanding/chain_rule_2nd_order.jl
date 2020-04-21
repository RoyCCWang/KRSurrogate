
import Random
using LinearAlgebra
import ForwardDiff
import Calculus
import Printf

include("./verification/chain_rule_helpers.jl")
Random.seed!(25)

D = 3

f = uu->collect( sin(norm(uu[1:d] .* 1.23 .* d)^2) for d = 1:D )

g = xx->collect( cos(norm(xx[1:d])^2 - norm(xx[1:d])^4 ) for d = 1:D )
h = xx->f(g(xx))

x0 = randn(D)

# ## experiment first-order.
# df = xx->ForwardDiff.jacobian(f,xx)
# dg = xx->ForwardDiff.jacobian(g,xx)
# dh = xx->ForwardDiff.jacobian(h,xx)
#
# df_x0 = df(x0)
# dg_x0 = dg(x0)
# dh_x0 = dh(x0)
# dgf_x0 = df(g(x0))*dg(x0)
#
# println("df(x0) = ", df_x0)
# println("dg(x0) = ", dg_x0)
# println("dh(x0) = ")
# display(dh_x0)
# println("d(g∘f)(x0) = ")
# display(dgf_x0)
# println()


f_components = collect( uu->f(uu)[d] for d = 1:D )
g_components = collect( xx->g(xx)[d] for d = 1:D )
h_components = collect( xx->h(xx)[d] for d = 1:D )

## experiment first-order.


# function packagetensor(f_array, x0::Vector{T}) where T <: Real
#
# # read: https://en.wikipedia.org/wiki/Multilinear_map#Coordinate_representation
# end

df = xx->collect( ForwardDiff.gradient(f_components[d], xx) for d = 1:D )
dg = xx->collect( ForwardDiff.gradient(g_components[d], xx) for d = 1:D )
dh = xx->collect( ForwardDiff.gradient(h_components[d], xx) for d = 1:D )

d2f = xx->collect( ForwardDiff.hessian(f_components[d], xx) for d = 1:D )
d2g = xx->collect( ForwardDiff.hessian(g_components[d], xx) for d = 1:D )
#d2h = xx->collect( ForwardDiff.hessian(h_components[d], xx) for d = 1:D )


k_select = 3
hk = xx->h(xx)[k_select]
d2hk = xx->ForwardDiff.hessian(hk, xx)

###
d2hk_x0_AD = d2hk(x0)

println("AD: d2hk(x0) = ")
display(d2hk_x0_AD)
println()

###
d2hk_x0_AN = d2fkwrtdx(x0, f, g, df, dg, d2f, d2g, k_select)

println("AN: d2hk(x0) = ")
display(d2hk_x0_AN)
println()

println("discrepancy between AN and AD: ", norm(d2hk_x0_AD-d2hk_x0_AN))
println()


LHS, RHS = d2fkwrtdxterms(x0, f, g, df, dg, d2f, d2g, k_select)

#### Figure out matrix-form of LHS.
# matrix-form of dg(x0).
dg_x0_mat = ForwardDiff.jacobian(g, x0)

d2fk = uu->ForwardDiff.hessian(f_components[k_select], uu)

d2fk_g0 = d2fk(g(x0))
guess_LHS = dg_x0_mat'*d2fk_g0*dg_x0_mat


println("LHS = ")
display(LHS)
println()

println("guess_LHS = ")
display(guess_LHS)
println()

println("discrepancy between LHS and guess: ", norm(LHS-guess_LHS))
println()

#### Figure out matrix-form for RHS.

# matrix-form of dg(x0).
dfk_g0 = ForwardDiff.gradient(f_components[k_select], g(x0))

d2g_x0 = d2g(x0)
guess_RHS = sum( dfk_g0[l] .* d2g_x0[l] for l = 1:D )

println("RHS = ")
display(RHS)
println()

println("guess_RHS = ")
display(guess_RHS)
println()

println("discrepancy between RHS and guess: ", norm(RHS-guess_RHS))
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
