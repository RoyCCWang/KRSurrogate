# packaged up version of fast phi.


using Distributed
using SharedArrays

#import JLD
import FileIO

import NearestNeighbors

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
import FiniteDiff

include("../tests/example_helpers/test_densities.jl")
include("../src/misc/declarations.jl")

include("../src/KR/engine.jl")
include("../src/misc/normal_distribution.jl")
include("../src/misc/utilities.jl")
include("../src/integration/numerical.jl")
include("../src/KR/dG.jl")
include("../src/KR/d2G.jl")
include("../src/KR/single_KR.jl")
include("../tests/verification/differential_verification.jl")
include("../src/kernel_centers/initial.jl")
include("../src/kernel_centers/subsequent.jl")
include("../src/kernel_centers/kDPP.jl")
include("../src/fit/fit_adaptive.jl")
include("../src/fit/RKHS.jl")
include("../src/misc/declarations.jl")
include("../src/kernel_centers/front_end.jl")

include("../src/kernel_centers/inference_kDPP.jl")

include("../src/Taylor_inverse/front_end.jl")
include("../src/Taylor_inverse/Taylor_inverse_helpers.jl")

include("../src/quantile/setupTaylorquantile.jl")
include("../src/quantile/quantile_engine.jl")
include("../src/Taylor_inverse/ROC_check.jl")
include("../src/Taylor_inverse/RQ_Taylor_quantile.jl")
include("../src/integration/double_exponential.jl")

PyPlot.close("all")
fig_num = 1

Random.seed!(25)

# ## select dimension.
# D = 2
# N_array = [100; 100]
# limit_a = [-14.0; -15.0]
# limit_b = [16.0; 15.0]

D = 3
#N_array = [100; 100; 100]
#N_array = [50; 50; 50]
N_array = [20; 20; 20]
limit_a = [-14.0; -16.0; -11.0]
limit_b = [16.0; 15.0; 5.0]

## select other parameters for synthetic dataset.
N_components = 3
N_realizations = 100

## generate dataset.

Y_dist, f, x_ranges = setupsyntheticdataset(N_components,
                                N_array, limit_a, limit_b,
                                getnDGMMrealizations, Val(D))

### get fit locations.

zero_tol_RKHS = 1e-13
prune_tol = 1.1*zero_tol_RKHS
max_iters_RKHS = 10000
σ_array = sqrt(1e-5) .* ones(Float64, D)
max_integral_evals = 10000 #500 #typemax(Int)
amplification_factor = 1.0 #50.0
attenuation_factor_at_cut_off = 2.0
N_bands = 5

N_preliminary_candidates = 100^D #10000 enough for 2D, not enough for 3D.
candidate_truncation_factor = 0.01 #0.005
candidate_zero_tol = 1e-12

base_gain = 1.0 #1/100 # the higher, the more selection among edges of f.
kDPP_zero_tol = 1e-12
N_kDPP_draws = 1
N_kDPP_per_draw = 50

close_radius_tol = 0.3 #1e-3 #0.3
N_refinements = 10 #50 #20

initial_divisions = 1

center_config = KernelCenterConfigType( amplification_factor,
                attenuation_factor_at_cut_off,
                N_bands,
                N_preliminary_candidates,
                candidate_truncation_factor,
                candidate_zero_tol,
                base_gain,
                kDPP_zero_tol,
                N_kDPP_draws,
                N_kDPP_per_draw,
                zero_tol_RKHS,
                prune_tol,
                close_radius_tol,
                N_refinements,
                max_iters_RKHS,
                #σ²,
                σ_array[end]^2,
                initial_divisions)

#
canonical_a = 0.7
θ_canonical = RKHSRegularization.RationalQuadraticKernelType(canonical_a)

X_nD = Utilities.ranges2collection(x_ranges, Val(D))
X_ref = vec(X_nD)

Y_nD = f.(X_nD)

θ_a, c_history, X_history, error_history,
    X_fit_history, X_pool = getkernelcenters(   θ_canonical,
                        Y_nD,
                        x_ranges,
                        X_ref,
                        f,
                        limit_a,
                        limit_b,
                        center_config)
#
min_error, min_ind = findmin(error_history)
𝑋 = X_history[min_ind]

#𝑋 = collect( vec(rand(Y_dist, 1)) for n = 1:N_realizations)
#

# prepare all marginal joint densities.
#f_joint = xx->evaljointpdf(xx,f,D)[1]
# f_joint = xx->evaljointpdf(xx,f,D,
#                 max_integral_evals = typemax(Int),
#                 initial_divisions = 3)[1] # higher accuracy.
f_joint = xx->evaljointpdfcompact(xx, f, limit_a, limit_b)[1]

###### fit KR, adaptive kernels with RQ as canonical kernel.


a_array = 0.7 .* ones(Float64, D) #[0.7; 0.7] # for each dimension.
N_X = length(𝑋) .* ones(Int, D) #[25; length(𝑋)] # The number of kernels to fit. Will prune some afterwards.

X_array = collect( collect( 𝑋[n][1:d] for n = 1:N_X[d] ) for d = 1:D )
removeclosepositionsarray!(X_array, close_radius_tol)
# to do: auto fuse points with self-adapting close_radius_tol in fit.jl
#       this is to avoid semidef error for the QP solver.

f_X_array = collect( f_joint.(X_array[d]) for d = 1:D )


println("Timing: fitproxydensities")
@time c_array, 𝓧_array, θ_array,
        dφ_array, d2φ_array,
        dϕ_array, d2ϕ_array,
        Y_array = fitmarginalsadaptive(f_X_array, X_array,
                                    x_ranges, f,
                                    max_iters_RKHS, a_array, σ_array,
                                    amplification_factor, N_bands,
                                    attenuation_factor_at_cut_off,
                                    zero_tol_RKHS, prune_tol, max_integral_evals)

println("Number of kernel centers kept, per dim:")
println(collect( length(c_array[d]) for d = 1:D))

gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                        max_integral_evals = max_integral_evals)




PyPlot.figure(fig_num)
fig_num += 1

xq = LinRange(limit_a[1], limit_b[1], 500)
Xq = collect( [xq[n]] for n = 1:length(xq) )

fq_Xq = gq_array[1].(Xq)
fq_𝓧 = gq_array[1].(𝓧_array[1])
#f_Xq = f_joint.(Xq)

PyPlot.plot(xq, fq_Xq, label = "fq")
PyPlot.plot(𝓧_array[1], fq_𝓧, "x", label = "fq kernel centers")
#PyPlot.plot(xq, f_Xq, "--", label = "f")
PyPlot.plot(x_ranges[1], Y_array[1], "^", label = "Y")

PyPlot.title("f vs. fq")
PyPlot.legend()

# try visualizing with histogram, if oracle density is easy to sample from.
# Y_array too low of a resolution.
#


# batch-related parameters.
N_viz = 1
#N_viz = 1000
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

initial_divisions = 1
CDF_f_D = xx->evalCDFv(f, xx, limit_a[D], limit_b[D];
        max_integral_evals = max_integral_evals,
        initial_divisions = initial_divisions )

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

#@assert 1==2

### derivatives.



u0 = collect( convert(Float64, drawstdnormalcdf()) for d = 1:D )

Tmap, dTmap_ND,
  d2Tmap_ND = setupTmapND(     KR_config,
                                c_array,
                                θ_array,
                                𝓧_array,
                                dφ_array,
                                d2φ_array)

x0 = Tmap(u0)


Gmap, dGmap_ND, d2Gmap_ND = setupGmapND(gq_array, CDF_array)

u0_rec = Gmap(x0)
println("F(x0) = ", u0_rec)
println("u0    = ", u0)
println("discrepancy = ", norm(u0-u0_rec))
println()

# verify inverse function theorem.
dG_x0_ND = collect( dGmap_ND[d](x0) for d = 1:D )

println("ND: dG(x0) = ")
display(dG_x0_ND)
println()

### verify dG

dG_x0_AN = computedG(   gq_array,
                        x0,
                        𝓧_array,
                        c_array,
                        θ_array,
                        dϕ_array,
                        limit_a,
                        limit_b;
                        max_integral_evals = max_integral_evals)


println("AN: dG(x0) = ")
display(dG_x0_AN)
println()

discrepancy = sum(collect(norm(dG_x0_ND[d]-dG_x0_AN[d])) for d = 1:D)
println("AN vs. ND for dG(x0) is ", discrepancy )
println()
println()

################ 2nd order.



d2G_x0_ND = collect( Calculus.hessian(CDF_array[d], x0[1:d]) for d = 1:D )
println("ND: d2G(x0):")
display(d2G_x0_ND)
println()


# d_select = 2
# fq = gq_array[d_select]
# v = x0[1:d_select-1]
# x_full = copy(x0)
#
# f_v, p_v, Z_v = setupfv(fq, v,
#                 limit_a[d_select],
#                 limit_b[d_select];
#                 max_integral_evals = max_integral_evals)
#
# ∂f_∂x_array = setupdGcomponent(fq, v,
#                 x0,
#                 𝓧_array[d_select],
#                 c_array[d_select],
#                 θ_array[d_select].warpfunc,
#                 dϕ_array[d_select],
#                 θ_array[d_select].canonical_params.a,
#                 limit_a[d_select],
#                 limit_b[d_select];
#                 max_integral_evals = max_integral_evals)
#
#
# ∂2f_∂x2_array = setupd2Gcomponent(fq, v,
#                         x_full,
#                         𝓧_array[d_select],
#                         c_array[d_select],
#                         θ_array[d_select].warpfunc,
#                         dϕ_array[d_select],
#                         d2ϕ_array[d_select],
#                         θ_array[d_select].canonical_params.a,
#                         limit_a[d_select],
#                         limit_b[d_select];
#                         max_integral_evals = max_integral_evals)
#
# d = 1
# d2G1_x0_AN = computed2Gcomponent(    x_full[d],
#                             f_v,
#                             Z_v,
#                             ∂f_∂x_array,
#                             ∂2f_∂x2_array,
#                             limit_a[d],
#                             limit_b[d],
#                             d)
#
# d = 2
# d2G2_x0_AN = computed2Gcomponent(    x_full[d],
#                             f_v,
#                             Z_v,
#                             ∂f_∂x_array,
#                             ∂2f_∂x2_array,
#                             limit_a[d],
#                             limit_b[d],
#                             d)
#
# println("d2G1_x0_AN = ", d2G1_x0_AN)
# println("d2G2_x0_AN = ", d2G2_x0_AN)
# println()
#
# @assert 1==2

d2G_x0_AN = computed2G(gq_array,
                        x0,
                        𝓧_array,
                        c_array,
                        θ_array,
                        dϕ_array,
                        d2ϕ_array,
                        limit_a,
                        limit_b;
                        max_integral_evals = max_integral_evals)

println("AN: d2G(x0):")
display(d2G_x0_AN)
println()



##### inverse function theorem.

dF_u0_mat,
  dG_x0_mat = computedF(  gq_array,
                        x0,
                        𝓧_array,
                        c_array,
                        θ_array,
                        dϕ_array,
                        limit_a,
                        limit_b;
                        max_integral_evals = max_integral_evals)

println("AN: dF(x0):")
display(dF_u0_mat)
println()

dTmap_u0_ND = dTmap_ND(u0)
println("dTmap_u0_ND =")
display(dTmap_u0_ND)
println()

# import FiniteDiff
# dTmap_u0_FD = FiniteDiff.finite_difference_jacobian(Tmap, u0)
# println("dTmap_u0_FD =")
# display(dTmap_u0_FD)
# println()
#
# @assert 1==2333

inv_dTmap_u0_ND = inv(dTmap_u0_ND)

println("ND: dTmap(u0) = ")
display(dTmap_u0_ND)
println()

println("subtractive discrepancy between AN and ND for dTmap[k] is ")
display(dTmap_u0_ND - dF_u0_mat)
println()

println("ratio discrepancy between AN and ND for dTmap[k] is ")
display(dTmap_u0_ND ./ dF_u0_mat)

println("inverse of ND: dTmap(u0) = ")
display(inv_dTmap_u0_ND)
println()

println("AN: dG(x0) = ")
display(dG_x0_AN)
println()


#### second-order inverse function theorem.
k_select = D

# normalize.
zero_tol = 1e-9
Z_v_sol = evalintegral(gq_array[1], limit_a[1], limit_b[1])
Z_v = clamp(Z_v_sol, zero_tol, 1.0)

if k_select > 1
        v = x0[1:k_select-1]
        f_v, p_v, Z_v = setupfv(gq_array[k_select], v,
                limit_a[k_select],
                limit_b[k_select];
                max_integral_evals = max_integral_evals)
end

d2G_x0_full = packaged2G(d2G_x0_AN)

RHS = sum( dF_u0_mat[k_select,l] .* d2G_x0_full[l] for l = 1:D )
#RHS2 = sum( d2G_x0_full[l] for l = 1:D )

dF_u0_AN = -(dF_u0_mat'*RHS*dF_u0_mat) #./ Z_v

sqrt_RHS = sqrt(RHS)
tmp = (dG_x0_mat')\sqrt_RHS
dF_u0_AN0 = -tmp*tmp'


println("AN: dF(x0) = ")
display(dF_u0_AN)
println()


## component maps.
Tmap_components = collect( xx->Tmap(xx)[d] for d = 1:D )

## second-order derivaitves of the component maps.
#d2Tmap_ND = collect( xx->Calculus.hessian(Tmap_components[d], xx) for d = 1:D )
d2Tmap_ND = collect( xx->FiniteDiff.finite_difference_hessian(Tmap_components[d], xx) for d = 1:D )

d2Tmapk_ND_u0 = d2Tmap_ND[k_select](u0)


println("ND: d2Tmap[k](u0) = ")
display(d2Tmapk_ND_u0)
println()

println("subtractive discrepancy between AN and ND for d2Tmap[k] is ")
display(d2Tmapk_ND_u0 - dF_u0_AN)
println()

println("ratio discrepancy between AN and ND for d2Tmap[k] is ")
display(d2Tmapk_ND_u0 ./ dF_u0_AN)

# next: debug computed2g.
# do not fit full dimension.
# to do: make sure all instances of evalintegral passes max_integral_evals and initial_div.

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5643013/
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5414245/
# https://www.hindawi.com/journals/bmri/2019/8304260/
