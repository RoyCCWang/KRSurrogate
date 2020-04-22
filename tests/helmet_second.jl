Random.seed!(205)

N_viz = 100000 # for 30 helmet.
#N_viz = 1000 # must be larger than N_batch.
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


include("../src/KR/parallel_KR.jl")

Printf.@printf("Timing for parallel application of KR, %d particles.\n", N_viz)
@time x_array, discrepancy_array = parallelevalKR( c_array,
                                            θ_array,
                                            𝓧_array,
                                            dφ_array,
                                            d2φ_array,
                                            N_viz,
                                            N_batches,
                                            KR_config)
#

#

max_val, max_ind = findmax(norm.(discrepancy_array))
println("l-2 norm( abs(u-u_rec) ), summed over all dimensions: ", sum(norm.(discrepancy_array)))
println("largest l-1 discrepancy is ", max_val)
println("At that case: x = ", x_array[max_ind])
println()

#### visualize histogram.
#### visualize histogram.
n_bins = N_array[1] *10
fig_num = visualize2Dhistogram(fig_num, x_array, limit_a, limit_b;
                                use_bounds = true, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "xp, bounds")

fig_num = visualize2Dhistogram(fig_num, x_array, limit_a, limit_b;
                                use_bounds = false, n_bins = n_bins,
                                axis_equal_flag = true,
                                title_string = "xp, no bounds")
