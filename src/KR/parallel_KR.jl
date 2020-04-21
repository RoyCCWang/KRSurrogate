function parallelevalKR(  c_array::Vector{Vector{T}},
                            θ_array::Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                            𝓧_array::Vector{Vector{Vector{T}}},
                            d_ϕ_array::Vector{Function},
                            d2_ϕ_array::Vector{Function},
                            M::Int,
                            N_batches::Int,
                            config::KRTaylorConfigType{T,RT}) where {T <: Real, RT}

    # work on intervals.
    M_for_each_batch = Vector{Int}(undef, N_batches)
    𝑀::Int = round(Int, M/N_batches)
    fill!(M_for_each_batch, 𝑀)
    M_for_each_batch[end] = abs(M - (N_batches-1)*𝑀)

    @assert M == sum(M_for_each_batch) # sanity check.

    # prepare worker function.
    workerfunc = xx->evalKR(   xx,
                                c_array,
                                θ_array,
                                𝓧_array,
                                config.max_integral_evals,
                                config.x_ranges,
                                d_ϕ_array,
                                d2_ϕ_array,
                                config.N_nodes_tanh,
                                config.m_tsq,
                                config.quantile_err_tol,
                                config.max_traversals,
                                config.N_predictive_traversals,
                                config.correction_epoch,
                                config.quantile_max_iters,
                                config.quantile_convergence_zero_tol,
                                config.n_limit,
                                config.n0)

    # compute solution.
    sol = pmap(workerfunc, M_for_each_batch )

    # unpack solution.
    x_array, discrepancy_array = unpackpmap(sol, M)

    #return x_array, discrepancy_array, sol
    return x_array, discrepancy_array
end
