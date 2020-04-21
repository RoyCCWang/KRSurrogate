
function setupKRallnonadaptive( c_array::Vector{Vector{T}},
                        θ_array::Vector{RKHSRegularization.RationalQuadraticKernelType{T}},
                        𝓧_array::Vector{Vector{Vector{T}}},
                        x_ranges,
                        N_nodes_tanh::Int,
                        m_tsq::Int,
                        quantile_max_iters::Int,
                        quantile_convergence_zero_tol::T;
                        df_target::Function = xx->xx,
                        d2f_target::Function = xx->xx,
                        f_target::Function = xx->xx,
                        zero_tol::T = eps(T)*2,
                        initial_divisions::Int = 1,
                        max_integral_evals::Int = 10000,
                        max_numerical_inverse_iters::Int = 500) where T<:Real

    # set up.
    D = length(x_ranges)
    @assert (length(c_array) == length(θ_array) == D) | (length(c_array) == length(θ_array) == D-1)

    D_proxy = length(c_array)

    # allocate output.
    q_array = Vector{Function}(undef, D)
    updatebuffer_dg_array = Vector{Function}(undef, D)
    dg_array = Vector{Function}(undef, D)
    d2g_array = Vector{Function}(undef, D)

    fq_array = Vector{Function}(undef, D)
    ∂f_∂x_array = Vector{Vector{Function}}(undef, D)
    ∂2f_∂x2_array = Vector{Matrix{Function}}(undef, D)

    for d = 1:D_proxy

        q, updatebuffer_dg,
            dg,
            d2g,
            fq_v0 = setupallKRcomponentnonadaptive(c_array[d],
                                𝓧_array[d],
                                θ_array[d].a,
                                x_ranges[d][1],
                                x_ranges[d][end],
                                d,
                                N_nodes_tanh,
                                m_tsq;
                                max_numerical_inverse_iters = quantile_max_iters)
        #
        q_array[d] = q
        updatebuffer_dg_array[d] = updatebuffer_dg
        dg_array[d] = dg
        d2g_array[d] = d2g
        fq_array[d] = fq_v0
        #∂f_∂x_array[d] = ∂fd_∂x_array
        #∂2f_∂x2_array[d] = ∂2fd_∂x2_array
    end

    if D_proxy != D
        #  we didn't fit a proxy for hte last dimension.

        q, updatedfbuffer, dg, d2g, f_v, ∂fd_∂x_array,
            ∂2fd_∂x2_array = setuplastKRcomponentusingf( f_target,
                        df_target,
                        d2f_target,
                        x_ranges[D][1],
                        x_ranges[D][end],
                        D,
                        N_nodes_tanh,
                        m_tsq;
                        max_numerical_inverse_iters = quantile_max_iters,
                        max_integral_evals = max_integral_evals,
                        initial_divisions = initial_divisions)
        #
        q_array[D] = q
        updatebuffer_dg_array[D] = updatedfbuffer
        fq_array[D] = f_v
        dg_array[D] = dg
        d2g_array[D] = d2g
        ∂f_∂x_array[D] = ∂fd_∂x_array
        ∂2f_∂x2_array[D] = ∂2fd_∂x2_array


    end

    return q_array, updatebuffer_dg_array, dg_array, d2g_array,
        fq_array, ∂f_∂x_array, ∂2f_∂x2_array
end
