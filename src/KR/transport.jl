#### front end routines for KR transport.

"""
f is the target density function.
"""
function setupKRall( c_array::Vector{Vector{T}},
                        θ_array::Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                        𝓧_array::Vector{Vector{Vector{T}}},
                        x_ranges,
                        d_ϕ_array::Vector{Function},
                        d2_ϕ_array::Vector{Function},
                                   dϕ_array,
                                   d2ϕ_array,
                        N_nodes_tanh::Int,
                        m_tsq::Int,
                        quantile_err_tol::T,
                        max_traversals::Int,
                        N_predictive_traversals::Int,
                        correction_epoch::Int,
                        quantile_max_iters::Int,
                        quantile_convergence_zero_tol::T,
                        n_limit::Int,
                        n0::Int;
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
            fq_v0,
            ∂fd_∂x_array,
            ∂2fd_∂x2_array = setupallKRcomponent(c_array[d],
                                θ_array[d],
                                𝓧_array[d],
                                x_ranges[d][1],
                                x_ranges[d][end],
                                d_ϕ_array[d],
                                d2_ϕ_array[d],
                                dϕ_array[d],
                                d2ϕ_array[d],
                                d,
                                N_nodes_tanh,
                                m_tsq,
                                quantile_err_tol,
                                max_traversals,
                                N_predictive_traversals,
                                correction_epoch,
                                quantile_max_iters,
                                quantile_convergence_zero_tol,
                                n_limit,
                                n0;
                                zero_tol = zero_tol,
                                initial_divisions = initial_divisions,
                                max_integral_evals = max_integral_evals)
        #
        q_array[d] = q
        updatebuffer_dg_array[d] = updatebuffer_dg
        dg_array[d] = dg
        d2g_array[d] = d2g
        fq_array[d] = fq_v0
        ∂f_∂x_array[d] = ∂fd_∂x_array
        ∂2f_∂x2_array[d] = ∂2fd_∂x2_array
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

"""
Sets up the quantile functions required for KR transport.
    f is the target density function.
"""
function setupKRquantiles( c_array::Vector{Vector{T}},
                        θ_array::Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                        𝓧_array::Vector{Vector{Vector{T}}},
                        max_integral_evals::Int,
                        x_ranges,
                        d_ϕ_array::Vector{Function},
                        d2_ϕ_array::Vector{Function},
                        N_nodes_tanh::Int,
                        m_tsq::Int,
                        quantile_err_tol::T,
                        max_traversals::Int,
                        N_predictive_traversals::Int,
                        correction_epoch::Int,
                        quantile_max_iters::Int,
                        quantile_convergence_zero_tol::T,
                        n_limit::Int,
                        n0::Int,
                        f::Function)::Vector{Function} where T<:Real

    # set up.
    D = length(x_ranges)
    @assert (length(c_array) == length(θ_array) == D) | (length(c_array) == length(θ_array) == D-1)

    D_proxy = length(c_array)

    # allocate output.
    q_array = Vector{Function}(undef, D)

    for d = 1:D_proxy

        q = setupquantilefordimW2(c_array[d],
                                θ_array[d],
                                𝓧_array[d],
                                max_integral_evals,
                                x_ranges[d][1],
                                x_ranges[d][end],
                                d_ϕ_array[d],
                                d2_ϕ_array[d],
                                d,
                                N_nodes_tanh,
                                m_tsq,
                                quantile_err_tol,
                                max_traversals,
                                N_predictive_traversals,
                                correction_epoch,
                                quantile_max_iters,
                                quantile_convergence_zero_tol,
                                n_limit,
                                n0)
        #
        q_array[d] = q
    end

    if D_proxy != D
        #  we didn't fit a proxy for hte last dimension.

        q_array[D] = setupquantilegenericproxy( f,
                        x_ranges[D][1],
                        x_ranges[D][end],
                        D;
                        max_numerical_inverse_iters = max_numerical_inverse_iters,
                        max_integral_evals = max_integral_evals,
                        initial_divisions = initial_divisions,
                        N_nodes_tanh = N_nodes_tanh,
                        m_tsq = m_tsq)[1]
    end

    return q_array
end

function evalKRquantiles(   u_input::Vector{T},
                            q_array::Vector{Function})::Tuple{Vector{T},Vector{T}} where T <: Real

    #
    D = length(q_array)
    @assert length(u_input) == D

    x_full = Vector{T}(undef, D)
    discrepancy = Vector{T}(undef, D)

    evalKRquantiles!(x_full, discrepancy, u_input, q_array)

    return x_full, discrepancy
end

"""
Maps a value from u_input ∈ 𝓤 := [0,1]^D to x_full ∈ 𝓨,
𝓨 is the target distribution's sample space, 𝓨.
"""
function evalKRquantiles!(x_full::Vector{T},
                        discrepancy::Vector{T},
                        u_input::Vector{T},
                        q_array::Vector{Function}) where T <: Real

    # set up.
    D = length(u_input)
    resize!(x_full, D)
    resize!(discrepancy, D)

    # pre-allocate.
    u = NaN

    for d = 1:D
        u = u_input[d]
        q = q_array[d]

        # get v.
        v = x_full[1:d-1]

        # evaluate quantile.
        𝑥, 𝑢, N_iters = q(v, u)

        # update buffer.
        x_full[d] = 𝑥
        discrepancy[d] = abs(𝑢-u)
    end

    return nothing
end

"""
Let T(x) := (f ∘ g) (x) denote the transport map,
g:ℝ^D → ℝ^D, x ↦ u,
f:ℝ^D → ℝ^D, u ↦ y.
This returns dT for source distributions that are fully factorized.
"""
function dTmapdiagonalsource(   x::Vector{T},
                                u::Vector{T},
                                y::Vector{T},
                                dg::Function,
                                dfinv::Function)::Matrix{T} where T <: Real
    #
    D = length(x)
    @assert length(u) == D

    # this is diagonal, from the assumption for this function.
    dg_x::Vector{T} = dg(x)

    # this is lower-triangular because we're doing KR transport.
    dfinv_y::Vector{Vector{T}} = dfinv(y)

    @assert length(dfinv_y) == length(dg_x) == D

    # apply the inverse function theorem.
    dfinv_y_mat = convertdFinvtolowertriangular(dfinv_y)
    df_u = inv(dfinv_y_mat)

    # # compute df_u*dg_x.
    # # Q*diagm(a) is Q with the j-th column per-element-multiplied with a[j].
    # out = zeros(T, D, D)
    # for j = 1:D
    #     for i = 1:j
    #         out[i,j] = df_u[i,j] *dg_x[j]
    #     end
    # end

    out2 = df_u*diagm(dg_x)

    out = out2
    println("norm(out-out2) = ", norm(out-out2))

    return out
end

function convertdFinvtolowertriangular(dfinv_y::Vector{Vector{T}})::Matrix{T} where T <: Real
    D = length(dfinv_y)

    dfinv_y_mat = zeros(T, D, D)
    for i = 1:D
        for j = 1:length(dfinv_y[i])
            dfinv_y_mat[i,j] = dfinv_y[i][j]
        end
    end

    return dfinv_y_mat
end

"""
Let T denote the transport map.
This returns dT for source distributions that are fully factorized.
"""
function dTmapdiagonalsource(   x::Vector{T},
                                T_map::Function,
                                g::Function,
                                dg::Function,
                                dfinv::Function )::Matrix{T} where T <: Real

    return dTmapdiagonalsource(x, g(x), T_map(x)[1], dg, dfinv)
end
