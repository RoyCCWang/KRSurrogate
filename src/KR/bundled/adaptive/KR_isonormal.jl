

### bundled derivatives.
# the idea of bundled is you call functions as updates to persists.
# the functions never return anything you use.
# You use the persists provided to you.
function setupbundledKR( KR_config,
                            c_array::Vector{Vector{T}},
                            θ_array,
                            𝓧_array,
                            dφ_array,
                            d2φ_array,
                                       dϕ_array,
                                       d2ϕ_array,
                            μ::Vector{T},
                            σ_array::Vector{T},
                            limit_a::Vector{T},
                            limit_b::Vector{T};
                            f_target::Function = xx->xx,
                            df_target::Function = xx->xx,
                            d2f_target::Function = xx->xx,
                            initial_divisions::Int = 1 ) where T <: Real
    #### set up.
    D = length(limit_a)
    @assert length(limit_b) == D == length(σ_array) == length(μ)

    max_integral_evals = KR_config.max_integral_evals

    #### Transport map.
    # CDF_map: ℝ^D → [0,1]^D, via CDF of 𝓝(x | μ, diagm(σ_array.^2)).
    CDF_map = xx->transportisonormaltoCDFspace(xx, μ, σ_array)

    # quantile_map: [0,1]^D → ℝ^D, via quantile of the target distribution.
    q_array, updatebuffer_dg_array,
        dg_array, d2g_array, fq_v_array,
        ∂f_∂x_array,
        ∂2f_∂x2_array = setupKRall( c_array,
                                θ_array,
                                𝓧_array,
                                KR_config.x_ranges,
                                dφ_array,
                                d2φ_array,
                                           dϕ_array,
                                           d2ϕ_array,
                                KR_config.N_nodes_tanh,
                                KR_config.m_tsq,
                                KR_config.quantile_err_tol,
                                KR_config.max_traversals,
                                KR_config.N_predictive_traversals,
                                KR_config.correction_epoch,
                                KR_config.quantile_max_iters,
                                KR_config.quantile_convergence_zero_tol,
                                KR_config.n_limit,
                                KR_config.n0;
                                f_target = f_target,
                                df_target = df_target,
                                d2f_target = d2f_target,
                                max_numerical_inverse_iters = KR_config.quantile_max_iters)

    quantile_map = uu->evalKRquantiles(uu, q_array)

    # display(size(∂2f_∂x2_array[end]))
    # @assert 1==2

    T_map = xx->quantile_map(CDF_map(xx))

    ##### first-order derivatives.

    dCDF = xx->collect1Dnormals(xx, μ, σ_array)

    dQ_via_y = yy->computedQviay(yy, dg_array, updatebuffer_dg_array)

    # chain rule.
    dT = (xx,yy)->dTmapdiagonalsource2( xx, yy, dCDF, dQ_via_y)

    #### second-order.

    d2Q_via_y = yy->evald2Q(yy, d2g_array, dQ_via_y) # could be unused.
    d2CDF = xx->evald2CDFisonormal(xx, μ, σ_array)

    #d2T = (xx,yy)->evald2Tmap(xx, yy, dCDF, d2CDF, d2Q_via_y, dQ_via_y)

    ds_T_map = (xx,yy)->evalbundledderivativesTmap( xx,
                                    yy,
                                    dCDF,
                                    dQ_via_y,
                                    d2g_array,
                                    d2CDF)

    return T_map, ds_T_map, dT, quantile_map, CDF_map
end


function evalbundledderivativesTmap(x::Vector{T},
                                    y::Vector{T},
                                    dCDF::Function,
                                    dQ_via_y::Function,
                                    d2g_array::Vector{Function},
                                    d2CDF::Function)::Tuple{Matrix{T},Vector{Matrix{T}}} where T <: Real
    #
    D = length(x)
    @assert length(y) == D

    ## first-order.

    # this is diagonal, from the assumption for this function.
    dCDF_x::Matrix{T} = diagm(dCDF(x))

    # this is lower-triangular because we're doing KR transport.
    dQ_u::Matrix{T} = dQ_via_y(y)

    # chain rule.
    dT_x = dQ_u*dCDF_x


    ## second-order.
    d2Q_u = evald2Q(y, d2g_array, dQ_u)
    #d2Q_u = evald2Q(y, d2g_array, dQ_via_y)
    #d2Q_u = d2Q_via_y(y)
    d2CDF_x = d2CDF(x)

    d2T_x = applychainruled2(dCDF_x, d2CDF_x,
                             dQ_u, d2Q_u)

    return dT_x, d2T_x
end
