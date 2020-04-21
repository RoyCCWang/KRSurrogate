


function setupbundledKRnonadaptive( KR_config,
                            c_array::Vector{Vector{T}},
                            θ_array,
                            𝓧_array,
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
        ∂2f_∂x2_array = setupKRallnonadaptive( c_array,
                                θ_array,
                                𝓧_array,
                                KR_config.x_ranges,
                                KR_config.N_nodes_tanh,
                                KR_config.m_tsq,
                                KR_config.quantile_max_iters,
                                KR_config.quantile_convergence_zero_tol;
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

    return T_map, ds_T_map, dT
end
