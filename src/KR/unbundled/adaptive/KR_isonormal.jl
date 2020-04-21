


function setupTmapfromisonormal2( KR_config,
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
                                initial_divisions::Int = 1 ) where T <: Real
    #### set up.
    D = length(limit_a)
    @assert length(limit_b) == D == length(σ_array) == length(μ)

    max_integral_evals = KR_config.max_integral_evals

    gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                            max_integral_evals = max_integral_evals,
                            initial_divisions = initial_divisions)

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
                                KR_config.n0)

    quantile_map = uu->evalKRquantiles(uu, q_array)

    #
    T_map = xx->quantile_map(CDF_map(xx))

    ##### first-order derivatives.

    dCDF = xx->collect1Dnormals(xx, μ, σ_array)

    dQ_via_y = yy->computedQviay(yy, dg_array, updatebuffer_dg_array)

    # chain rule.
    dT = (xx,yy)->dTmapdiagonalsource2(xx, yy, dCDF, dQ_via_y)

    #### second-order.

    d2Q_via_y = yy->evald2Q(yy, d2g_array, dQ_via_y)
    d2CDF = xx->evald2CDFisonormal(xx, μ, σ_array)

    d2T = (xx,yy)->evald2Tmap(xx, yy, dCDF, d2CDF, d2Q_via_y, dQ_via_y)

    ## stand-alone. Slow.
    d2G = xx->computed2G(gq_array,
                            xx,
                            𝓧_array,
                            c_array,
                            θ_array,
                            dϕ_array,
                            d2ϕ_array,
                            limit_a,
                            limit_b;
                            max_integral_evals = max_integral_evals)

    return T_map, dT, d2T, CDF_map, quantile_map,
        dCDF, d2CDF, dg_array, d2g_array, updatebuffer_dg_array,
        fq_v_array, gq_array, dQ_via_y, d2Q_via_y, ∂f_∂x_array, d2G,
        ∂2f_∂x2_array
end



function dTmapdiagonalsource2(   x::Vector{T},
                                y::Vector{T},
                                dCDF::Function,
                                dQ_via_y::Function)::Matrix{T} where T <: Real
    #
    D = length(x)
    @assert length(y) == D

    # this is diagonal, from the assumption for this function.

    dCDF_x::Vector{T} = dCDF(x)

    # this is lower-triangular because we're doing KR transport.
    dQ_via_y_y::Matrix{T} = dQ_via_y(y)


    out = dQ_via_y_y*diagm(dCDF_x)

    return out
end









#### legacy.
function setupTmapfromisonormal( KR_config,
                                c_array::Vector{Vector{T}},
                                θ_array,
                                𝓧_array,
                                dφ_array,
                                d2φ_array,
                                μ::Vector{T},
                                σ_array::Vector{T},
                                limit_a::Vector{T},
                                limit_b::Vector{T};
                                initial_divisions::Int = 1 ) where T <: Real
    #### set up.
    max_integral_evals = KR_config.max_integral_evals

    gq_array, CDF_array = packagefitsolution(c_array, θ_array, 𝓧_array;
                            max_integral_evals = max_integral_evals,
                            initial_divisions = initial_divisions)

    #### Transport map.
    # CDF_map: ℝ^D → [0,1]^D, via CDF of 𝓝(x | μ, diagm(σ_array.^2)).
    CDF_map = xx->transportisonormaltoCDFspace(xx, μ, σ_array)

    # quantile_map: [0,1]^D → ℝ^D, via quantile of the target distribution.
    q_array = setupKRquantiles( c_array,
                                θ_array,
                                𝓧_array,
                                max_integral_evals,
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

    quantile_map = uu->evalKRquantiles(uu, q_array)

    #
    T_map = xx->quantile_map(CDF_map(xx))

    ### first-order derivatives.
    dCDF = xx->collect1Dnormals(xx, μ, σ_array)

    dQinv = yy->computedG(   gq_array,
                        yy,
                        𝓧_array,
                        c_array,
                        θ_array,
                        dϕ_array,
                        limit_a,
                        limit_b;
                        max_integral_evals = max_integral_evals)
    # chain rule.
    dT = xx->dTmapdiagonalsource( xx, T_map, CDF_map, dCDF, dQinv)

    # faster.
    dg_array, updatev_array = setupdfinv(gq_array, 𝓧_array, c_array, θ_array,
                dϕ_array, limit_a, limit_b;
                initial_divisions = initial_divisions,
                max_integral_evals = KR_config.max_integral_evals)

    return T_map, dT,
        CDF_map, quantile_map,
        dCDF, dQinv, dg_array, updatev_array
end



function setupdfinv( gq_array::Vector{Function},
                    𝓧_array,
                    c_array,
                    θ_array,
                    dϕ_array::Vector{Function},
                    limit_a::Vector{T},
                    limit_b::Vector{T};
                    zero_tol::T = eps(T)*2,
                    initial_divisions::Int = 1,
                    max_integral_evals::Int = 100000 ) where T <: Real

    #
    D = length(limit_a)


    # component functions.
    dg_array = Vector{Function}(undef, D)
    updatev_array = Vector{Function}(undef, D)

    for d = 1:D
        fq = gq_array[d]

        # persists.
        v_persist::Vector{T} = ones(T, d-1)
        y1d_persist::Vector{T} = ones(T, d)

        ∂f_∂x_array, f_v, p_v,
            Z_v_persist = setupdgcomponent99( fq,
                                v_persist,
                                y1d_persist,
                                𝓧_array[d],
                                c_array[d],
                                θ_array[d].warpfunc,
                                dϕ_array[d],
                                θ_array[d].canonical_params.a,
                                limit_a[d],
                                limit_b[d];
                                max_integral_evals = max_integral_evals)
        #
        updatev_array[d] = (yy,dd)->updatevy1d!( v_persist,
                                                y1d_persist,
                                                Z_v_persist,
                                                f_v,
                                                yy,
                                                dd,
                                                limit_a,
                                                limit_b)
        #
        dg_array[d] = yy->computedGcomponent( yy[d],
                            f_v,
                            Z_v_persist,
                            ∂f_∂x_array,
                            limit_a[d],
                            limit_b[d],
                            d;
                            zero_tol = zero_tol,
                            initial_divisions = initial_divisions,
                            max_integral_evals = max_integral_evals)
    end

    return dg_array, updatev_array
end



function setupdgcomponent99( f::Function,
                            v::Vector{T},
                            x_full::Vector{T},
                            X::Vector{Vector{T}},
                            c::Vector{T},
                            ϕ::Function,
                            dϕ::Function,
                            a::T,
                            lower_bound::T,
                            upper_bound::T;
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 100000)::Tuple{Vector{Function},Function,Function,Vector{T}} where T <: Real

    D = length(v) + 1

    f_v = xx->f([v; xx])::T

    Z_v_persist::Vector{T} = ones(T, 1)
    p_v = xx->f_v(xx)/Z_v_persist[1]


    ∂f_∂x_array = Vector{Function}(undef, D)
    for i = 1:D
        buffer_i = copy(x_full)
        ∂f_∂x_array[i] = xx->eval∂fwrt∂xi!(buffer_i, xx, X, c,
                                ϕ, dϕ, a, i)
    end

    return ∂f_∂x_array, f_v, p_v, Z_v_persist
end
