
"""
Numerical differentiation of KR map from uniform to target.
"""
function setupTmapND(   KR_config,
                        c_array::Vector{Vector{T}},
                        θ_array,
                        𝓧_array,
                        dφ_array,
                        d2φ_array) where T <: Real

    #
    Tmap = uu->evalKR(  1,
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

    ##
    dTmap_ND = xx->Calculus.jacobian(Tmap, xx, :central)

    ## component maps.
    Tmap_components = collect( xx->Tmap(xx)[d] for d = 1:D )

    ## second-order derivaitves of the component maps.
    d2Tmap_ND = Vector{Function}(undef, D)
    for d = 1:D
        d2Tmap_ND[d] = xx->Calculus.hessian(Tmap_components[d], xx)
    end

    return Tmap, dTmap_ND, d2Tmap_ND
end

function setupGmapND(gq_array::Vector{Function},
                     CDF_array::Vector{Function})

    ## component maps.
    Gmap = xx->collect( CDF_array[d](xx[1:d]) for d = 1:D )

    ## first-order.
    dGmap_ND = Vector{Function}(undef, D)
    for d = 1:D
        dGmap_ND[d] = xx->Calculus.gradient(CDF_array[d], xx[1:d])
    end

    ## second-order.
    d2Gmap_ND = Vector{Function}(undef, D)
    for d = 1:D
        d2Gmap_ND[d] = xx->Calculus.hessian(CDF_array[d], xx[1:d])
    end

    return Gmap, dGmap_ND, d2Gmap_ND
end

function evalquery(x::T, c::Vector{T}, X::Vector{Vector{T}}, θ::KT)::T where {T,KT}

    return sum( c[n]*RKHSRegularization.evalkernel([x], X[n][end:end], θ) for n = 1:length(c) )
end

function packagefitsolution(c_array::Vector{Vector{T}},
                            θ_array,
                            𝓧_array;
                            max_integral_evals::Int = 10000,
                            initial_divisions::Int = 1) where T <: Real
    #
    D = length(c_array)
    @assert length(θ_array) == length(𝓧_array)

    gq_array = Vector{Function}(undef, D)
    gq_array[1] = xx->evalquery(xx[1], c_array[1], 𝓧_array[1], θ_array[1])
    for d = 2:D
        gq_array[d] = xx->RKHSRegularization.evalquery(xx, c_array[d], 𝓧_array[d], θ_array[d])
    end

    CDF_array = Vector{Function}(undef, D)
    CDF_array[1] = xx->evalCDFv(gq_array[1], xx, limit_a[1], limit_b[1];
                    max_integral_evals = max_integral_evals,
                    initial_divisions = initial_divisions )
    for d = 2:D
        CDF_array[d] = xx->evalCDFv(gq_array[d], xx, limit_a[d], limit_b[d];
                        max_integral_evals = max_integral_evals,
                        initial_divisions = initial_divisions )
    end

    return gq_array, CDF_array
end
