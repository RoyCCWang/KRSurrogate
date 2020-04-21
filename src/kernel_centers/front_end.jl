

function getkernelcenters(θ_canonical::KT,
                            Y_nD::Array{T,D},
                            x_ranges,
                            X_ref::Vector{Vector{T}},
                            f::Function,
                            limit_a::Vector{T},
                            limit_b::Vector{T},
                            config::KernelCenterConfigType{T}) where {T,D,KT}

    θ_a, ϕ_map_func, d_ϕ_map_func,
        d2_ϕ_map_func = setupadaptivekernel(θ_canonical,
                                            Y_nD,
                                            f,
                                            x_ranges;
                                            amplification_factor = config.amplification_factor,
                                            attenuation_factor_at_cut_off = config.attenuation_factor_at_cut_off,
                                            N_bands = config.N_bands)
    #

    # get candidates.
    X_candidates, y_unused = getinitialcandidatesviauniform(   config.N_preliminary_candidates,
                                limit_a,
                                limit_b,
                                f;
                                zero_tol = config.candidate_zero_tol,
                                truncation_factor = config.candidate_truncation_factor)

    # debug. store initial pool of fit locations.
    X_pool = copy(X_candidates)

    # first round of fit location selection.
    f_scale_factor::T = 1/maximum(Y_nD)
    DPP_base = θ_a
    additive_function = xx->f_scale_factor*f(xx)

    c1, X1,
        X_fit1 = selectkernelcenters!(  X_candidates,
                            additive_function,
                            f,
                            DPP_base,
                            θ_a;
                            max_iters_RKHS = config.max_iters_RKHS,
                            base_gain = config.base_gain,
                            kDPP_zero_tol = config.kDPP_zero_tol,
                            N_kDPP_draws = config.N_kDPP_draws,
                            N_kDPP_per_draw  = config.N_kDPP_per_draw ,
                            zero_tol_RKHS = config.zero_tol_RKHS,
                            prune_tol = config.prune_tol,
                            σ² = config.σ²)
    #
    fq1 = xx->RKHSRegularization.evalquery(xx, c1, X1, θ_a)
    f_X_ref = f.(X_ref)
    err_1 = norm(fq1.(X_ref)-f_X_ref)

    ## subsequent refinement.
    ##### refine via sequential.

    #X0 = copy(X1)
    X0 = copy(X_fit1)

    # println("length(X_candidates) = ", length(X_candidates))
    # println("length(X1) = ", length(X1))
    # println("length(X_pool) = ", length(X_pool))
    # @assert 1==2

    println("refining.")
    @time c_history, X_history,
        error_history,
        X_fit_history = refinecenters!( X_candidates,
                                X0,
                                config.N_refinements,
                                f,
                                fq1,
                                θ_a,
                                X_ref;
                                zero_tol_RKHS = config.zero_tol_RKHS,
                                prune_tol = config.prune_tol,
                                close_radius_tol = config.close_radius_tol,
                                max_iters_RKHS = config.max_iters_RKHS,
                                σ² = config.σ²)
    println()

    # add initial results.

    insert!(c_history, 1, c1)
    insert!(X_history, 1, X1)
    insert!(error_history, 1, err_1)
    insert!(X_fit_history, 1, X_fit1)

    return θ_a, c_history, X_history, error_history, X_fit_history, X_pool
end
