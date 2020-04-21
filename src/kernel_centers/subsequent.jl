

# h is preference function. It has non-negative output.
function selectsubsequentkernelcenterssequentially!( X_candidates::Vector{Vector{T}},
                            X0::Vector{Vector{T}},
                            h::Function,
                            f::Function,
                            θ_RKHS::KT;
                            zero_tol_RKHS::T = 1e-13,
                            prune_tol::T = 1.1*zero_tol_RKHS,
                            close_radius_tol::T = 1e-6,
                            max_iters_RKHS::Int = 5000,
                            σ²::T = 1e-3 ) where {T,KT}
    # find the candidate with largest h value.
    h_X_can = h.(X_candidates)
    val_max, ind_max = findmax(h_X_can)

    # remove entries that are too close.
    X_fit = copy(X0)
    push!(X_fit, X_candidates[ind_max])
    #println("X_fit[1] = ", X_fit[1])
    X_fit = removeclosepositions!(X_fit, close_radius_tol)

    # remove from candidate pool.
    #println("pre: length(X_candidates) = ", length(X_candidates))
    deleteat!(X_candidates, ind_max)
    #println("aft: length(X_candidates) = ", length(X_candidates))

    # fit.
    f_X_fit = f.(X_fit)
    c_q, 𝓧_q,
        keep_indicators_unused = fitRKHSdensity(  f_X_fit,
                                X_fit, max_iters_RKHS, σ²,
                                θ_RKHS, zero_tol_RKHS,
                                prune_tol)



    return c_q, 𝓧_q, X_fit
end

# M refinements.
function refinecenters!( X_candidates::Vector{Vector{T}},
                    X0::Vector{Vector{T}},
                    M::Int,
                    f::Function,
                    fq::Function,
                    θ_RKHS::KT,
                    X_ref;
                    zero_tol_RKHS::T = 1e-13,
                    prune_tol::T = 1.1*zero_tol_RKHS,
                    close_radius_tol::T = 1e-6,
                    max_iters_RKHS::Int = 5000,
                    σ²::T = 1e-3 ) where {T,KT}
    # check.
    if M > length(X_candidates)
        Printf.@printf("Warning: Not enough candidates for %d refinements. Going with length(X_candidates) = %d refinements instead.",
            M, length(X_candidates))
        M = length(X_candidates)
    end

    # set up query.
    fq_next = fq

    # allocate for scoping purposes.
    c_next::Vector{T} = Vector{T}(undef, 0)
    X_next::Vector{Vector{T}} = Vector{Vector{T}}(undef, 0)

    # output.
    c_history = Vector{Vector{T}}(undef, M)
    X_history = Vector{Vector{Vector{T}}}(undef, M)

    X_fit = copy(X0)

    # debug: track total norm error.
    #X_ref = copy(X_candidates)
    f_X_ref = f.(X_ref)
    error_history = -ones(T, M)
    X_fit_history = Vector{Vector{Vector{T}}}(undef, M)

    for m = 1:M
        #display(X_fit)
        h = xx->abs(f(xx)-fq_next(xx))
        c_next, X_next, X_fit_next = selectsubsequentkernelcenterssequentially!( X_candidates,
                                X_fit, # X0
                                h,
                                f,
                                θ_RKHS;
                                zero_tol_RKHS = zero_tol_RKHS,
                                prune_tol = prune_tol,
                                close_radius_tol = close_radius_tol,
                                max_iters_RKHS = max_iters_RKHS,
                                σ² = σ²)

        #
        # debug.
        #println("m = ", m)
        #println("length(X_next) = ", length(X_next))
        #println("length(X_fit_next) = ", length(X_fit_next))
        #println("length(X_candidates) = ", length(X_candidates))
        error_history[m] = norm(fq_next.(X_ref)-f_X_ref)
        X_fit_history[m] = copy(X_fit)

        # update.
        fq_next = xx->RKHSRegularization.evalquery(xx, c_next, X_next, θ_RKHS)
        c_history[m] = c_next
        X_history[m] = copy(X_next)

        # slower. plateaus.
        X_fit = X_fit_next

        # higher error in main function for some reason.
        #X_fit = X_next

        # semi-definite error for some reason.
        # push!(X_fit, X_next...)
        # removeclosepositions!(X_fit, close_radius_tol)
    end

    return c_history, X_history, error_history, X_fit_history
end
