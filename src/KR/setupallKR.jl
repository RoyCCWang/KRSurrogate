# set up the quantile for the current dimension, d_select.
"""
dϕ_map is for differentials.
d_ϕ_map is for quantile. it takes two parameters.
"""
function setupallKRcomponent( c::Vector{T},
                                θ::RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                X::Vector{Vector{T}},
                                lower_bound::T,
                                upper_bound::T,
                                d_ϕ_map::Function,
                                d2_ϕ_map::Function,
                                dϕ_map::Function,
                                d2ϕ_map::Function,
                                d_select::Int,
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
                                zero_tol::T = eps(T)*2,
                                initial_divisions::Int = 1,
                                max_integral_evals::Int = 10000) where T <: Real

    # set up given current RKHS fit of pdf for the d_select-th dimension.
    K_array, sqrt_K_array, B_array,
    C_array, A_array, w2_t_array, b_array, w_X, P, 𝑐,
        𝑑 = setupTaylorquantilebuffers(length(X), one(T)) # specifically for the RQ adaptive kernel.

    updatewarpfuncevals!(w_X, θ.warpfunc, X, d_select)

    ### quantities that need updating once the content of v0 changes.
    v0, Z, Taylor_multiplier, F, F_no_clamp_with_err, update𝑐, update𝑑, updatev, 𝑓,
        ∂𝑓_∂x, q_v_initial, q_v, q,
        fq_v0 = setupTaylorquantilemethods(  𝑐, 𝑑, w_X,
                                            b_array,
                                            K_array,
                                            sqrt_K_array,
                                            B_array,
                                            C_array,
                                            A_array,
                                            w2_t_array,
                                            P,
                                        θ,
                                        c,
                                        X,
                                        max_integral_evals,
                                        lower_bound,
                                        upper_bound,
                                        d_ϕ_map,
                                        d2_ϕ_map,
                                        d_select,
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
    ∂f_∂x_array, x_persist = setup∂f∂xfull( v0, X,
                                        c,
                                        θ.warpfunc,
                                        dϕ_map,
                                        θ.canonical_params.a,
                                        d_select,
                                        w_X,
                                        b_array)
    #
    #bma_array = Vector{T}(undef, length(b_array)) # same as b_array, but without the RQ kernel parameter a.
    updatedfbuffer = yy->updatebufferderivatives!(  b_array,
                                x_persist,
                                updatev,
                                yy,
                                θ.canonical_params.a,
                                X)

    #
    integral_∂f_∂x_from_a_to_b::Vector{T} = Vector{T}(undef, d_select-1)
    integral_∂f_∂x_from_a_to_x::Vector{T} = Vector{T}(undef, d_select-1)
    h_x_persist::Vector{T} = ones(T, 1)
    f_v_x_persist::Vector{T} = ones(T, 1)
    dg = xx->computedgcomponent!( integral_∂f_∂x_from_a_to_b,
                                    integral_∂f_∂x_from_a_to_x,
                                    h_x_persist,
                                    f_v_x_persist,
                            xx,
                            fq_v0,
                            Z,
                            ∂f_∂x_array,
                            lower_bound,
                            upper_bound,
                            d_select;
                            zero_tol = zero_tol,
                            initial_divisions = initial_divisions,
                            max_integral_evals = max_integral_evals)

    #### second-order.

    ∂2f_∂x2_array = setup∂2f∂x2!( x_persist,
                                    v0,
                                    X,
                                    c_array[d_select],
                                    θ.warpfunc,
                                    dϕ_map,
                                    d2ϕ_map,
                                    θ.canonical_params.a,
                                    d_select,
                                    w_X,
                                    b_array)
    #
    fetchZ = xx->fetchscalarpersist(Z)
    d2g = xx->computed2gcomponent(    xx,
                            fq_v0,
                            #Z,
                            fetchZ,
                            ∂f_∂x_array,
                            ∂2f_∂x2_array,
                            lower_bound,
                            upper_bound,
                            d_select,
                            integral_∂f_∂x_from_a_to_b,
                            integral_∂f_∂x_from_a_to_x,
                            h_x_persist,
                            f_v_x_persist;
                            zero_tol = zero_tol,
                            initial_divisions = initial_divisions,
                            max_integral_evals = max_integral_evals)

    return q, updatedfbuffer, dg, d2g, fq_v0, ∂f_∂x_array, ∂2f_∂x2_array
end

# function updateallKRparameters( updatev,
#                                 updatedfbuffer,
#                                 y::Vector{T})
#     v = y[]
#     updatev(v)
#     updatedfbuffer(y)
#
#     return nothing
# end


function updatebufferderivatives!(  b_array::Vector{T},
                                    x_persist::Vector{T},
                                    updatev::Function,
                                    y::Vector{T},
                                    a::T,
                                    X::Vector{Vector{T}}) where T

    d = length(y)

    resize!(b_array, length(X))

    #  update f_v and Z_v.
    v = y[1:end-1]
    updatev(v)

    # update b_array.
    D_m_1 = length(v)
    for n = 1:length(b_array)
        b_array[n] = a + norm(v - X[n][1:D_m_1])^2
    end

    # update x_persist. Used for ∂f_∂x.
    x_persist[:] = y
    #println("x_persist = ", x_persist)
    return nothing
end


# function updatederivativewarpfuncevals!(  dw_X::Vector{Vector{T}},
#                         d2w_X::Vector{Matrix{T}},
#                         dϕ::Function,
#                         d2ϕ::Function,
#                         X::Vector{Vector{T}},
#                         D::Int)::Nothing where T <: Real
#
#     for n = 1:length(X)
#         dw_X[n] = dϕ(X[n][1:D])
#         d2w_X[n] = d2ϕ(X[n][1:D])
#     end
#
#     return nothing
# end

### legacy.
"""
If y ∈ ℝ^D is the input, then fq_array[d]: y[d] ↦ f(y).
Before evaluating fq_array[d], must execute updatey_array[d](y, d),
    for each fixed d ∈ [D].
"""
function getfvarray(c_array::Vector{Vector{T}},
                    X_array::Vector{Vector{Vector{T}}},
                    θ_array;
                    zero_tol::T = 1e-8,
                    initial_divisions::Int = 1,
                    max_integral_evals::Int = 100000) where {T,KT}
    #

    fq_array = Vector{Function}(undef, D)
    updatey_array = Vector{Function}(undef, D)
    fetchZ_array = Vector{Function}(undef, D)
    fetchv_array = Vector{Function}(undef, D)
    fetchy1d_array = Vector{Function}(undef, D)

    for d = 1:D
        fq_array[d], updatey_array[d], fetchZ_array[d], fetchv_array[d],
            fetchy1d_array[d] = setupfv( c_array[d],
                                        X_array[d],
                                        θ_array[d],
                                        d,
                                        limit_a[d],
                                        limit_b[d];
                                        zero_tol = zero_tol,
                                        initial_divisions = initial_divisions,
                                        max_integral_evals = max_integral_evals)
    end

    return fq_array, updatey_array, fetchZ_array,
            fetchv_array, fetchy1d_array
end

function setupfv(   c::Vector{T},
                    X::Vector{Vector{T}},
                    θ::KT,
                    d::Int,
                    lower_limit::T,
                    upper_limit::T;
                    zero_tol::T = 1e-8,
                    initial_divisions::Int = 1,
                    max_integral_evals::Int = 100000) where {T,KT}

    # allocate b_array and w_X.
    K_array, sqrt_K_array, B_array,
    C_array, A_array, w2_t_array, b_array, w_X, P, 𝑐,
        𝑑 = setupTaylorquantilebuffers(length(X), one(T)) # specifically for the RQ adaptive kernel.

    # compute w_X.
    updatewarpfuncevals!(w_X, θ.warpfunc, X, d)



    v0::Vector{T} = ones(T, d-1) # y[1:d-1].
    y1d::Vector{T} = ones(T, d) # y[1:d].
    x_full::Vector{T} = [v0; zero(T)] # y[1:d].
    Z::Vector{T} = ones(T,1)

    #fq_a = (vv,xx)->evalquery([vv; xx], η_a.c, η_a.X, η_a.θ)
    fq_multiplier = sqrt(θ.canonical_params.a)^3
    fq_a = (vv,xx)->evalqueryRQ!(x_full, xx, c, X, θ.warpfunc, b_array, w_X, fq_multiplier)

    # conditional density, unnormalized.
    fq_v0 = xx->fq_a(v0,xx)

    # # conditional density, normalized.
    # f = xx->fq_a(v0,xx)/Z[1]
    #
    # ∂f_∂x = (xx,dir_varr)->eval∂querydensitywrtx(xx, v0, X, c, θ, w1, w2,
    #                 b_array, w_X, dir_varr)*sqrt(θ.canonical_params.a)^3/Z[1]

    # update routines.
    updatey = (yy,dd)->updatevy1d!(    b_array,
                                    v0,
                                    y1d,
                                    Z,
                                    fq_v0,
                                    x_full,
                                    yy,
                                    dd,
                                    θ,
                                    X,
                                    lower_limit,
                                    upper_limit;
                                    zero_tol = zero_tol,
                                    initial_divisions = initial_divisions,
                                    max_integral_evals = max_integral_evals)

    # read-only routines.
    fetchZ = xx->fetchscalarpersist(Z)
    fetchv = xx->fetchscalarpersist(v0)
    fetchy1d = xx->fetchscalarpersist(y1d)

    return fq_v0, updatey, fetchZ, fetchv, fetchy1d
end
