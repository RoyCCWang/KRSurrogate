
### set up functions.
# set up all persist variables, and their functions.
function setupTaylorquantilemethods(𝑐::Vector{T},
                                    𝑑::Vector{T},
                                    w_X::Vector{T},
                                    b_array::Vector{T},
                                    K_array::Vector{Vector{T}},
                                    sqrt_K_array::Vector{Vector{T}},
                                    B_array::Vector{Vector{T}},
                                    C_array::Vector{Vector{T}},
                                    A_array::Vector{Vector{T}},
                                    w2_t_array::Vector{T},
                                    P::Matrix{T},

                                θ::RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                c::Vector{T},
                                X::Vector{Vector{T}},
                                max_integral_evals::Int,
                                lower_limit::T,
                                upper_limit::T,
                                d_ϕ_map::Function,
                                d2_ϕ_map::Function,
                                D::Int,
                                N_nodes_tanh::Int,
                                m_tsq::Int,
                                quantile_err_tol::T,
                                max_traversals::Int,
                                N_predictive_traversals::Int,
                                correction_epoch::Int,
                                quantile_max_iters::Int,
                                quantile_convergence_zero_tol::T,
                                n_limit::Int,
                                n0::Int) where T <: Real

    #
    v0, Z, Taylor_multiplier, F,
        F_no_clamp_with_err, update𝑐, update𝑑, updatev,
        𝑓, ∂𝑓_∂x, fq_v0 = setupTaylormethods(𝑐, 𝑑, w_X, b_array, K_array,
                            sqrt_K_array, B_array, C_array,
                            A_array, w2_t_array, P, θ, c, X, max_integral_evals,
                            lower_limit, upper_limit, d_ϕ_map, d2_ϕ_map, D)

    ### quantile methods.
    q_v_initial = setupquantileintialsearch(lower_limit, upper_limit, 𝑓, N_nodes_tanh, m_tsq)

    q_v = yy->evalquantileviaTaylor( yy,
                                    q_v_initial,
                                    F,
                                    𝑓,
                                    quantile_err_tol,
                                    𝑐,
                                    𝑑,
                                    update𝑐,
                                    update𝑑,
                                    max_traversals,
                                    N_predictive_traversals,
                                    correction_epoch,
                                    quantile_max_iters,
                                    quantile_convergence_zero_tol,
                                    n_limit,
                                    n0)

    q = (vv,yy)->evalconditionalquantileviaTaylor(yy, vv, updatev, q_v)

    return v0, Z, Taylor_multiplier, F, F_no_clamp_with_err,
            update𝑐, update𝑑, updatev, f, ∂f_∂x, q_v_initial, q_v, q, fq_v0
end

function evalconditionalquantileviaTaylor(y::T, v::Vector{T}, updatev::Function, q_v::Function)::Tuple{T,T,Int} where T <: Real
    updatev(v)

    return q_v(y)
end

# knot_zero_tol is the radius of the epsilon-neighbourhood around a knot of
#   the warpmap where we force the derivative to be on the same side of
#   the knot as the direction of traversal.
function setupTaylormethods(𝑐::Vector{T},
                                    𝑑::Vector{T},
                                    w_X::Vector{T},
                                    b_array::Vector{T},
                                    K_array::Vector{Vector{T}},
                                    sqrt_K_array::Vector{Vector{T}},
                                    B_array::Vector{Vector{T}},
                                    C_array::Vector{Vector{T}},
                                    A_array::Vector{Vector{T}},
                                    w2_t_array::Vector{T},
                                    P::Matrix{T},

                                θ::RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                c::Vector{T},
                                X::Vector{Vector{T}},
                                max_integral_evals::Int,
                                lower_limit::T,
                                upper_limit::T,
                                d_ϕ_map::Function,
                                d2_ϕ_map::Function,
                                D::Int,
                                knot_zero_tol::T = zero(T)) where T <: Real

    ### persistant variables.
    v0::Vector{T} = ones(T,D-1)

    Z::Vector{T} = ones(T,1)
    Taylor_multiplier::Vector{T} = ones(T,1)
    x_full::Vector{T} = [v0; zero(T)]

    ### functions that are static, given RKHS solution..
    ϕ = (vv,aa)->θ.warpfunc([vv;aa])
    ϕ1 = (vv, aa, dir_varr)->d_ϕ_map([vv;aa], dir_varr, knot_zero_tol)
    ϕ2 = (vv, aa, dir_varr)->d2_ϕ_map([vv;aa], dir_varr, knot_zero_tol)

    #fq_a = (vv,xx)->evalquery([vv; xx], η_a.c, η_a.X, η_a.θ)
    fq_multiplier = sqrt(θ.canonical_params.a)^3
    fq_a = (vv,xx)->evalqueryRQ!(x_full, xx, c, X, θ.warpfunc, b_array, w_X, fq_multiplier)

    ### functions that use the persistant variables.

    # warp functions derivatives wrt the last dimension.
    w = xx->ϕ(v0,xx)
    w1 = (xx, dir_varr)->ϕ1(v0, xx, dir_varr)
    w2 = (xx, dir_varr)->ϕ2(v0, xx, dir_varr)

    # conditional density, unnormalized.
    fq_v0 = xx->fq_a(v0,xx)

    # conditional density, normalized.
    f = xx->fq_a(v0,xx)/Z[1]

    # derivative of f wrt. x.
    ∂f_∂x = (xx,dir_varr)->eval∂querydensitywrtx(xx, v0, X, c, θ, w1, w2,
                    b_array, w_X, dir_varr)*sqrt(θ.canonical_params.a)^3/Z[1]

    # conditional CDF, normalized.
    F = xx->clamp(evalcdfviaHCubature(f, max_integral_evals, lower_limit, xx)[1], zero(T), one(T))
    F_no_clamp_with_err = xx->evalcdfviaHCubature(f, max_integral_evals, lower_limit, xx)


    # method for getting the Taylor coefficients of the CDF.
    update𝑐 = (xx, dir_varr)->evalRQTaylorcoeffsquery!(   𝑐,
                                            K_array,
                                            sqrt_K_array,
                                            B_array,
                                            C_array,
                                            A_array,
                                            w2_t_array,
                                        xx,
                                             v0,
                                             X,
                                             c,
                                             w,
                                             w1,
                                             w2,
                                             b_array,
                                             w_X,
                                             Taylor_multiplier,
                                             f,
                                        dir_varr)

    # method for getting the Taylor coefficients of the quantile.
    N_Taylor_terms = length(𝑐)
    update𝑑 = xx->computeTaylorinversederivativeswfactorial!(𝑑, P, 𝑐, N_Taylor_terms)

    # method for updating persist variables and buffers given a new value for v0.
    updatev = vv->updateTaylorhelpersv0!(   b_array,
                                            v0,
                                            x_full,
                                            Z,
                                            Taylor_multiplier,
                                            fq_v0,
                                            vv,
                                            θ,
                                            X,
                                            lower_limit,
                                            upper_limit,
                                            max_integral_evals)



    return v0, Z, Taylor_multiplier, F, F_no_clamp_with_err, update𝑐, update𝑑, updatev,
            f, ∂f_∂x, fq_v0
end


# updates b_array, v0, Z, and Taylor_multiplier.
function updateTaylorhelpersv0!(b_array::Vector{T},
                                v0::Vector{T},
                                x_full::Vector{T},
                                Z::Vector{T},
                                Taylor_multiplier::Vector{T},
                                fq_v0::Function,
                                v0_in::Vector{T},
                                θ::RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}},
                                X::Vector{Vector{T}},
                                lower_limit::T,
                                upper_limit::T,
                                max_integral_evals::Int)::Nothing where T <: Real

    ### updates.
    @assert length(v0_in) == length(v0)
    v0[:] = v0_in
    x_full[1:end-1] = v0_in

    D_m_1 = length(v0)
    for n = 1:length(b_array)
        b_array[n] = θ.canonical_params.a + norm(v0 - X[n][1:D_m_1])^2
    end

    # update normalizing constant.
    val_Z, err_Z = HCubature.hcubature(fq_v0, [lower_limit], [upper_limit];
                                        norm = norm, rtol = sqrt(eps(T)),
                                        atol = 0, maxevals = max_integral_evals, initdiv = 1)

    Z[1] = val_Z

    # update Taylor multiplier.
    Taylor_multiplier[1] = (sqrt(θ.canonical_params.a)^3)/val_Z

    # # exit if Z is too ill-conditioned. Need to use BigFloat.
    # if !(1e-50 < Z < 1e50)
    #     return false
    # end

    return nothing
end

function updatewarpfuncevals!(  w_X::Vector{T},
                        warpfunc::Function,
                        X::Vector{Vector{T}},
                        D::Int)::Nothing where T <: Real

    for n = 1:length(X)
        w_X[n] = warpfunc(X[n][1:D])
    end

    return nothing
end

# regardless of d_select.
function setupTaylorquantilebuffers(N::Int, dummy::T) where T <: Real

    # misc.
    N_Taylor_terms = 11
    𝑐 = Vector{T}(undef, N_Taylor_terms)
    𝑑 = Vector{T}(undef, N_Taylor_terms)
    P = Matrix{T}(undef, N_Taylor_terms-1, N_Taylor_terms)

    # objects that change according to the RKHS solution.
    w_X = Vector{T}(undef, N)

    # quantities that change according to v0.
    b_array = Vector{T}(undef, N)

    # objects that change according to x.
    K_array = Vector{Vector{T}}(undef, N)
    sqrt_K_array = Vector{Vector{T}}(undef, N)
    B_array = Vector{Vector{T}}(undef, N)
    C_array = Vector{Vector{T}}(undef, N)
    A_array = Vector{Vector{T}}(undef, N)

    for n = 1:N
        K_array[n] = Vector{T}(undef,23)
        sqrt_K_array[n] = Vector{T}(undef,23)
        B_array[n] = Vector{T}(undef, 10)
        C_array[n] = Vector{T}(undef, 5)
        A_array[n] = Vector{T}(undef, 3)
    end
    w2_t_array = Vector{T}(undef,7)

    return  K_array,
            sqrt_K_array,
            B_array,
            C_array,
            A_array,
            w2_t_array,
            b_array,
            w_X,
            P,
            𝑐,
            𝑑
end
