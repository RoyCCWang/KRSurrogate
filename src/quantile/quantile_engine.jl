


function setupquantileintialsearch( limit_a::T,
                                    limit_b::T,
                                    𝑓::Function,
                                    N_nodes_tanh::Int = 200,
                                    m_tsq::Int = 7)::Function where T <: Real

    # get sinh-tanh nodes and weights.
    x_tsq, w_tsq = gettanhsinhquad(N_nodes_tanh, m_tsq, one(T))

    # modify the nodes for the set [limit_a, limit_b].
    x_tsq1 = preparextsq(x_tsq, limit_a, limit_b)

    # set up intermediate buffers.
    out_cdf = Vector{T}(undef, N_nodes_tanh)
    out_f = Vector{T}(undef, N_nodes_tanh)

    q_initial = yy->estimatequantileviatsq!(out_cdf, out_f, yy,
                    x_tsq1, w_tsq, 𝑓, m_tsq, limit_a, limit_b)

    return q_initial
end

function evalquantileviaTaylor(   y_target::T,
                                        q_initial::Function,
                                        F::Function,
                                        f::Function,
                                        err_tol::T,
                                        𝑐::Vector{T},
                                        𝑑::Vector{T},
                                        update𝑐::Function,
                                        update𝑑::Function,
                                        max_traversals::Int,
                                        N_predictive_traversals::Int,
                                        correction_epoch::Int,
                                        max_iters::Int,
                                        zero_tol::T,
                                        n_limit::Int,
                                        n0::Int)::Tuple{T,T,Int} where T <: Real
    # initial search.
    x_next = q_initial(y_target)
    y_next = F(x_next)

    # refine.
    n_iters = 1
    epoch_counter = 1
    while n_iters <= max_iters && abs(y_target - y_next) > zero_tol
        x_next, y_next = evalquantileviaTayloroneiter(y_target,
                                    x_next,
                                    y_next,
                                    F,
                                    f,
                                    err_tol,
                                    𝑐,
                                    𝑑,
                                    update𝑐,
                                    update𝑑,
                                    max_traversals,
                                    N_predictive_traversals,
                                    n_limit,
                                    n0)

        if epoch_counter > correction_epoch
            y_next = F(x_next)
            epoch_counter = 1
        end
        n_iters += 1
    end

    # # final refinement where we force y_next to be numerically integrated.
    # x0 = x_next
    # y0 = F(x_next)
    # ϵ = y_target-y0
    # x_next, y_next = computeTaylorinverse(  x0,
    #                                 y0,
    #                                 update𝑐,
    #                                 update𝑑,
    #                                 f,
    #                                 F,
    #                                 𝑐,
    #                                 𝑑,
    #                                 f(x0),
    #                                 err_tol,
    #                                 sign(ϵ),
    #                                 ϵ,
    #                                 n_limit,
    #                                 n0)

    return x_next, y_next, n_iters
end

function evalquantileviaTayloroneiter(   y_target::T,
                                        x0::T,
                                        y0::T,
                                        F::Function,
                                        f::Function,
                                        err_tol::T,
                                        𝑐::Vector{T},
                                        𝑑::Vector{T},
                                        update𝑐::Function,
                                        update𝑑::Function,
                                        max_traversals::Int,
                                        N_predictive_traversals::Int,
                                        n_limit::Int,
                                        n0::Int)::Tuple{T,T} where T <: Real

    # update quantities.

    ϵ = y_target - y0
    dir_var = sign(ϵ)

    x_next, y_next = computeTaylorinverse(  x0,
                                    y0,
                                    update𝑐,
                                    update𝑑,
                                    f,
                                    F,
                                    𝑐,
                                    𝑑,
                                    f(x0),
                                    err_tol,
                                    dir_var,
                                    ϵ,
                                    n_limit,
                                    n0)

    return x_next, y_next
end

# function evalquantileviaNewtonTaylor(   y_target::T,
#                                         q_initial::Function,
#                                         F::Function,
#                                         f::Function,
#                                         ∂f_∂x::Function,
#                                         err_tol::T,
#                                         𝑐::Vector{T},
#                                         𝑑::Vector{T},
#                                         update𝑐::Function,
#                                         update𝑑::Function,
#                                         limit_a::T,
#                                         limit_b::T,
#                                         max_traversals::Int,
#                                         N_predictive_traversals::Int,
#                                         max_iters::Int,
#                                         zero_tol::T,
#                                         n_limit::Int,
#                                         n0::Int)::Tuple{T,T,Int} where T <: Real
#     # initial search.
#     x_next = q_initial(y_target)
#     y_next = F(x_next)
#
#     # refine.
#     n_iters = 1
#     while n_iters <= max_iters && abs(y_target - y_next) > zero_tol
#         x_next, y_next = evalquantileviaNewtonTayloroneiter(y_target,
#                                     x_next,
#                                     y_next,
#                                     F,
#                                     f,
#                                     ∂f_∂x,
#                                     err_tol,
#                                     𝑐,
#                                     𝑑,
#                                     update𝑐,
#                                     update𝑑,
#                                     limit_a,
#                                     limit_b,
#                                     max_traversals,
#                                     N_predictive_traversals,
#                                     n_limit,
#                                     n0)
#         #println("x_next = ", x_next, ", F(x_next) = ", F(x_next), ", y_next = ", y_next)
#         n_iters += 1
#     end
#
#     # # final refinement where we force y_next to be numerically integrated.
#     # x0 = x_next
#     # y0 = F(x_next)
#     # ϵ = y_target-y0
#     # x_next, y_next = computeTaylorinverse(  x0,
#     #                                 y0,
#     #                                 update𝑐,
#     #                                 update𝑑,
#     #                                 f,
#     #                                 F,
#     #                                 𝑐,
#     #                                 𝑑,
#     #                                 f(x0),
#     #                                 err_tol,
#     #                                 sign(ϵ),
#     #                                 ϵ,
#     #                                 n_limit,
#     #                                 n0)
#
#     return x_next, y_next, n_iters
# end
#
# # y is the target that we want to evaluate the quantile at.
# # The answer, x, is assumed to be ∈ [limit_a, limit_b].
# # f is the PDF, F is the CDF.
# # no damping.
# # formula for the object function and its derivatives. y is y_target.
# # costfunc = xx->(F(xx)-y)^2
# # ∂1costfunc = 2*(F(xx)-y)*f(xx)
# # ∂2costfunc = 2*( f(xx)^2 + (F(xx)-y)*∂f_∂x(xx, dir_var) )
# function evalquantileviaNewtonTayloroneiter(   y_target::T,
#                                         x0::T,
#                                         y0::T,
#                                         F::Function,
#                                         f::Function,
#                                         ∂f_∂x::Function,
#                                         err_tol::T,
#                                         𝑐::Vector{T},
#                                         𝑑::Vector{T},
#                                         update𝑐::Function,
#                                         update𝑑::Function,
#                                         limit_a::T,
#                                         limit_b::T,
#                                         max_traversals::Int,
#                                         N_predictive_traversals::Int,
#                                         n_limit::Int,
#                                         n0::Int)::Tuple{T,T} where T <: Real
#
#
#
#     # update quantities.
#     #y0 = F(x0)
#     f_x0 = f(x0)
#
#     ϵ = y_target - y0
#     dir_var = sign(ϵ)
#
#     ∂1y0 = f_x0 # copy for readability.
#     ∂2y0 = ∂f_∂x(x0, dir_var)
#
#     # allocate for scope reasons.
#     status_Newton = false
#     x_next = NaN
#     y_next = NaN
#
#     # Second derivative test to decide whether to use Newton step.
#     if ∂2y0 > 0
#         status_Newton, x_next = computeNewtonstep(y0, ∂1y0, ∂2y0, x0, y_target, limit_a, limit_b, dir_var)
#     end
#
#     if status_Newton
#         ### use Newton step.
#         # traverse or evaluate to proposed location.
#         x_next, y_next = simpletraverse!(   𝑐,
#                                     f_x0,
#                                     N_predictive_traversals,
#                                     update𝑐,
#                                     err_tol,
#                                     x0,
#                                     y0,
#                                     x_next,
#                                     f,
#                                     F,
#                                     max_traversals,
#                                     y_target,
#                                     n_limit,
#                                     n0)
#
#         return x_next, y_next
#     end
#
#     ### use Taylor inverse.
#     x_next, y_next = computeTaylorinverse(  x0,
#                                     y0,
#                                     update𝑐,
#                                     update𝑑,
#                                     f,
#                                     F,
#                                     𝑐,
#                                     𝑑,
#                                     f_x0,
#                                     err_tol,
#                                     dir_var,
#                                     ϵ,
#                                     n_limit,
#                                     n0)
#
#     return x_next, y_next
# end
#
# # y is y_target.
# function computeNewtonstep(y0::T, ∂1y0::T, ∂2y0::T, x0::T, y::T, limit_a::T, limit_b::T, sign_var::T)::Tuple{Bool, T} where T <: Real
#     # Newton step.
#     d2_eval = 2*( ∂1y0^2 + (y0-y)*∂2y0 )
#     d1_eval = 2*( y0 - y )*∂1y0
#
#     x_proposed = x0 -d1_eval/d2_eval
#
#
#     # reject if the proposed step is out of domain, or different directions.
#     if limit_a <= x_proposed <= limit_b && sign(x_proposed-x0)*sign_var > 0
#
#         return true, x_proposed
#     end
#
#     return false, x_proposed
# end
