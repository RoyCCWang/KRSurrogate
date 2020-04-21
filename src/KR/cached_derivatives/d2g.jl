


function setup∂2f∂x2!( x_persist::Vector{T},
                        v::Vector{T},
                        X::Vector{Vector{T}},
                        c::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        d2ϕ::Function,
                        a::T,
                        D::Int,
                        w_X::Vector{T},
                        b_array::Vector{T}) where T
    #D = length(X[1])
    #@assert length(x_persist) == length(v) + 1

    x_persist[1:end-1] = v
    x_persist[end] = one(T)
    #x_persist::Vector{T} = [v; one(T)]

    multiplier_value = 3*sqrt(a)^3

    ∂2f_∂x2_array = Matrix{Function}(undef, D, D)
    for i = 1:D
        for j = 1:D
            #buffer_ij = x_persist #copy(x_persist)
            #buffer_ij = x_persist #copy(x_persist)

            # ∂2f_∂x2_array[i,j] = xx->eval∂2fwrt∂xi∂xj!(buffer_ij, xx, X, c,
            #                         ϕ, dϕ, d2ϕ, a, i, j)
            ∂2f_∂x2_array[i,j] = xx->eval∂2fwrt∂xi∂xjfast!(x_persist, xx, X, c,
                                    ϕ, dϕ, d2ϕ, a, i, j, w_X, b_array, multiplier_value)
        end
    end

    return ∂2f_∂x2_array#, x_persist
end

## I think this isn't used anymore. bugged for D > 2. Use the fetchZ version.
# function computed2gcomponent(    x::T,
#                             f_v::Function,
#                             Z_v_persist::Vector{T},
#                             ∂f_∂x_array::Vector{Function},
#                             ∂2f_∂x2_array::Matrix{Function},
#                             a::T,
#                             b::T,
#                             d::Int,
#                             integral_∂f_∂x_from_a_to_b::Vector{T},
#                             integral_∂f_∂x_from_a_to_x::Vector{T},
#                             h_x_in::Vector{T},
#                             f_v_x_in::Vector{T};
#                             zero_tol::T = eps(T)*2,
#                             initial_divisions::Int = 1,
#                             max_integral_evals::Int = 100000)::Matrix{T} where T <: Real
#
#     #D = length(∂f_∂x_array)
#     Z_v = Z_v_persist[1]
#
#     d2g_dx = Matrix{T}(undef, d, d)
#     fill!(d2g_dx, -Inf)
#
#     ### common objects.
#     #h_x::T = evalintegral(f_v, a, x)
#     h_x::T = h_x_in[1]
#
#     obj_B::T = Z_v^2
#
#     #f_v_x::T = f_v(x)
#     f_v_x::T = f_v_x_in[1]
#
#     #integral_∂f_∂x_from_a_to_b::Vector{T} = collect( evalintegral(∂f_∂x_array[i], a, b) for i = 1:d-1)
#     #integral_∂f_∂x_from_a_to_x::Vector{T} = collect( evalintegral(∂f_∂x_array[i], a, x) for i = 1:d-1)
#
#
#     ### main routine.
#     # # old, working
#     # # terms that do not involve component d.
#     # for i = 1:d-1
#     #
#     #     #numerator_j = ∂h_xj*Z_v - h_x*∂Z_vj
#     #
#     #     ∂h_xi = evalintegral(∂f_∂x_array[i], a, x)
#     #     ∂Z_vi = evalintegral(∂f_∂x_array[i], a, b)
#     #
#     #     obj_A = ∂h_xi*Z_v - h_x*∂Z_vi
#     #
#     #     for j = 1:d-1
#     #     #for j = 1:i
#     #
#     #         ∂Awrt∂vj = eval∂Awrt∂vj(  x,
#     #                                 f_v,
#     #                                 Z_v,
#     #                                 ∂f_∂x_array,
#     #                                 ∂2f_∂x2_array,
#     #                                 a,
#     #                                 b,
#     #                                 i,
#     #                                 j)
#     #         #
#     #
#     #         ∂Bwrt∂vj = eval∂Bwrt∂vj( Z_v,
#     #                                 ∂f_∂x_array,
#     #                                 a,
#     #                                 b,
#     #                                 j)
#     #         #
#     #         numerator = ∂Awrt∂vj*obj_B - obj_A*∂Bwrt∂vj
#     #         denominator = obj_B^2
#     #
#     #         #Printf.@printf("processed (%d,%d)\n", i, j)
#     #         d2g_dx[i,j] = numerator/denominator
#     #
#     #     end
#     # end
#     #
#     # d2g_dx0 = copy(d2g_dx)
#     fill!(d2g_dx, -Inf)
#
#     for i = 1:d-1
#
#         #numerator_j = ∂h_xj*Z_v - h_x*∂Z_vj
#
#         ∂h_xi = integral_∂f_∂x_from_a_to_x[i]
#         ∂Z_vi = integral_∂f_∂x_from_a_to_b[i]
#
#         obj_A = ∂h_xi*Z_v - h_x*∂Z_vi
#
#         #for j = 1:d-1
#         for j = 1:i
#
#             ∂Awrt∂vj = eval∂Awrt∂vjfast(  x,
#                                     h_x,
#                                     Z_v,
#                                     ∂2f_∂x2_array,
#                                     a,
#                                     b,
#                                     i,
#                                     j,
#                                     integral_∂f_∂x_from_a_to_b,
#                                     integral_∂f_∂x_from_a_to_x;
#                                     zero_tol = zero_tol,
#                                     initial_divisions = initial_divisions,
#                                     max_integral_evals = max_integral_evals)
#             #
#
#             ∂Bwrt∂vj = eval∂Bwrt∂vjfast( Z_v, integral_∂f_∂x_from_a_to_b[j] )
#             #
#             numerator = ∂Awrt∂vj*obj_B - obj_A*∂Bwrt∂vj
#             denominator = obj_B^2
#
#             #Printf.@printf("processed (%d,%d)\n", i, j)
#             d2g_dx[i,j] = numerator/denominator
#
#         end
#     end
#
#     for j = 1:d-2
#         for i = j+1:d-1
#             d2g_dx[i,j] = d2g_dx[j,i]
#         end
#     end
#
#
#     # terms involving component d.
#     for i = 1:d-1
#         ∂f_∂vi = ∂f_∂x_array[i](x)
#         ∂Z_∂vi = integral_∂f_∂x_from_a_to_b[i]
#
#         numerator = ∂f_∂vi*Z_v - f_v_x*∂Z_∂vi
#
#         value = numerator / obj_B
#         d2g_dx[d,i] = value
#         d2g_dx[i,d] = value
#     end
#
#     # singleton term.
#     d2g_dx[end] = ∂f_∂x_array[d](x) / Z_v
#
#     println("d2g_dx = ", d2g_dx)
#     @assert all(isfinite.(d2g_dx))
#     #@assert 555==5
#
#     return d2g_dx
# end



function eval∂Awrt∂vjfast(  x::T,
                        h_x::T,
                        Z_v::T,
                        ∂2f_∂x2_array::Matrix{Function},
                        a::T,
                        b::T,
                        i::Int,
                        j::Int,
                        integral_∂f_∂x_from_a_to_b::Vector{T},
                        integral_∂f_∂x_from_a_to_x::Vector{T};
                        zero_tol::T = eps(T)*2,
                        initial_divisions::Int = 1,
                        max_integral_evals::Int = 100000)::T where T <: Real



    ∂h_∂xi = integral_∂f_∂x_from_a_to_x[i]
    ∂Z_∂xi = integral_∂f_∂x_from_a_to_b[i]

    ∂h_∂xj = integral_∂f_∂x_from_a_to_x[j]
    ∂Z_∂xj = integral_∂f_∂x_from_a_to_b[j]

    ∂2h_∂xi∂xj = evalintegral(∂2f_∂x2_array[i,j], a, x)
    ∂2Z_∂xi∂xj = evalintegral(∂2f_∂x2_array[i,j], a, b)

    term1 = ∂2h_∂xi∂xj * Z_v
    term2 = ∂h_∂xi * ∂Z_∂xj
    term3 = -∂h_∂xj * ∂Z_∂xi
    term4 = -h_x*∂2Z_∂xi∂xj

    return term1 + term2 + term3 + term4
end


function eval∂Bwrt∂vjfast(  Z_v::T, ∂Z_∂xj::T)::T where T <: Real

    return 2 * Z_v * ∂Z_∂xj
end


function computed2gcomponent(    x::T,
                            f_v::Function,
                            fetchZ::Function,
                            ∂f_∂x_array::Vector{Function},
                            ∂2f_∂x2_array::Matrix{Function},
                            a::T,
                            b::T,
                            d::Int,
                            integral_∂f_∂x_from_a_to_b::Vector{T},
                            integral_∂f_∂x_from_a_to_x::Vector{T},
                            h_x_in::Vector{T},
                            f_v_x_in::Vector{T};
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 100000)::Matrix{T} where T <: Real

    #D = length(∂f_∂x_array)
    Z_v::T = fetchZ(NaN)

    d2g_dx = Matrix{T}(undef, d, d)
    fill!(d2g_dx, -Inf)

    ### common objects.
    #h_x::T = evalintegral(f_v, a, x)
    h_x::T = h_x_in[1]

    obj_B::T = Z_v^2

    #f_v_x::T = f_v(x)
    f_v_x::T = f_v_x_in[1]

    #integral_∂f_∂x_from_a_to_b::Vector{T} = collect( evalintegral(∂f_∂x_array[i], a, b) for i = 1:d-1)
    #integral_∂f_∂x_from_a_to_x::Vector{T} = collect( evalintegral(∂f_∂x_array[i], a, x) for i = 1:d-1)


    ## main routine.

    # # old, working
    # # terms that do not involve component d.
    # for i = 1:d-1
    #
    #     #numerator_j = ∂h_xj*Z_v - h_x*∂Z_vj
    #
    #     ∂h_xi = evalintegral(∂f_∂x_array[i], a, x)
    #     ∂Z_vi = evalintegral(∂f_∂x_array[i], a, b)
    #
    #     obj_A = ∂h_xi*Z_v - h_x*∂Z_vi
    #
    #     for j = 1:d-1
    #     #for j = 1:i
    #
    #         ∂Awrt∂vj = eval∂Awrt∂vj(  x,
    #                                 f_v,
    #                                 Z_v,
    #                                 ∂f_∂x_array,
    #                                 ∂2f_∂x2_array,
    #                                 a,
    #                                 b,
    #                                 i,
    #                                 j)
    #         #
    #
    #         ∂Bwrt∂vj = eval∂Bwrt∂vj( Z_v,
    #                                 ∂f_∂x_array,
    #                                 a,
    #                                 b,
    #                                 j)
    #         #
    #         numerator = ∂Awrt∂vj*obj_B - obj_A*∂Bwrt∂vj
    #         denominator = obj_B^2
    #
    #         #Printf.@printf("processed (%d,%d)\n", i, j)
    #         d2g_dx[i,j] = numerator/denominator
    #
    #     end
    # end
    # d2g_dx0 = copy(d2g_dx)

    fill!(d2g_dx, -Inf)

    for i = 1:d-1

        #numerator_j = ∂h_xj*Z_v - h_x*∂Z_vj

        ∂h_xi = integral_∂f_∂x_from_a_to_x[i]
        ∂Z_vi = integral_∂f_∂x_from_a_to_b[i]

        obj_A = ∂h_xi*Z_v - h_x*∂Z_vi

        for j = i:d-1

            ∂Awrt∂vj = eval∂Awrt∂vjfast(  x,
                                    h_x,
                                    Z_v,
                                    ∂2f_∂x2_array,
                                    a,
                                    b,
                                    i,
                                    j,
                                    integral_∂f_∂x_from_a_to_b,
                                    integral_∂f_∂x_from_a_to_x;
                                    zero_tol = zero_tol,
                                    initial_divisions = initial_divisions,
                                    max_integral_evals = max_integral_evals)
            #

            ∂Bwrt∂vj = eval∂Bwrt∂vjfast( Z_v, integral_∂f_∂x_from_a_to_b[j] )
            #
            numerator = ∂Awrt∂vj*obj_B - obj_A*∂Bwrt∂vj
            denominator = obj_B^2

            d2g_dx[i,j] = numerator/denominator

        end
    end

    for j = 1:d-2
        for i = j+1:d-1
            d2g_dx[i,j] = d2g_dx[j,i]
        end
    end

    ## debug.
    # println("old:")
    # display(d2g_dx0)
    # println("new:")
    # display(d2g_dx)
    # @assert 1==2

    # terms involving component d.
    for i = 1:d-1
        ∂f_∂vi = ∂f_∂x_array[i](x)
        ∂Z_∂vi = integral_∂f_∂x_from_a_to_b[i]

        numerator = ∂f_∂vi*Z_v - f_v_x*∂Z_∂vi

        value = numerator / obj_B
        d2g_dx[d,i] = value
        d2g_dx[i,d] = value
    end

    # singleton term.
    d2g_dx[end] = ∂f_∂x_array[d](x) / Z_v

    return d2g_dx
end
