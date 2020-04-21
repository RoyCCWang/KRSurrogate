function computedgcomponent!( integral_∂f_∂x_from_a_to_b::Vector{T},
                                integral_∂f_∂x_from_a_to_x::Vector{T},
                                h_x_out::Vector{T},
                                f_v_x_out::Vector{T},
                            x::T,
                            f_v::Function,
                            Z_v_in::Vector{T},
                            ∂f_∂x_array::Vector{Function},
                            a::T,
                            b::T,
                            d::Int;
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 100000)::Vector{T} where T <: Real

    #
    # read precomputed Z(v).
    Z_v = Z_v_in[1]

    dg_dx = Vector{T}(undef, d)
    fill!(dg_dx, -1.23)

    # outputs, for caching.
    resize!(integral_∂f_∂x_from_a_to_b, d-1)
    resize!(integral_∂f_∂x_from_a_to_x, d-1)
    for i = 1:d-1
        integral_∂f_∂x_from_a_to_b[i] = evalintegral(∂f_∂x_array[i], a, b;
                                            initial_divisions = initial_divisions,
                                            max_integral_evals = max_integral_evals)
        integral_∂f_∂x_from_a_to_x[i] = evalintegral(∂f_∂x_array[i], a, x;
                                            initial_divisions = initial_divisions,
                                            max_integral_evals = max_integral_evals)
    end

    h_x = evalintegral(f_v, a, x;
                    initial_divisions = initial_divisions,
                    max_integral_evals = max_integral_evals)
    #
    h_x_out[1] = h_x

    # terms that do not involve component d.
    for i = 1:d-1

        ∂h_x = integral_∂f_∂x_from_a_to_x[i]

        ∂Z_v = integral_∂f_∂x_from_a_to_b[i]

        numerator = ∂h_x*Z_v - h_x*∂Z_v
        denominator = Z_v^2

        dg_dx[i] = numerator/denominator
    end

    # terms involving component d.
    f_v_x = f_v(x)
    f_v_x_out[1] = f_v_x
    dg_dx[end] = f_v_x / Z_v

    return dg_dx
end


function computedgcomponent!( integral_∂f_∂x_from_a_to_b::Vector{T},
                                integral_∂f_∂x_from_a_to_x::Vector{T},
                                h_x_out::Vector{T},
                                f_v_x_out::Vector{T},
                            x::T,
                            f_v::Function,
                            fetchZ::Function,
                            ∂f_∂x_array::Vector{Function},
                            a::T,
                            b::T,
                            d::Int;
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 100000)::Vector{T} where T <: Real

    #
    # read precomputed Z(v).
    Z_v::T = fetchZ(NaN)

    dg_dx = Vector{T}(undef, d)
    fill!(dg_dx, -1.23)

    # outputs, for caching.
    resize!(integral_∂f_∂x_from_a_to_b, d-1)
    resize!(integral_∂f_∂x_from_a_to_x, d-1)
    for i = 1:d-1
        integral_∂f_∂x_from_a_to_b[i] = evalintegral(∂f_∂x_array[i], a, b;
                                            initial_divisions = initial_divisions,
                                            max_integral_evals = max_integral_evals)
        integral_∂f_∂x_from_a_to_x[i] = evalintegral(∂f_∂x_array[i], a, x;
                                            initial_divisions = initial_divisions,
                                            max_integral_evals = max_integral_evals)
    end

    h_x = evalintegral(f_v, a, x;
                    initial_divisions = initial_divisions,
                    max_integral_evals = max_integral_evals)
    #
    h_x_out[1] = h_x

    # terms that do not involve component d.
    for i = 1:d-1

        ∂h_x = integral_∂f_∂x_from_a_to_x[i]

        ∂Z_v = integral_∂f_∂x_from_a_to_b[i]

        numerator = ∂h_x*Z_v - h_x*∂Z_v
        denominator = Z_v^2

        dg_dx[i] = numerator/denominator
    end

    # terms involving component d.
    f_v_x = f_v(x)
    f_v_x_out[1] = f_v_x
    dg_dx[end] = f_v_x / Z_v

    return dg_dx
end
