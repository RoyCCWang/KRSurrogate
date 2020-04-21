
# f is target unnormalized density function.
function setup∂f∂xfull( grad_f::Function,
                        D::Int,
                        dummy_val::T) where T
    #D = length(X[1])

    x_persist::Vector{T} = ones(T, D)

    ∂f_∂x_array = Vector{Function}(undef, D)
    for i = 1:D

        ∂f_∂x_array[i] = xx->evalgradffast!(x_persist, xx, grad_f, i)
    end

    return ∂f_∂x_array, x_persist
end

function evalgradffast!(   x::Vector{T},
                        t::T,
                        grad_f::Function,
                        i::Int)::T where T

    #
    x[end] = t

    return grad_f(x)[i]
end


function setup∂2f∂x2!( x_persist::Vector{T}, d2f::Function) where T
    D = length(x_persist)

    ∂2f_∂x2_array = Matrix{Function}(undef, D, D)
    for j = 1:D
        for i = 1:D
            ∂2f_∂x2_array[i,j] = xx->evalhessianffast!(x_persist, xx, d2f, i, j)
        end
    end

    return ∂2f_∂x2_array
end

function evalhessianffast!(   x::Vector{T},
                        t::T,
                        hess_f::Function,
                        i::Int,
                        j::Int)::T where T

    #
    x[end] = t

    return hess_f(x)[i,j]
end

function setuplastKRcomponentusingf(f::Function,
                                    df::Function,
                                    d2f::Function,
                                    lower_bound::T,
                                    upper_bound::T,
                                    D::Int,
                                    N_nodes_tanh::Int,
                                    m_tsq::Int;
                                    max_numerical_inverse_iters = 10000,
                                    zero_tol::T = eps(T)*2,
                                    initial_divisions::Int = 1,
                                    max_integral_evals::Int = 10000) where T

    q, f_v, updatev, fetchZ = setupquantilegenericproxy( f,
                    lower_bound,
                    upper_bound,
                    D;
                    max_numerical_inverse_iters = max_numerical_inverse_iters,
                    max_integral_evals = max_integral_evals,
                    initial_divisions = initial_divisions,
                    N_nodes_tanh = N_nodes_tanh,
                    m_tsq = m_tsq)

    #
    ∂f_∂x_array, x_persist = setup∂f∂xfull(df, D, one(T))

    #
    updatedfbuffer = yy->updatebuffergenericf!( x_persist,
                                                updatev,
                                                yy)

    #
    integral_∂f_∂x_from_a_to_b::Vector{T} = Vector{T}(undef, D-1)
    integral_∂f_∂x_from_a_to_x::Vector{T} = Vector{T}(undef, D-1)
    h_x_persist::Vector{T} = ones(T, 1)
    f_v_x_persist::Vector{T} = ones(T, 1)
    dg = xx->computedgcomponent!( integral_∂f_∂x_from_a_to_b,
                                    integral_∂f_∂x_from_a_to_x,
                                    h_x_persist,
                                    f_v_x_persist,
                            xx,
                            f_v,
                            fetchZ,
                            ∂f_∂x_array,
                            lower_bound,
                            upper_bound,
                            D;
                            zero_tol = zero_tol,
                            initial_divisions = initial_divisions,
                            max_integral_evals = max_integral_evals)
    #
    #### second-order.

    ∂2f_∂x2_array = setup∂2f∂x2!( x_persist, d2f)
    #
    d2g = xx->computed2gcomponent(    xx,
                        f_v,
                        fetchZ,
                        ∂f_∂x_array,
                        ∂2f_∂x2_array,
                        lower_bound,
                        upper_bound,
                        D,
                        integral_∂f_∂x_from_a_to_b,
                        integral_∂f_∂x_from_a_to_x,
                        h_x_persist,
                        f_v_x_persist;
                        zero_tol = zero_tol,
                        initial_divisions = initial_divisions,
                        max_integral_evals = max_integral_evals)


    # println("∂2f_∂x2_array[1](1.23) = "),
    # display(∂2f_∂x2_array[1](1.23))
    # println("size(∂2f_∂x2_array)")
    # println(size(∂2f_∂x2_array))
    # @assert 1==2
    return q, updatedfbuffer, dg, d2g, f_v, ∂f_∂x_array, ∂2f_∂x2_array
end

function updatebuffergenericf!( x_persist::Vector{T},
                                updatev::Function,
                                y::Vector{T}) where T
    x_persist[:] = y

    updatev(y[1:end-1])

    return nothing
end


## this is the next speed up tool. not used.
function evalgenericfv!(x::Vector{T},
                        t::T,
                        f::Function,
                        i::Int) where T
    #
    x[end] = t

    return f(x)
end

function setupgenericfv(f::Function,
                        D::Int,
                        dummy_val::T) where T
    #
    v0::Vector{T} = ones(T,D-1)
    Z::Vector{T} = ones(T,1)

    x_persist::Vector{T}

    f = xx->evalgenericfv!(x_persist,xx)/Z[1]

end
