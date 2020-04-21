# updates Z_persist and b_array.
function updatebuffernonadaptive(updatev::Function,
                                y::Vector{T}) where T

    updatev(y[1:end-1])

    return nothing
end

function setupallKRcomponentnonadaptive(    c::Vector{T},
                                        X::Vector{Vector{T}},
                                        a_RQ::T,
                                    lower_bound::T,
                                    upper_bound::T,
                                    d_select::Int,
                                    N_nodes_tanh::Int,
                                    m_tsq::Int;
                                    max_numerical_inverse_iters = 10000) where T
    #
    N = length(c)
    @assert length(X) == N



    b_array::Vector{T} = ones(T, N)
    q, f_v, updatev, fetchZ,
    v_persist = setupquantilenonadaptive( c, X, a_RQ,
                    b_array,
                    lower_bound,
                    upper_bound,
                    d_select;
                    max_numerical_inverse_iters = max_numerical_inverse_iters,
                    max_integral_evals = max_integral_evals,
                    initial_divisions = initial_divisions,
                    N_nodes_tanh = N_nodes_tanh,
                    m_tsq = m_tsq)

    #
    updatey = yy->updatebuffernonadaptive(updatev, yy)

    #
    integral_∂f_∂x_from_a_to_b::Vector{T} = Vector{T}(undef, d_select-1)
    integral_∂f_∂x_from_a_to_x::Vector{T} = Vector{T}(undef, d_select-1)
    h_x_persist::Vector{T} = ones(T, 1)
    f_v_x_persist::Vector{T} = ones(T, 1)
    dg = xx->computedgcomponentnonadaptive!( integral_∂f_∂x_from_a_to_b,
                                    integral_∂f_∂x_from_a_to_x,
                                    h_x_persist,
                                    f_v_x_persist,
                            xx,
                            f_v,
                            fetchZ,
                            lower_bound,
                            upper_bound,
                            c,
                            X,
                            a_RQ,
                            d_select,
                            v_persist,
                            b_array)
    #
    #### second-order.

    #
    d2g = xx->computed2gcomponentnonadaptive(    xx,
                        fetchZ,
                        c, X, a_RQ,
                        lower_bound,
                        upper_bound,
                        d_select,
                        integral_∂f_∂x_from_a_to_b,
                        integral_∂f_∂x_from_a_to_x,
                        h_x_persist,
                        f_v_x_persist,
                        v_persist,
                        b_array)


    # println("∂2f_∂x2_array[1](1.23) = "),
    # display(∂2f_∂x2_array[1](1.23))
    # println("size(∂2f_∂x2_array)")
    # println(size(∂2f_∂x2_array))
    # @assert 1==2
    return q, updatey, dg, d2g, f_v
end
