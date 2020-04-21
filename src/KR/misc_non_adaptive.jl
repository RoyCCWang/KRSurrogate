
function updatevnonadaptive!( v::Vector{T},
                            b_array::Vector{T},
                            Z::Vector{T},
                            v_input::Vector{T},
                            c::Vector{T},
                            X::Vector{Vector{T}},
                            a_RQ::T,
                            lower_limit::T,
                            upper_limit::T) where T
    # debug.
    # println("length(v) = ", length(v))
    # println("length(X[1])-1 = ", length(X[1])-1)
    # println("length(v_input) = ", length(v_input))

    @assert length(v_input) == length(X[1])-1 == length(v)

    v[:] = v_input

    N = length(b_array)
    @assert length(c) == N

    # update b_v.
    D_m_1 = length(v_input)
    for n = 1:N
        #b_array[n] = a_RQ + norm(v_input - X[n][1:D_m_1])^2

        b_array[n] = getbv(v, a_RQ, X[n]) #a_RQ + norm(v_input - X[n][1:D_m_1])^2

        # a1 = getbv(v, a_RQ, X[n])
        # a2 = a_RQ + norm(v_input - X[n][1:D_m_1])^2
        # if abs(a1-a2) > 1e-10
        #     println("a1 = ", a1)
        #     println("a2 = ", a2)
        #     println("norm(v-v_input) = ", norm(v-v_input))
        #     println(" norm^2 = ", norm(v_input - X[n][1:D_m_1])^2)
        #     println(" sum ^2 = ", sum((v[d]-X[n][d])^2 for d = 1:length(v)))
        #
        #     println(" sum ^2 = ", sum((v[d]-X[n][d])^2 for d = 1:length(v)))
        #
        #     @assert 555==4
        # end
    end

    # update normalizing constant.
    Z[1] = evalnonadaptiveRQqueryCDF(c, X, a_RQ, lower_limit, upper_limit, v)

    return nothing
end

function normalizefunc(f::Function, x, Z::Vector{T})::T where T
    return f(x)/Z[1]
end

function setupquantilenonadaptive( c::Vector{T},
                                        X::Vector{Vector{T}},
                                        a_RQ::T,
                                b_array::Vector{T},
                                lower_limit::T,
                                upper_limit::T,
                                d::Int;
                                max_numerical_inverse_iters::Int = 1000000,
                                max_integral_evals::Int = 10000,
                                initial_divisions::Int = 1,
                                N_nodes_tanh::Int = 100,
                                m_tsq::Int = 7) where T <: Real
    # conditional PDF.
    v_persist::Vector{T} = ones(T,d-1) # used for the CDF, which is used by quantile solver.
    Z_persist::Vector{T} = ones(T,1)
    #f_v = xx->f([v_persist; xx]) # unnormalized pdf.
    f_v = tt->evalnonadaptiveRQquery(tt, b_array, c, X, a_RQ)

    𝑓 = xx->f_v(xx)/Z_persist[1] # normalized pdf.

    # update for the conditioning variables.
    updatev = vv->updatevnonadaptive!( v_persist,
                                b_array,
                                Z_persist,
                                vv,
                                c,
                                X,
                                a_RQ,
                                lower_limit,
                                upper_limit)

    # coniditional CDF.
    # to do: pass initial_divisions to evalcdfviaHCubature.
    tmp_func = tt->evalnonadaptiveRQqueryCDF(c, X, a_RQ, lower_limit,
                                        tt, v_persist)

    𝐹 = tt->clamp( normalizefunc(tmp_func, tt, Z_persist),
                    zero(T), one(T) )

    # # this works.
    # 𝐹0 = xx->clamp(evalintegral(𝑓,
    #                 lower_limit, xx)[1], zero(T), one(T))
    #
    # #
    # F_v = xx->evalintegral(f_v,
    #                 lower_limit, xx)[1]
    #
    # # so we have a problem here.
    # println()
    # println("d = ", d)
    # x0 = randn(d)
    # t0 = x0[end]
    #
    # v0 = x0[1:end-1]
    # updatev(v0)
    #
    # Z_NI = evalintegral(f_v,
    #                 lower_limit, upper_limit)[1]
    #
    # println("v0 = ", v0)
    # println("v  = ", v_persist)
    # println("Z     = ", Z_persist)
    # println("NI: Z = ", Z_NI)
    # println("abs(Z_persist-Z_NI) = ", abs(Z_persist[1] - Z_NI))
    # println()
    #
    # println("𝐹(t0)  = ", 𝐹(t0))
    # println("𝐹2(t0) = ", 𝐹2(t0))
    #
    # println("tmp_func(t0) = ", tmp_func(t0))
    # println("F_v(t0)      = ", F_v(t0))
    # println()
    #
    # if d > 1
    #     @assert 1==2
    # end

    # initial search.
    q_v_initial = setupquantileintialsearch(    lower_limit,
                                                upper_limit,
                                                𝑓,
                                                N_nodes_tanh,
                                                m_tsq)

    q_v = uu->evalquantilewithoutTaylor(q_v_initial, 𝐹, uu,
                                lower_limit, upper_limit,
                            𝑓, f_v;
                            max_iters = max_numerical_inverse_iters)

    # to do: add checks and speed this up.
    q = (vv,uu)->evalconditionalquantile(uu, vv, updatev, q_v)
    #q(randn(d-1), rand())

    fetchZ = xx->fetchscalarpersist(Z_persist)

    return q, f_v, updatev, fetchZ, v_persist
end

function computedgcomponentnonadaptive!( integral_∂f_∂x_from_a_to_b::Vector{T},
                                integral_∂f_∂x_from_a_to_x::Vector{T},
                                h_x_out::Vector{T},
                                f_v_x_out::Vector{T},
                            x::T,
                            f_v::Function,
                            fetchZ::Function,
                            a::T,
                            b::T,
                            c::Vector{T},
                            X::Vector{Vector{T}},
                            a_RQ::T,
                            d::Int,
                            v::Vector{T},
                            b_array::Vector{T})::Vector{T} where T <: Real

    #
    # read precomputed Z(v).
    Z_v::T = fetchZ(NaN)

    dg_dx = Vector{T}(undef, d)
    fill!(dg_dx, -1.23)

    # outputs, for caching.
    resize!(integral_∂f_∂x_from_a_to_b, d-1)
    resize!(integral_∂f_∂x_from_a_to_x, d-1)
    for i = 1:d-1
        integral_∂f_∂x_from_a_to_b[i] = eval∂w∂viquery(c, X, a_RQ, a, b, i, v)
        integral_∂f_∂x_from_a_to_x[i] = eval∂w∂viquery(c, X, a_RQ, a, x, i, v)
    end

    h_x = evalnonadaptiveRQqueryCDF(c, X, a_RQ, a, x, v)
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

function eval∂Awrt∂vjfastnonadaptive(  x::T,
                        h_x::T,
                        Z_v::T,
                        c::Vector{T},
                        X::Vector{Vector{T}},
                        a_RQ::T,
                        a::T,
                        b::T,
                        i::Int,
                        j::Int,
                        integral_∂f_∂x_from_a_to_b::Vector{T},
                        integral_∂f_∂x_from_a_to_x::Vector{T},
                        v::Vector{T})::T where T <: Real



    ∂h_∂xi = integral_∂f_∂x_from_a_to_x[i]
    ∂Z_∂xi = integral_∂f_∂x_from_a_to_b[i]

    ∂h_∂xj = integral_∂f_∂x_from_a_to_x[j]
    ∂Z_∂xj = integral_∂f_∂x_from_a_to_b[j]

    ∂2h_∂xi∂xj = eval∂2w∂vijquery(c, X, a_RQ, a, x, i, j, v)
    ∂2Z_∂xi∂xj = eval∂2w∂vijquery(c, X, a_RQ, a, b, i, j, v)

    term1 = ∂2h_∂xi∂xj * Z_v
    term2 = ∂h_∂xi * ∂Z_∂xj
    term3 = -∂h_∂xj * ∂Z_∂xi
    term4 = -h_x*∂2Z_∂xi∂xj

    return term1 + term2 + term3 + term4
end

function computed2gcomponentnonadaptive(    x::T,
                            fetchZ::Function,
                            c::Vector{T},
                            X::Vector{Vector{T}},
                            a_RQ::T,
                            a::T,
                            b::T,
                            d::Int,
                            integral_∂f_∂x_from_a_to_b::Vector{T},
                            integral_∂f_∂x_from_a_to_x::Vector{T},
                            h_x_in::Vector{T},
                            f_v_x_in::Vector{T},
                            v::Vector{T},
                            b_array::Vector{T})::Matrix{T} where T <: Real

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

    ### main routine.
    fill!(d2g_dx, -Inf)

    for i = 1:d-1

        #numerator_j = ∂h_xj*Z_v - h_x*∂Z_vj

        ∂h_xi = integral_∂f_∂x_from_a_to_x[i]
        ∂Z_vi = integral_∂f_∂x_from_a_to_b[i]

        obj_A = ∂h_xi*Z_v - h_x*∂Z_vi

        for j = i:d-1

            ∂Awrt∂vj = eval∂Awrt∂vjfastnonadaptive(  x,
                                    h_x,
                                    Z_v,
                                    c, X, a_RQ,
                                    a,
                                    b,
                                    i,
                                    j,
                                    integral_∂f_∂x_from_a_to_b,
                                    integral_∂f_∂x_from_a_to_x,
                                    v)
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

    # I am here. search for ∂f_∂x_array and  ∂2f_∂x2_array terms in this file.

    # terms involving component d.
    for i = 1:d-1
        #∂f_∂vi = ∂f_∂x_array[i](x)
        ∂f_∂vi = evalnonadaptiveRQ∂f∂xj(x, v, b_array, c, X, a_RQ, i)

        ∂Z_∂vi = integral_∂f_∂x_from_a_to_b[i]

        numerator = ∂f_∂vi*Z_v - f_v_x*∂Z_∂vi

        value = numerator / obj_B
        d2g_dx[d,i] = value
        d2g_dx[i,d] = value
    end

    # singleton term.
    #d2g_dx[end] = ∂f_∂x_array[d](x) / Z_v
    d2g_dx[end] = evalnonadaptiveRQ∂f∂xj(x, v, b_array, c, X, a_RQ, d) / Z_v

    return d2g_dx
end
