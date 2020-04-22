
function evalKR(    M::Int,
                    c_array::Vector{Vector{T}},
                    θ_array::Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                    𝓧_array::Vector{Vector{Vector{T}}},
                    max_integral_evals::Int,
                    x_ranges,
                    d_ϕ_array::Vector{Function},
                    d2_ϕ_array::Vector{Function},
                    N_nodes_tanh::Int,
                    m_tsq::Int,
                    quantile_err_tol::T,
                    max_traversals::Int,
                    N_predictive_traversals::Int,
                    correction_epoch::Int,
                    quantile_max_iters::Int,
                    quantile_convergence_zero_tol::T,
                    n_limit::Int,
                    n0::Int,
                    u_input::Vector{T} = zeros(T,0) )::Tuple{Vector{Vector{T}},Vector{Vector{T}}} where T<:Real

    # set up.
    D = length(d_ϕ_array)
    @assert length(c_array) == length(θ_array) == D

    # allocate output.
    x_array::Vector{Vector{T}} = collect( Vector{T}(undef,D) for m = 1:M )
    discrepancy_array::Vector{Vector{T}} = collect( Vector{T}(undef,D) for m = 1:M ) # debug.

    # pre-allocate intermediate objects.
    u::T = NaN

    for d = 1:D
        println("d = ", d)

        # set up.
        q = setupquantilefordimW2(c_array[d],
                                θ_array[d],
                                𝓧_array[d],
                                max_integral_evals,
                                x_ranges[d][1],
                                x_ranges[d][end],
                                d_ϕ_array[d],
                                d2_ϕ_array[d],
                                d,
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

        for m = 1:M
            ### transport standard normal sample space to [0,1].

            # draw realization from standard normal.
            if isempty(u_input)
                u = convert(T, drawstdnormalcdf())
            else
                # case for debug.
                u = u_input[d]
            end

            # get v.
            v = x_array[m][1:d-1]

            # evaluate quantile.
            𝑥, 𝑢, N_iters = q(v, u)

            # update buffer.
            x_array[m][d] = 𝑥
            discrepancy_array[m][d] = abs(𝑢-u)
        end

    end

    return x_array, discrepancy_array
end

# this is the skip_D version of evalKR.
function evalKR2(    M::Int,
                    c_array::Vector{Vector{T}},
                    θ_array::Vector{RKHSRegularization.AdaptiveKernelType{RKHSRegularization.RationalQuadraticKernelType{T}}},
                    𝓧_array::Vector{Vector{Vector{T}}},
                    max_integral_evals::Int,
                    x_ranges,
                    d_ϕ_array::Vector{Function},
                    d2_ϕ_array::Vector{Function},
                    N_nodes_tanh::Int,
                    m_tsq::Int,
                    quantile_err_tol::T,
                    max_traversals::Int,
                    N_predictive_traversals::Int,
                    correction_epoch::Int,
                    quantile_max_iters::Int,
                    quantile_convergence_zero_tol::T,
                    n_limit::Int,
                    n0::Int,
                    f::Function;
                    u_input::Vector{T} = zeros(T,0),
                    max_numerical_inverse_iters::Int = 1000)::Tuple{Vector{Vector{T}},Vector{Vector{T}}} where T<:Real

    # set up.
    D = length(x_ranges)
    @assert length(c_array) == length(θ_array) == D-1

    # allocate output.
    x_array::Vector{Vector{T}} = collect( Vector{T}(undef,D) for m = 1:M )
    discrepancy_array::Vector{Vector{T}} = collect( Vector{T}(undef,D) for m = 1:M ) # debug.

    # pre-allocate intermediate objects.
    u::T = NaN

    for d = 1:D-1
        println("d = ", d)

        # set up.
        q = setupquantilefordimW2(c_array[d],
                                θ_array[d],
                                𝓧_array[d],
                                max_integral_evals,
                                x_ranges[d][1],
                                x_ranges[d][end],
                                d_ϕ_array[d],
                                d2_ϕ_array[d],
                                d,
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

        applyquantile!( x_array,
                        discrepancy_array,
                        M,
                        u_input,
                        q,
                        d)


    end

    q_D = setupquantilegenericproxy( f,
                    limit_a[D],
                    limit_b[D],
                    D;
                    max_numerical_inverse_iters = max_numerical_inverse_iters,
                    max_integral_evals = max_integral_evals,
                    initial_divisions = initial_divisions,
                    N_nodes_tanh = N_nodes_tanh,
                    m_tsq = m_tsq)[1]

    applyquantile!( x_array,
                    discrepancy_array,
                    M,
                    u_input,
                    q_D,
                    D)

    return x_array, discrepancy_array
end

function eval1Dnumericalinverse(f::Function,
                                y::T,
                                x0::T,
                                a::T,
                                b::T,
                                max_iters::Int) where T <: Real
    @assert a < b

    obj_func = xx->((f(xx[1])-y)^2)::T

    op = Optim.Options( iterations = max_iters,
                         store_trace = false,
                         show_trace = false)

    results = Optim.optimize(   obj_func,
                                [x0],
                                Optim.NewtonTrustRegion(),
                                op)

    x_star = results.minimizer
    x_out = clamp(x_star[1], a, b)

    return x_out, results
end

function runBrentoptim99(obj_func::Function, a::T, b::T, max_iters::Int) where T <: Real

    results = Optim.optimize(obj_func, a, b, Optim.Brent())
    #results = Optim.optimize(obj_func, a, b, Optim.GoldenSection())

    op = Optim.Options( iterations = max_iters,
                         store_trace = false,
                         show_trace = false)

    x = results.minimizer

    return x, results
end

function applyquantile!(x_array::Vector{Vector{T}},
                        discrepancy_array::Vector{Vector{T}},
                        M::Int,
                        u_input,
                        q::Function,
                        d::Int) where T <: Real

    for m = 1:M
        ### transport standard normal sample space to [0,1].

        # draw realization from standard normal.
        if isempty(u_input)
            u = convert(T, drawstdnormalcdf())
        else
            # case for debug.
            u = u_input[d]
        end

        # get v.
        v = x_array[m][1:d-1]

        # evaluate quantile.
        # println("d = ", d)
        # println("v = ", v)
        # println("u = ", u)
        # println("q(v, u) = ", q(v, u))
        # println()
        𝑥, 𝑢, N_iters_unused = q(v, u)

        # update buffer.
        x_array[m][d] = 𝑥
        discrepancy_array[m][d] = abs(𝑢-u)
    end

    return nothing
end
