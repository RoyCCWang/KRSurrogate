"""
f is the unnormalized joint density for variates 1:d.
"""
function setupdGcomponent( f::Function,
                            v::Vector{T},
                            x_full,
                            X::Vector{Vector{T}},
                            c::Vector{T},
                            ϕ::Function,
                            dϕ::Function,
                            a::T,
                            lower_bound::T,
                            upper_bound::T;
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 10000)::Vector{Function} where T <: Real

    D = length(v) + 1

    # f_v = xx->f([v; xx])
    #
    # Z_v_persist = Vector{T}(undef, 1)
    # Z_v = evalintegral(f_v, lower_bound, upper_bound)
    # Z_v_persist[1] = clamp(Z_v, zero_tol, one(T))

    ∂f_∂x_array = Vector{Function}(undef, D)
    #∂f_loadx_array = Vector{Function}(undef, D)
    for i = 1:D
        buffer_i = copy(x_full)
        #buffer_i = x_full[1:i]
        ∂f_∂x_array[i] = xx->eval∂fwrt∂xi!(buffer_i, xx, X, c,
                                ϕ, dϕ, a, i)


    end

    return ∂f_∂x_array
end

function setupfv( f::Function,
                    v::Vector{T},
                    lower_bound::T,
                    upper_bound::T;
                    zero_tol::T = eps(T)*2,
                    initial_divisions::Int = 1,
                    max_integral_evals::Int = 1000)::Tuple{Function,Function,T} where T <: Real

    D = length(v) + 1

    f_v = xx->f([v; xx])


    Z_v_sol = evalintegral(f_v, lower_bound, upper_bound)
    Z_v = clamp(Z_v_sol, zero_tol, one(T))

    Z_v_persist = Vector{T}(undef, 1)
    Z_v_persist[1] = Z_v
    p_v = xx->f_v(xx)/Z_v_persist[1]

    return f_v, p_v, Z_v
end

# for use with hquadrature. Integrates at dim d.
function eval∂fwrt∂xi!(   x_full::Vector{T},
                        t,
                        X::Vector{Vector{T}},
                        c::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        a::T,
                        i::Int) where T <: Real
    #
    N = length(c)
    #println("length(x_full) = ", length(x_full))
    #println("length(X[1]) = ", length(X[1]))
    #println()
    @assert length(X) == N
    @assert length(x_full) == length(X[1])

    x_full[end] = t

    out = zero(T)
    for n = 1:N
        out += c[n]*eval∂kwrt∂xiRQ(x_full, X[n], ϕ, dϕ, a, i)
    end
    out = out*sqrt(a)^3

    return out
end

## speed this up later.
function eval∂kwrt∂xiRQ( x::Vector{T},
                        z::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        a::T,
                        i::Int)::T where T <: Real
    #
    r::Vector{T} = x-z
    w_x::T = ϕ(x)
    w_z::T = ϕ(z)
    dw_x::Vector{T} = dϕ(x)

    u = dot(r,r) + (w_x-w_z)^2
    ∂k_∂u = -1.5/sqrt(a+u)^5

    ∂u_∂xi = 2*( r[i] + (w_x - w_z)*dw_x[i] )

    return ∂k_∂u * ∂u_∂xi
end

# function eval∂kwrt∂xRQ( x,
#                         z::Vector{T},
#                         ϕ::Function,
#                         a::T)::Vector{T} where T <: Real
#     #
#     r::Vector{T} = x-z
#     w_x::T = ϕ(x)
#     w_z::T = ϕ(z)
#     dw_x::Vector{T} = dϕ(x)
#
#     u = dot(r,r) + (w_x-w_z)^2
#     ∂k_∂u = -1.5/sqrt(a+u)^5
#
#     ∂u_∂x = 2*r + 2*w_x*dw_x - 2*w_z*dw_x
#
#     return ∂k_∂u .* ∂u_∂x
# end

"""
g is a component map of G. This function computes g's gradient.
f is unnormalized joint density for x_{1:d}. Computes the gradient for CDF wrt f.
a,b are the integration bounds for dim d.
"""
function computedGcomponent(    x::T,
                            f_v::Function,
                            Z_v_in,
                            ∂f_∂x_array::Vector{Function},
                            a::T,
                            b::T,
                            d::Int;
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 100000)::Vector{T} where T <: Real
    # read precomputed Z(v).
    Z_v = Z_v_in[1]

    dg_dx = Vector{T}(undef, d)
    #fill!(dg_dx, -1.23)

    # terms that do not involve component d.
    for i = 1:d-1
        # println("timing:")
        # @time h_x = evalintegral(f_v, a, x;
        #                 initial_divisions = initial_divisions,
        #                 max_integral_evals = max_integral_evals)
        #
        # @time ∂h_x = evalintegral(∂f_∂x_array[i], a, x;
        #                 initial_divisions = initial_divisions,
        #                 max_integral_evals = max_integral_evals)
        #
        # @time ∂Z_v = evalintegral(∂f_∂x_array[i], a, b;
        #                 initial_divisions = initial_divisions,
        #                 max_integral_evals = max_integral_evals)
        #
        h_x = evalintegral(f_v, a, x;
                        initial_divisions = initial_divisions,
                        max_integral_evals = max_integral_evals)

        ∂h_x = evalintegral(∂f_∂x_array[i], a, x;
                        initial_divisions = initial_divisions,
                        max_integral_evals = max_integral_evals)

        ∂Z_v = evalintegral(∂f_∂x_array[i], a, b;
                        initial_divisions = initial_divisions,
                        max_integral_evals = max_integral_evals)

        numerator = ∂h_x*Z_v - h_x*∂Z_v
        denominator = Z_v^2

        dg_dx[i] = numerator/denominator
    end

    # terms involving component d.
    dg_dx[end] = f_v(x) / Z_v

    # @time     dg_dx[end] = f_v(x) / Z_v
    # println("end")
    # println()
    return dg_dx
end


function computedG( gq_array::Vector{Function},
                    x0::Vector{T},
                    𝓧_array,
                    c_array,
                    θ_array,
                    dϕ_array::Vector{Function},
                    limit_a::Vector{T},
                    limit_b::Vector{T};
                    max_integral_evals::Int = 10000 ) where T <: Real

    #
    D = length(limit_a)

    dG_x0 = Vector{Vector{T}}(undef, D)
    for d = 1:D
        fq = gq_array[d]

        v = x0[1:d-1]

        f_v, p_v, Z_v = setupfv(fq, v,
                                limit_a[d],
                                limit_b[d];
                                max_integral_evals = max_integral_evals)

        ∂f_∂x_array = setupdGcomponent( fq, v,
                                x0[1:d],
                                𝓧_array[d],
                                c_array[d],
                                θ_array[d].warpfunc,
                                dϕ_array[d],
                                θ_array[d].canonical_params.a,
                                limit_a[d],
                                limit_b[d];
                                max_integral_evals = max_integral_evals)
        #
        dG_x0[d] = computedGcomponent( x0[d],
                            f_v,
                            Z_v,
                            ∂f_∂x_array,
                            limit_a[d],
                            limit_b[d],
                            d)

    end

    return dG_x0
end


##### inverse function.

function computedF(gq_array::Vector{Function},
                    x0::Vector{T},
                    X_array,
                    c_array,
                    θ_array,
                    dϕ_array,
                    limit_a,
                    limit_b;
                    max_integral_evals::Int = 10000) where T <: Real

    dG_x0 = computedG( gq_array,
                        x0,
                        X_array,
                        c_array,
                        θ_array,
                        dϕ_array,
                        limit_a,
                        limit_b;
                        max_integral_evals = max_integral_evals)

    dG_x0_mat = zeros(T, D, D)
    for j = 1:D
        for i = j:D
            dG_x0_mat[i,j] =  dG_x0[i][j]
        end
    end

    # inverse function theorem.
    dF_u0_mat = inv(dG_x0_mat)

    # force zero entries.
    for j = 2:D
        for i = 1:j-1
            dF_u0_mat[i,j] = zero(T)
        end
    end

    return dF_u0_mat, dG_x0_mat
end

function computed2F(gq_array::Vector{Function},
                    x0::Vector{T},
                    𝓧_array,
                    c_array,
                    θ_array,
                    dϕ_array,
                    d2ϕ_array,
                    limit_a,
                    limit_b;
                    max_integral_evals::Int = 10000) where T <: Real

    #
    dF_u0_mat, dG_x0_mat_unused = computedF(  gq_array,
                            x0,
                            𝓧_array,
                            c_array,
                            θ_array,
                            dϕ_array,
                            limit_a,
                            limit_b;
                            max_integral_evals = max_integral_evals)


    d2G_x0 = computed2G(gq_array,
                            x0,
                            𝓧_array,
                            c_array,
                            θ_array,
                            dϕ_array,
                            d2ϕ_array,
                            limit_a,
                            limit_b;
                            max_integral_evals = max_integral_evals)
    # I am here.
    d2F_x0 = Vector{Matrix{T}}(undef,D)
    for k = 1:D
        RHS = sum( dF_u0_mat[k,l] .* d2G_x0[l] for l = 1:D )


        ( dfk_g0[l] .* d2g_x0[l] for l = 1:D )
    end


    return d2F_x0
end

"""
Turns each element into a full matrix of size D x D.
"""
function packaged2G(d2G_x0::Vector{Matrix{T}}) where T <: Real

    D = size(d2G_x0[end],1)

    d2G_x0_full = Vector{Matrix{T}}(undef, D)
    for k = 1:D
        d2G_x0_full[k] = zeros(T, D, D)
        d2G_x0_full[k][1:k,1:k] = d2G_x0[k]
    end

    return d2G_x0_full
end
