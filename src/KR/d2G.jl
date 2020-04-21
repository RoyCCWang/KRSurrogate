
# this is used in a verification function.
function eval∂2kwrt∂x2RQ( x::Vector{T},
                        z::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        d2ϕ::Function,
                        a::T)::Matrix{T} where T <: Real
    #
    r::Vector{T} = x-z
    w_x::T = ϕ(x)
    w_z::T = ϕ(z)
    dw_x::Vector{T} = dϕ(x)
    d2w_x2::Matrix{T} = d2ϕ(x)

    u = dot(r,r) + (w_x-w_z)^2
    ∂k_∂u = -1.5/sqrt(a+u)^5
    ∂2k_∂u2 = 3.75/sqrt(a+u)^7

    ∂u_∂x = 2*( r + (w_x - w_z)*dw_x )
    ∂2u_∂x2 = 2*( LinearAlgebra.I + dw_x*dw_x' + (w_x-w_z)*d2w_x2 )

    out = ∂k_∂u * ∂2u_∂x2 + ∂2k_∂u2 * ∂u_∂x * ∂u_∂x'

    return out
end



function eval∂2kwrt∂xi∂xjRQ( x::Vector{T},
                        z::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        d2ϕ::Function,
                        a::T,
                        i::Int,
                        j::Int)::T where T <: Real
    #
    r::Vector{T} = x-z
    w_x::T = ϕ(x)
    w_z::T = ϕ(z)
    dw_x::Vector{T} = dϕ(x)
    d2w_x2::Matrix{T} = d2ϕ(x)

    u::T = dot(r,r) + (w_x-w_z)^2
    ∂k_∂u::T = -1.5/sqrt(a+u)^5
    ∂2k_∂u2::T = 3.75/sqrt(a+u)^7

    δ_i_j = zero(T)
    if i == j
        δ_i_j = one(T)
    end
    ∂2u_∂xi_∂xj::T = 2*( δ_i_j + dw_x[i]*dw_x[j] + (w_x-w_z)*d2w_x2[i,j] ) #


    ∂u_∂xi::T = 2*( r[i] + (w_x - w_z)*dw_x[i] )
    ∂u_∂xj::T = 2*( r[j] + (w_x - w_z)*dw_x[j] )
    out = ∂k_∂u * ∂2u_∂xi_∂xj + ∂2k_∂u2 * ∂u_∂xi * ∂u_∂xj

    # #### debug.
    # ∂u_∂x = 2*( r + (w_x - w_z)*dw_x )
    # ∂2u_∂x2 = 2*( LinearAlgebra.I + dw_x*dw_x' + (w_x-w_z)*d2w_x2 ) #
    #
    # ref = ∂k_∂u * ∂2u_∂x2 + ∂2k_∂u2 * ∂u_∂x * ∂u_∂x'
    #
    #
    # Printf.@printf("(i,j) = (%d,%d) \n", i, j)
    # println("out = ", out)
    # println("ref = ", ref)
    # println()
    #
    # # probe1 = LinearAlgebra.I + dw_x[i]*dw_x[j] #+ (w_x-w_z)*d2w_x2[i,j]
    # # probe2 = LinearAlgebra.I + dw_x*dw_x' #+ (w_x-w_z)*d2w_x2
    # # println("probe1 = ", probe1)
    # # println("probe2 = ", probe2)
    # # println()
    #
    # if i != j
    #     @assert 444==3
    # end

    return out
end


# this is a verification function.
function eval∂2kwrt∂x2RQviacomponents(x::Vector{T},
                        z::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        d2ϕ::Function,
                        a::T,
                        Nr::Int,
                        Nc::Int)::Matrix{T} where T <: Real
    #
    out = zeros(T, Nr, Nc)
    fill!(out, 1.23)
    for i = 1:Nr
        for j = 1:Nc
            out[i,j] = eval∂2kwrt∂xi∂xjRQ(x, z, ϕ, dϕ, d2ϕ, a, i, j)

        end
    end

    return out
end

function eval∂2fwrt∂xi∂xj!(   x_full::Vector{T},
                        t,
                        X::Vector{Vector{T}},
                        c::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        d2ϕ::Function,
                        a::T,
                        i::Int,
                        j::Int) where T <: Real
    #
    N = length(c)
    @assert length(X) == N

    x_full[end] = t

    out = zero(T)
    for n = 1:N
        out += c[n]*eval∂2kwrt∂xi∂xjRQ(x_full, X[n], ϕ, dϕ, d2ϕ, a, i, j)
    end
    out = out*sqrt(a)^3

    return out
end


"""
f is the unnormalized joint density for variates 1:d.
"""
function setupd2Gcomponent( f::Function,
                            v::Vector{T},
                            x_full,
                            X::Vector{Vector{T}},
                            c::Vector{T},
                            ϕ::Function,
                            dϕ::Function,
                            d2ϕ::Function,
                            a::T,
                            lower_bound::T,
                            upper_bound::T;
                            zero_tol::T = eps(T)*2,
                            initial_divisions::Int = 1,
                            max_integral_evals::Int = 1000)::Matrix{Function} where T <: Real

    D = length(v) + 1

    # f_v = xx->f([v; xx])
    #
    # Z_v_persist = Vector{T}(undef, 1)
    # Z_v = evalintegral(f_v, lower_bound, upper_bound)
    # Z_v_persist[1] = clamp(Z_v, zero_tol, one(T))

    ∂2f_∂x2_array = Matrix{Function}(undef, D, D)
    #∂f_loadx_array = Vector{Function}(undef, D)
    for i = 1:D
        for j = 1:D
            buffer_ij = copy(x_full)

            ∂2f_∂x2_array[i,j] = xx->eval∂2fwrt∂xi∂xj!(buffer_ij, xx, X, c,
                                    ϕ, dϕ, d2ϕ, a, i, j)
        end
    end

    return ∂2f_∂x2_array#, f_v, Z_v
end

# I think this is really vi.
function eval∂Awrt∂vj(  x::T,
                        f_v::Function,
                        Z_v::T,
                        ∂f_∂x_array::Vector{Function},
                        ∂2f_∂x2_array::Matrix{Function},
                        a::T,
                        b::T,
                        i::Int,
                        j::Int;
                        max_integral_evals::Int = 10000)::T where T <: Real

    h_x = evalintegral(f_v, a, x)

    ∂h_∂xi = evalintegral(∂f_∂x_array[i], a, x)
    ∂Z_∂xi = evalintegral(∂f_∂x_array[i], a, b)

    ∂h_∂xj = evalintegral(∂f_∂x_array[j], a, x)
    ∂Z_∂xj = evalintegral(∂f_∂x_array[j], a, b)

    ∂2h_∂xi∂xj = evalintegral(∂2f_∂x2_array[i,j], a, x)
    ∂2Z_∂xi∂xj = evalintegral(∂2f_∂x2_array[i,j], a, b)

    term1 = ∂2h_∂xi∂xj * Z_v
    term2 = ∂h_∂xi * ∂Z_∂xj
    term3 = -∂h_∂xj * ∂Z_∂xi
    term4 = -h_x*∂2Z_∂xi∂xj

    return term1 + term2 + term3 + term4
end


function eval∂Bwrt∂vj(Z_v::T,
                                ∂f_∂x_array::Vector{Function},
                                a::T,
                                b::T,
                                j::Int)::T where T <: Real

    ∂Z_∂xj = evalintegral(∂f_∂x_array[j], a, b)

    return 2 * Z_v * ∂Z_∂xj
end

function computed2Gcomponent(    x::T,
                            f_v::Function,
                            Z_v::T,
                            ∂f_∂x_array::Vector{Function},
                            ∂2f_∂x2_array::Matrix{Function},
                            a::T,
                            b::T,
                            d::Int) where T <: Real

    #D = length(∂f_∂x_array)

    d2g_dx = Matrix{T}(undef, d, d)
    fill!(d2g_dx, -1.23)

    # common objects.
    h_x = evalintegral(f_v, a, x)
    obj_B = Z_v^2

    # terms that do not involve component d.
    for i = 1:d-1

        #∂h_xj = evalintegral(∂f_∂x_array[j], a, x)
        #∂Z_vj = evalintegral(∂f_∂x_array[j], a, b)

        #numerator_j = ∂h_xj*Z_v - h_x*∂Z_vj


        ∂h_xi = evalintegral(∂f_∂x_array[i], a, x)
        ∂Z_vi = evalintegral(∂f_∂x_array[i], a, b)

        obj_A = ∂h_xi*Z_v - h_x*∂Z_vi

        for j = 1:d-1



            ∂Awrt∂vj = eval∂Awrt∂vj(  x,
                                    f_v,
                                    Z_v,
                                    ∂f_∂x_array,
                                    ∂2f_∂x2_array,
                                    a,
                                    b,
                                    i,
                                    j)
            #

            ∂Bwrt∂vj = eval∂Bwrt∂vj( Z_v,
                                    ∂f_∂x_array,
                                    a,
                                    b,
                                    j)
            #
            numerator = ∂Awrt∂vj*obj_B - obj_A*∂Bwrt∂vj
            denominator = obj_B^2

            #Printf.@printf("processed (%d,%d)\n", i, j)
            d2g_dx[i,j] = numerator/denominator

        end
    end



    # terms involving component d.
    for i = 1:d-1
        ∂f_∂vi = ∂f_∂x_array[i](x)
        ∂Z_∂vi = evalintegral(∂f_∂x_array[i], a, b)

        numerator = ∂f_∂vi*Z_v - f_v(x)*∂Z_∂vi

        value = numerator / Z_v^2
        d2g_dx[d,i] = value
        d2g_dx[i,d] = value
    end

    # singleton term.
    #d2g_dx[end] = Calculus.derivaitve(f_v, x) / Z_v
    d2g_dx[end] = ∂f_∂x_array[d](x) / Z_v

    return d2g_dx
end

function computed2G(    gq_array::Vector{Function},
                        x0::Vector{T},
                        X_array::Vector{Vector{Vector{T}}},
                        c_array::Vector{Vector{T}},
                        θ_array,
                        dϕ_array::Vector{Function},
                        d2ϕ_array::Vector{Function},
                        limit_a::Vector{T},
                        limit_b::Vector{T};
                        zero_tol::T = 1e-9,
                        max_integral_evals::Int = 10000 ) where T <: Real

    D = length(x0)

    d2G_x0 = Vector{Matrix{T}}(undef, D)

    d2G_x0[1] = Matrix{T}(undef,1,1)
    ∂f1_x0 = eval∂fwrt∂xi!(copy(x0[1:1]), x0[1], X_array[1], c_array[1],
                            θ_array[1].warpfunc,
                            dϕ_array[1], θ_array[1].canonical_params.a, 1)
    #
    Z_v_sol = evalintegral(gq_array[1], limit_a[1], limit_b[1])
    Z_v = clamp(Z_v_sol, zero_tol, one(T))

    d2G_x0[1][1] = ∂f1_x0 / Z_v

    for d = 2:D
        fq = gq_array[d]
        v = x0[1:d-1]

        d2G_x0[d] = computed2Ginner(gq_array[d],
                        v,
                        x0,
                        X_array[d],
                        c_array[d],
                        θ_array[d],
                        dϕ_array[d],
                        d2ϕ_array[d],
                        θ_array[d].canonical_params.a,
                        limit_a[d],
                        limit_b[d],
                        d;
                        max_integral_evals = max_integral_evals)
        #
    end

    return d2G_x0
end

function computed2Ginner( fq::Function,
                            v::Vector{T},
                            x0::Vector{T},
                            X,
                            c,
                            θ,
                            dϕ,
                            d2ϕ,
                            a_θ,
                            lower_bound,
                            upper_bound,
                            d::Int;
                            max_integral_evals::Int = 10000 ) where T <: Real
    #

    f_v, p_v, Z_v = setupfv( fq, v,
                            lower_bound,
                            upper_bound;
                            max_integral_evals = max_integral_evals)

    ∂f_∂x_array = setupdGcomponent( fq, v,
                            x0[1:d],
                            X,
                            c,
                            θ.warpfunc,
                            dϕ,
                            θ.canonical_params.a,
                            lower_bound,
                            upper_bound;
                            max_integral_evals = max_integral_evals)
    #
    ∂2f_∂x2_array = setupd2Gcomponent(fq, v,
                            x0[1:d],
                            X,
                            c,
                            θ.warpfunc,
                            dϕ,
                            d2ϕ,
                            a_θ,
                            lower_bound,
                            upper_bound;
                            max_integral_evals = max_integral_evals)

    d2Gcomponent_x0 = computed2Gcomponent(    x0[d],
                            f_v,
                            Z_v,
                            ∂f_∂x_array,
                            ∂2f_∂x2_array,
                            lower_bound,
                            upper_bound,
                            d)


    return d2Gcomponent_x0
end
