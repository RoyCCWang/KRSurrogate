function eval∂2fwrt∂xi∂xjfast!( x::Vector{T},
                                t,
                                X::Vector{Vector{T}},
                                c::Vector{T},
                                ϕ::Function,
                                dϕ::Function,
                                d2ϕ::Function,
                                a::T,
                                i::Int,
                                j::Int,
                                w_X::Vector{T},
                                b_array::Vector{T},
                                multiplier_value::T)::T where T <: Real
    #
    N = length(c)
    @assert length(X) == N
    #println("x = ", x)
    x[end] = t

    # common objects.
    w_x::T = ϕ(x)

    dw_x::Vector{T} = dϕ(x)
    dw_x_i::T = dw_x[i]
    dw_x_j::T = dw_x[j]

    d2w_x2_i_j = d2ϕ(x)[i,j]

    # pre-allocate.
    w_z::T = zero(T)
    Δw::T = zero(T)

    ∂k_∂u_denominator::T = zero(T)
    #∂2k_∂u2_denominator::T = zero(T)

    ∂2u_∂xi_∂xj::T = zero(T)
    ∂u_∂xi::T = zero(T)
    ∂u_∂xj::T = zero(T)

    au::T = zero(T)

    δ_i_j = zero(T)
    if i == j
        δ_i_j = one(T)
    end

    out::T = zero(T)
    for n = 1:N

        z = X[n]

        w_z = ϕ(z)
        Δw = w_x-w_z

        au = b_array[n] + (x[end]-z[end])^2 + Δw^2
        ∂k_∂u_denominator = sqrt(au)^5

        ∂2u_∂xi_∂xj = δ_i_j + dw_x_i*dw_x_j + Δw*d2w_x2_i_j

        ∂u_∂xi = x[i]-z[i] + Δw*dw_x_i
        ∂u_∂xj = x[j]-z[j] + Δw*dw_x_j

        out += c[n]*( -∂2u_∂xi_∂xj + ∂u_∂xi*∂u_∂xj*5.0/au )/∂k_∂u_denominator

        #out += c[n]*eval∂2kwrt∂xi∂xjRQ(x_full, X[n], ϕ, dϕ, d2ϕ, a, i, j)
    end
    out = out*multiplier_value

    return out
end


function setup∂f∂xfull( v::Vector{T},
                        X::Vector{Vector{T}},
                        c::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        a::T,
                        D::Int,
                        w_X::Vector{T},
                        b_array::Vector{T}) where T
    #D = length(X[1])

    x_persist::Vector{T} = [v; one(T)]

    ∂f_∂x_array = Vector{Function}(undef, D)
    for i = 1:D
        # ∂f_∂x_array[i] = xx->eval∂fwrt∂xiold!(x_persist, xx, X, c,
        #                         ϕ, dϕ, a, i, w_X, b_array)
        ∂f_∂x_array[i] = xx->eval∂fwrt∂xifast!(x_persist, xx, X, c,
                                ϕ, dϕ, a, i, w_X, b_array)
    end

    return ∂f_∂x_array, x_persist
end

# for use with hquadrature. Integrates at dim d.
function eval∂fwrt∂xifast!(   x::Vector{T},
                        t::T,
                        X::Vector{Vector{T}},
                        c::Vector{T},
                        ϕ::Function,
                        dϕ::Function,
                        a::T,
                        i::Int,
                        w_X::Vector{T},
                        b_array::Vector{T})::T where T <: Real
    #
    N = length(c)
    @assert length(X) == N
    @assert length(x) == length(X[1])

    # update.
    x[end] = t

    # common objects.
    w_x::T = ϕ(x)
    dw_x_i::T = dϕ(x)[i]

    # pre-allocate.
    w_z::T = zero(T)
    Δw::T = zero(T)
    r_i::T = zero(T)
    r_end::T = zero(T)
    ∂k_∂u::T = zero(T)
    ∂u_∂xi::T = zero(T)

    out = zero(T)
    for n = 1:N

        # prepare z.
        z = X[n]
        w_z = w_X[n]
        #w_z = ϕ(z)

        # objects that varies with x and requires z.
        Δw = w_x-w_z

        r_i = x[i]-z[i]
        r_end = x[end]-z[end]

        # chain rule factors.
        denominator_of_∂k_∂u = sqrt( b_array[n] + r_end^2 + Δw^2 )^5

        ∂u_∂xi = r_i + Δw*dw_x_i

        out += c[n]*∂u_∂xi/denominator_of_∂k_∂u
    end
    out = -3.0*out*sqrt(a)^3

    return out
end

# # working.
# function eval∂fwrt∂xiold!(   x::Vector{T},
#                         t::T,
#                         X::Vector{Vector{T}},
#                         c::Vector{T},
#                         ϕ::Function,
#                         dϕ::Function,
#                         a::T,
#                         i::Int,
#                         w_X::Vector{T},
#                         b_array::Vector{T})::T where T <: Real
#     #
#     N = length(c)
#     @assert length(X) == N
#     @assert length(x) == length(X[1])
#
#     # update.
#     x[end] = t
#
#     out = zero(T)
#     for n = 1:N
#         out += c[n]*eval∂kwrt∂xiRQ(x, X[n], ϕ, dϕ, a, i)
#     end
#     out = out*sqrt(a)^3
#
#     return out
# end
