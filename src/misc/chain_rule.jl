
"""
Second-order inverse function theorem, via multivariate chain rule.
f: ℝ^D → ℝ^D, g: ℝ^D → ℝ^D.
     u ↦ f(u),     x ↦ u := g(x).

f is inverse of g, i.e., (f∘g)(x) = x.

Returns d2f with respect to du2.
"""
function applyinvfuncTHMd2(  d2g_x::Vector{Matrix{T}},
                            df_u::Matrix{T})::Vector{Matrix{T}} where T <: Real

    D = length(d2g_x)
    @assert size(df_u, 1) == D == size(df_u, 2)

    d2f_u = Vector{Matrix{T}}(undef, D)
    for k = 1:D

        # speed up later.
        d2g_x_full_mat = packaged2G(d2g_x)
        RHS = sum( df_u[k,l] .* d2g_x_full_mat[l] for l = 1:D )

        d2f_u[k] = -(df_u'*RHS*df_u)
    end

    return d2f_u
end

"""
This is applyinvfuncTHMd2() when df_u and dg_x are lower-triangular,
which allows length(d2g_x[d]) = (d,d).
"""
function applyinvfuncTHMd2lowertriangular(  d2g_x::Vector{Matrix{T}},
                            df_u::Matrix{T})::Vector{Matrix{T}} where T <: Real

    D = length(d2g_x)
    @assert size(df_u, 1) == D == size(df_u, 2)

    d2f_u = Vector{Matrix{T}}(undef, D)
    for k = 1:D

        RHS = evalRHSinvfuncTHMlowertriangular(d2g_x, df_u, k)

        d2f_u[k] = -(df_u'*RHS*df_u)
    end

    return d2f_u
end

"""
Second-order multivariate chain rule.
f: ℝ^D → ℝ^D, g: ℝ^D → ℝ^D.
     u ↦ f(u),     x ↦ u := g(x).

Returns d2f with respect to dx2.

applychainruled2(   dg_x::Matrix{T},
                    d2g_x::Vector{Matrix{T}},
                    df_u::Matrix{T},
                    d2f_u::Vector{Matrix{T}})::Vector{Matrix{T}}
"""
function applychainruled2(  dg_x::Matrix{T},
                            d2g_x::Vector{Matrix{T}},
                            df_u::Matrix{T},
                            d2f_u::Vector{Matrix{T}})::Vector{Matrix{T}} where T <: Real

    D = length(d2g_x)

    # sanity checks.
    @assert size(df_u, 1) == D == size(df_u, 2)

    for l = 1:D
        @assert size(d2g_x[l],1) == size(d2g_x[l], 2) == D
    end

    # apply chain rule.
    d2f_x = Vector{Matrix{T}}(undef, D)
    for k = 1:D

        # speed up later.
        term2 = sum( df_u[k,l] .* d2g_x[l] for l = 1:D )

        term1 = dg_x'*d2f_u[k]*dg_x

        d2f_x[k] = term1 + term2
    end

    return d2f_x
end

# does RHS = sum( df_u[k,l] .* d2g_x_full_mat[l] for l = 1:D )
# can probably speed this up even more.
function evalRHSinvfuncTHMlowertriangular(d2g_x::Vector{Matrix{T}},
                    df_u::Matrix{T},
                    k::Int)::Matrix{T} where T <: Real

    D = length(d2g_x)

    out = zeros(T, D, D)
    for l = 1:D
        out[1:l,1:l] += df_u[k,l] .* d2g_x[l]
    end

    return out
end
