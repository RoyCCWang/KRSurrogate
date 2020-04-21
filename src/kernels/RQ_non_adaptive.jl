
##### weighted sum of RQ kernels.

function evalnonadaptiveRQ∂f∂xj(t::T2,
                                v,
                                c::Vector{T},
                                X::Vector{Vector{T}},
                                a::T,
                                j::Int)::T2 where {T,T2}
    #
    b_array = collect( getbv(v, a_RQ, X[n]) for n = 1:length(X) )

    return evalnonadaptiveRQ∂f∂xj(t, v, b_array, c, X, a, j)
end

function evalnonadaptiveRQ∂f∂xj(t::T2,
                                v,
                                b_array,
                                c::Vector{T},
                                X::Vector{Vector{T}},
                                a::T,
                                j::Int)::T2 where {T,T2}
    N = length(c)
    @assert length(X) == N

    x = [v; t]

    out = zero(T)
    for n = 1:N
        #b_v = getbv(v, a_RQ, X[n])

        r_j = x[j]-X[n][j]

        out -= c[n]*3*r_j/sqrt(b_array[n] + (t-X[n][end])^2)^5
    end

    return out*sqrt(a)^3
end


function evalnonadaptiveRQquery(t::T2,
                                b_array,
                                c::Vector{T},
                                X::Vector{Vector{T}},
                                a::T)::T2 where {T,T2}
    N = length(c)
    @assert length(X) == N == length(b_array)

    out = zero(T)
    for n = 1:N
        out += c[n]/sqrt(b_array[n] + (t-X[n][end])^2)^3
    end

    return out*sqrt(a)^3
end

function evalnonadaptiveRQquery(x::Vector{T2},
                                c::Vector{T},
                                X::Vector{Vector{T}},
                                a::T)::T2 where {T,T2}
    N = length(c)
    @assert length(X) == N

    out = zero(T)
    for n = 1:N
        out += c[n]*evalnonadaptiveRQ(x, X[n], a)
    end

    return out*sqrt(a)^3
end


function evalnonadaptiveRQqueryCDF( c::Vector{T},
                                    X::Vector{Vector{T}},
                                    a::T,
                                    α::T,
                                    t::T2,
                                    v)::T2 where {T,T2}
    #
    N = length(c)
    @assert length(X) == N

    #v = x[1:end-1]

    out = zero(T)
    for n = 1:N
        out += c[n]*evalCDFnonadaptiveRQ(v, X[n], a, α, t)
    end

    return out*sqrt(a)^3
end

function evalnonadaptiveRQqueryCDF( c::Vector{T},
                                    X::Vector{Vector{T}},
                                    a::T,
                                    α::T,
                                    x::Vector{T2})::T2 where {T,T2}
    #
    return evalnonadaptiveRQqueryCDF(c, X, a, α, x[end], x[1:end-1])
end

function eval∂w∂viquery( c::Vector{T},
                            X::Vector{Vector{T}},
                            a::T,
                            α::T,
                            x::Vector{T2},
                            i::Int)::T2 where {T,T2}
    #
    return eval∂w∂viquery(c, X, a, α, x[end], i, x[1:end-1])
end

function eval∂w∂viquery( c::Vector{T},
                            X::Vector{Vector{T}},
                            a::T,
                            α::T,
                            t::T2,
                            i::Int,
                            v)::T2 where {T,T2}
    #
    N = length(c)
    @assert length(X) == N

    #v = x[1:end-1]

    out = zero(T)
    for n = 1:N
        out += c[n]*eval∂w∂vi(v, X[n], a, α, t, i)
    end

    return out*sqrt(a)^3
end


function eval∂2w∂vijquery( c::Vector{T},
                            X::Vector{Vector{T}},
                            a::T,
                            α::T,
                            x::Vector{T2},
                            i::Int,
                            j::Int)::T2 where {T,T2}
    #
    return eval∂2w∂vijquery(c, X, a, α, x[end], i, j, x[1:end-1])
end

function eval∂2w∂vijquery( c::Vector{T},
                            X::Vector{Vector{T}},
                            a::T,
                            α::T,
                            t::T2,
                            i::Int,
                            j::Int,
                            v)::T2 where {T,T2}
    #
    N = length(c)
    @assert length(X) == N

    #v = x[1:end-1]

    out = zero(T)
    for n = 1:N
        out += c[n]*eval∂2w∂vij(v, X[n], a, α, t, i, j)
    end

    return out*sqrt(a)^3
end

##### RQ kernel.

# without the multiple sqrt(a)^3.
function evalnonadaptiveRQ(x, z::Vector{T}, a::T) where T <: Real
    r = x-z
    return 1/sqrt(a + dot(r,r))^3
end

function evalCDFnonadaptiveRQnumerically(   x::Vector{T},
                                            z::Vector{T},
                                            a::T,
                                            α::T)::T where T
    #
    v = x[1:end-1]
    f = tt->evalnonadaptiveRQ([v; tt], z, a)

    return evalintegral(f, α, x[end])
end

# drop this later to cached bv.
function getbv(v::Vector{T2}, a_RQ::T, z::Vector{T})::T2 where {T,T2}
    D = length(z)
    @assert length(v) == D-1

    b = a_RQ
    for d = 1:D-1
        b += (v[d]-z[d])^2
    end

    return b
end

function getbwithoutj(v::Vector{T2}, a_RQ::T, z::Vector{T}, j::Int)::T2 where {T,T2}

    if length(v) == 0
        return a_RQ
    end

    @assert j <= length(v)

    b = getbv(v, a_RQ, z) - (v[j]-z[j])^2

    return b
end

# integrates the last dim from α to β.
#  this is w in notes.
# for AD.
function evalCDFnonadaptiveRQ(  v,
                                z::Vector{T},
                                a::T,
                                α::T,
                                β::T2)::T2 where {T,T2}
    #
    D = length(z)
    @assert length(v) == D-1

    b = getbv(v, a, z)

    τ_β = β-z[end]
    τ_α = α-z[end]

    return (τ_β/sqrt(b + τ_β^2) - τ_α/sqrt(b + τ_α^2))/b
end

## for verifying derivatives. For AD
function evalCDFnonadaptiveRQ(  z::Vector{T},
                                a::T,
                                α::T,
                                x::Vector{T2})::T2 where {T,T2}
    #
    return evalCDFnonadaptiveRQ(x[1:end-1], z, a, α, x[end])
end

"""
aD = a[d] in notes. It is the lower bound of the CDF integral.
xD = x[d] in notes. It is the upper bound.
"""
function eval∂w∂vi( v::Vector{T},
                    z::Vector{T},
                    a::T,
                    aD::T,
                    xD::T,
                    i::Int )::T where T <: Real
    #
    D = length(z)
    @assert length(v) == D-1

    b = getbv(v, a, z)
    A = xD-z[D]
    B = aD - z[D]

    #
    numerator1 = -A*(2*A^2 + 3*b)
    denominator1 = 2*b^2 * sqrt(A^2 + b)^3

    numerator2 = B*(2*B^2 + 3*b)
    denominator2 = 2*b^2 * sqrt(B^2 + b)^3

    ∂w_∂b = numerator1/denominator1 + numerator2/denominator2

    ∂b_∂vi = 2*(v[i]-z[i])

    return ∂w_∂b * ∂b_∂vi
end



# wolfram command for ∂2w_∂b2:
# derivative of -A*(2*A^2+3*b)/(2*b^2*sqrt(A^2+b)^3) + B*(2*B^2+3*b)/(2*b^2*sqrt(B^2+b)^3) with respect to b
#
function eval∂2w∂vij(   v::Vector{T},
                        z::Vector{T},
                        a::T,
                        aD::T,
                        xD::T,
                        i::Int,
                        j::Int )::T where T <: Real

    # set up.
    b = getbv(v, a, z)
    A = xD - z[end]
    B = aD - z[end]

    # first-order.
    numerator1 = -A*(2*A^2 + 3*b)
    denominator1 = 2*b^2 * sqrt(A^2 + b)^3

    numerator2 = B*(2*B^2 + 3*b)
    denominator2 = 2*b^2 * sqrt(B^2 + b)^3

    ∂w_∂b = numerator1/denominator1 + numerator2/denominator2

    ∂b_∂vi = 2*(v[i]-z[i])
    ∂b_∂vj = 2*(v[j]-z[j])

    # second-order.
    term1 = (A*(8*A^4 + 20*A^2 *b + 15 *b^2))/(4 *b^3 *sqrt(A^2 + b)^5)
    term2 = -(B *(15 *b^2 + 20 *b *B^2 + 8 *B^4))/(4 *b^3 *sqrt(b + B^2)^5)

    ∂2w_∂b2 = term1 + term2

    ∂2b_∂vij = zero(T)
    if i == j
        ∂2b_∂vij = convert(T, 2)
    end

    # put it together.
    out = ∂w_∂b *∂2b_∂vij + ∂2w_∂b2 *∂b_∂vi *∂b_∂vj

    return out
end
