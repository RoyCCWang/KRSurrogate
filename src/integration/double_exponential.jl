### DE integration.

function tanhsinhtransform(x::T)::T where T <: Real
    half_π = π/(one(T)+one(T))

    g_x = tanh(half_π*sinh(x))

    return g_x
end

# derivative of tanh(pi/2*sinh(x))
function derivativetanhsinhtransform(x::T) where T <: Real
    half_π = π/(one(T)+one(T))

    dg_x = half_π *cosh(x)*sech(half_π*sinh(x))^2

    return dg_x
end

# returns positions, weights. Not working.
function gettanhsinhquad(N::Int, m::Int, dummy::T)::Tuple{Vector{T},Vector{T}} where T <: Real

    j_range = -N:N
    h = one(T)/convert(T, 2^m)

    x = collect( tanhsinhtransform(h*j) for j in j_range )
    w = collect( derivativetanhsinhtransform(h*j) for j in j_range )

    return x, w
end

function preparextsq(x_tsq::Vector{T}, c::T, d::T)::Vector{T} where T <: Real
    @assert d > c
    N = length(x_tsq)
    α::T = (d-c)/2

    out = Vector{T}(undef,N)
    for i = 1:length(x_tsq)
        out[i] = (x_tsq[i]+1)*α + c
    end

    return out
end

# Mutates CDF_evals and f_evals.
# x contain the TSQ nodes.
# returns the index for x that corresponds to a rough guess of the
#   quantile evaluated at y.
# f is the probability density function that corresponds to the quantile.
function estimatequantileviatsq!(CDF_evals::Vector{T},
                                f_evals::Vector{T},
                                y::T,
                                x::Vector{T},
                                w::Vector{T},
                                f::Function,
                                m::Int,
                                c::T,
                                d::T)::T where T <: Real
    @assert d > c

    N = length(w)

    h::T = (d-c)/(2^(m+1))
    #tmp::T = (d-c)/2

    resize!(f_evals, N)
    for i = 1:N
        #u = (x[i]+1)*tmp + c
        f_evals[i] = w[i]*f(x[i])*h
    end
    Z = sum(f_evals) # normalize.

    resize!(CDF_evals, N)
    cumsum!(CDF_evals, f_evals)
    for i = 1:N
        CDF_evals[i] = CDF_evals[i]/Z
    end

    i_upper = findfirst( xx -> xx>y, CDF_evals)

    if typeof(i_upper) == Int
        return x[i_upper]
    end

    return x[1]
end
