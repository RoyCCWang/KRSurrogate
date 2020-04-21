function evalunivariatenormalpdf(x::T, μ::T, σ::T)::T where T <: Real
    return exp(-0.5*((x-μ)/σ)^2)/(σ*sqrt(2*π))
end

# derivaitve wrt x.
function evalderivativeunivariatenormalpdf(x::T, μ::T, σ::T)::T where T <: Real
    r = x-μ
    return (-r/(sqrt(2*π)*σ^3))*exp(-0.5*(r/σ)^2)
end



# cdf( randn() ), cdf is that of the standard normal.
function drawstdnormalcdf()::Float64
    return 0.5*( 1.0 + SpecialFunctions.erf(randn()/sqrt(2)) )
end

function drawstdnormalcdf!(x::Vector{T}) where T <: Real
    for i = 1:length(x)
        x[i] = convert(T, drawstdnormalcdf())
    end

    return nothing
end

function computestdnormalcdf(x::T)::T where T <: Real

    return 0.5*( 1.0 + SpecialFunctions.erf(x/sqrt(2)) )
end

"""
evalunivariatenormalcdf(x::T, μ::T, σ::T)::T
"""
function evalunivariatenormalcdf(x::T, μ::T, σ::T)::T where T <: Real
    z = (x-μ)/σ
    return computestdnormalcdf(z)
end

"""
Evaluate univariate normal pdf, and put them in a Vector{T}.
"""
function collect1Dnormals(   x::Vector{T},
                            μ::Vector{T},
                            σ_array::Vector{T})::Vector{T} where T <: Real
    #
    D = length(x)

    out = Vector{T}(undef, D)
    for d = 1:D
        out[d] = evalunivariatenormalpdf(x[d], μ[d], σ_array[d])
    end

    return out
end

# this is collect1Dnormals(), but stores the output in out.
function collect1Dnormalsbundled!( out::Vector{T},
                            x::Vector{T},
                            μ::Vector{T},
                            σ_array::Vector{T})::Nothing where T <: Real
    #
    D = length(x)

    resize!(out, D)
    for d = 1:D
        out[d] = evalunivariatenormalpdf(x[d], μ[d], σ_array[d])
    end

    return nothing
end

"""
Evaluate derivatives of univariate normal pdf, and put them in a Vector{Matrix{T}}.
"""
function evald2CDFisonormal(   x::Vector{T},
                            μ::Vector{T},
                            σ_array::Vector{T})::Vector{Matrix{T}} where T <: Real
    #
    D = length(x)

    out = Vector{Matrix{T}}(undef, D)
    for d = 1:D
        out[d] = zeros(T, D, D)
        out[d][d,d] = evalderivativeunivariatenormalpdf(x[d], μ[d], σ_array[d])
    end

    return out
end
