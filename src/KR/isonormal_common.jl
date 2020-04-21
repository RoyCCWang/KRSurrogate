
function transportisonormaltoCDFspace(  x::Vector{T},
                            μ::Vector{T},
                            σ_array::Vector{T})::Vector{T} where T <: Real
    #
    D = length(x)
    @assert length(μ) == length(σ_array)

    out = Vector{T}(undef, D)

    transportisonormaltoCDFspace!(out, x, μ, σ_array)

    return out
end

"""
Transport x ~ 𝓝(μ, Σ) to [0,1]^D via CDF.
Σ is diagonal, with elements σ_array.^2.
"""
function transportisonormaltoCDFspace!(  out::Vector{T},
                            x::Vector{T},
                            μ::Vector{T},
                            σ_array::Vector{T}) where T <: Real
    #
    D = length(x)
    resize!(out,D)

    for d = 1:D
        out[d] = clamp( evalunivariatenormalcdf(x[d], μ[d], σ_array[d]),
                        zero(T), one(T) )
    end

    return nothing
end

function evald2Tmap(x::Vector{T},
                    y::Vector{T},
                    dCDF::Function,
                    d2CDF::Function,
                    d2Q_via_y::Function,
                    dQ_via_y::Function)::Vector{Matrix{T}} where T <: Real

    d2Q_u = d2Q_via_y(y)
    dQ_u = dQ_via_y(y)

    d2CDF_x = d2CDF(x)
    dCDF_x = diagm(dCDF(x))

    d2T_x = applychainruled2(dCDF_x, d2CDF_x,
                                 dQ_u, d2Q_u)

    return d2T_x
end

function evald2Q( y::Vector{T},
                  d2g_array::Vector{Function},
                  dQ_u_mat::Matrix{T})::Vector{Matrix{T}} where T <: Real
    #
    D = length(y)
    @assert length(d2g_array) == D
    @assert size(dQ_u_mat, 1) == D == size(dQ_u_mat, 2)

    d2Qinv_y = collect( d2g_array[d](y[d]) for d = 1:D )
    ##dQ_u_mat = dQ_via_y(y)

    # # # debug.
    # println("dQ_u_mat = ")
    # display(dQ_u_mat)
    # println("d2Qinv_y = ")
    # display(d2Qinv_y)
    # @assert 1==2
    d2Q_u = applyinvfuncTHMd2lowertriangular(d2Qinv_y, dQ_u_mat)

    return d2Q_u
end

function evald2Q( y::Vector{T},
                  d2g_array::Vector{Function},
                  dQ_via_y::Function)::Vector{Matrix{T}} where T <: Real
    #
    D = length(y)
    @assert length(d2g_array) == D

    d2Qinv_y = collect( d2g_array[d](y[d]) for d = 1:D )
    dQ_u_mat = dQ_via_y(y)
    d2Q_u = applyinvfuncTHMd2lowertriangular(d2Qinv_y, dQ_u_mat)

    return d2Q_u
end

function computedQviay( y::Vector{T},
                        dg_array::Vector{Function},
                        updatedfbuffer_array::Vector{Function})::Matrix{T} where T
    #
    D = length(y)

    dg_y = zeros(T, D, D)
    for i = 1:D
        updatedfbuffer_array[i](y[1:i])

        dg_y[i, 1:i] = dg_array[i](y[i])
    end

    return inv(dg_y)
end
