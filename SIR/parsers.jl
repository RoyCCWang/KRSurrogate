
#####

"""
parser1( vars::Vector{T},
         constants::Vector{T})

vars = [k, β, γ]
constants = [N, R_0, S_0, I_0]
output order: k, I_0, S_0, R_0, β, γ, N.
"""
function parser1( vars::Vector{T},
                  constants::Vector{T})::Tuple{T,T,T,T,T,T,T} where T <: Real
    #
    k, β, γ = vars
    N, R_0, S_0, I_0 = constants

    return k, I_0, S_0, R_0, β, γ, N
end

"""
parser2( vars::Vector{T},
         constants::Vector{T})

vars = [β, γ]
constants = [k, N, R_0, S_0, I_0]
output order: k, I_0, S_0, R_0, β, γ, N.
"""
function parser2( vars::Vector{T},
                  constants::Vector{T} )::Tuple{T,T,T,T,T,T,T} where T <: Real
    #
    β, γ = vars
    k, N, R_0, S_0, I_0 = constants

    return k, I_0, S_0, R_0, β, γ, N
end

"""
parserfull( vars::Vector{T},
            constants::Vector{T})

vars = [k, I_0, S_0, R_0, β, γ, N]
constants is not used.
output order: k, I_0, S_0, R_0, β, γ, N.
"""
function parserfull( vars::Vector{T},
                     constants)::Tuple{T,T,T,T,T,T,T} where T <: Real
    #
    return tuple(vars...)
end
